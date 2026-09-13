"""
scripts/stageC_transition_loss_weight_sweep_smoke_test.py

Follow-up to the Stage C transition-loss REAL run (confirmed at full
122-sample scale: every saved checkpoint beat control on both boundary-
F1 criteria, predicted-vs-true blob-count gap shrank 41-54%). That run
used transition_loss_weight=1.0 -- never actually tuned, just copied
boundary_loss_weight's own convention by default. This script checks
whether a different weight extends the gain further, SMOKE-TEST SCALE
ONLY, before committing to anything bigger (WLASL multi-task, self-
supervised pretraining).

ONE variable only: transition_loss_weight in {0.5, 1.0, 2.0, 3.0}.
Everything else -- resume checkpoint, sigma=1.0 (unchanged), seed,
step count, boundary_loss_weight=1.0, blank_logit_bias=0.0, dropout,
weight_decay, no regularization/oversampling -- held fixed and
IDENTICAL across all four runs, same isolation discipline as every
prior stage.

--- RESUME POINT: SAME AS THE ORIGINAL TRANSITION-LOSS REAL RUN ---
checkpoints/month2_asllrp_gated/step_0008000.pt -- per the task's own
explicit instruction, NOT checkpoints/stageB1_real/control/best_by_wer.pt
(which the Stage A/B1 mechanism smoke tests used). This keeps every
weight value's starting point identical to what the already-confirmed
real run itself started from, so this sweep is asking exactly "does a
different weight beat weight=1.0 from the SAME starting point," not
conflating that question with a different resume point.

--- WEIGHT=1.0 RERUN FRESH, NOT REUSED FROM THE EARLIER MECHANISM
SMOKE TEST ---
The earlier stageB1_boundary_mechanisms_smoke_test.py's
transition_loss_alone config also used weight=1.0, sigma=1.0, and
1000 steps -- but it resumed from a DIFFERENT checkpoint
(stageB1_real/control/best_by_wer.pt) and used a different seed
(TRAIN_SEED=160 there vs 161 here). Those numbers are not directly
comparable to this sweep's 0.5/2.0/3.0 runs (different starting
point AND different batch order), so weight=1.0 is rerun HERE, in
this exact script, with the exact same seed/infra/probe as the other
three weights -- a true apples-to-apples four-way comparison, not a
mix of old and new numbers.

--- WHAT'S TRACKED, PER TASK ---
Same probe machinery as the mechanism smoke test (imported, not
reimplemented): predicted-vs-true blob count and BOTH boundary-F1
criteria (strict IoU, loose center-tolerance) on the fixed 30-sample
val probe (seed=157). PLUS, per this task's own specific ask: ctc_loss
tracked BOTH ways -- cheap mean training ctc_loss every 100 steps
(noisy, per-batch) AND a held-out ctc_loss probe on the SAME fixed
val subset at each probe step (less noisy, the more trustworthy
signal for "is a higher weight starting to fight the recognition
objective"). No WER/beam-search decoding here -- not asked for by
this task and not needed to answer it; the val-probe ctc_loss is the
right cheap proxy for the specific tradeoff question being asked.
"""
import sys
import json
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader

from src.models.sign_transformer import SignTransformer
from src.models.train_utils import load_checkpoint_partial, save_checkpoint, get_device
from src.models.dataset import SignLanguageDataset, collate_fn
from src.models.combined_loss import compute_combined_loss_with_gate_and_transition
from scripts.stageB1_boundary_mechanisms_smoke_test import (
    pick_fixed_indices, probe_ctc_loss, probe_blob_count_and_boundary_f1,
)

VOCAB_MERGED_PATH = "data/vocab_merged.json"
DECODER_VOCAB_PATH = "data/vocab.json"
ASLLRP_TOKEN_TO_ID_PATH = "data/asllrp_token_to_id.json"
TRAIN_NPZ_DIR = "data/processed/asllrp/train"
VAL_NPZ_DIR = "data/processed/asllrp/val"
RESUME_CHECKPOINT = "checkpoints/month2_asllrp_gated/step_0008000.pt"  # same as the original transition-loss REAL run

BATCH_SIZE = 4
LEARNING_RATE = 1e-4
GRAD_CLIP_MAX_NORM = 1.0
GATE_REG_WEIGHT = 0.0
BOUNDARY_LOSS_WEIGHT = 1.0   # held fixed -- isolation
BLANK_LOGIT_BIAS = 0.0       # held fixed -- isolation
DROPOUT = 0.1                # model default, unchanged -- regularization is not part of this sweep
WEIGHT_DECAY = 0.01          # AdamW default, unchanged
NUM_STEPS = 1000
LOSS_PROBE_EVERY_N_STEPS = 100
PROBE_STEPS = {0, 200, 400, 600, 800, 1000}
CHECKPOINT_SAVE_DIR = Path("checkpoints/stageC_transition_loss_weight_sweep")

TRAIN_SEED = 161  # fixed across ALL FOUR weights -- identical batch order, weight is the only variable

VAL_PROBE_SEED = 157
VAL_PROBE_SIZE = 30

WEIGHTS = [0.5, 1.0, 2.0, 3.0]


def run_one_weight(weight, device, ctc_vocab_size, decoder_vocab_size, blank_id, bos_id, eos_id, pad_id,
                    val_probe_idx):
    label = f"weight_{weight}"
    print(f"\n{'='*90}\nCONFIG: transition_loss_weight={weight}\n{'='*90}")
    torch.manual_seed(TRAIN_SEED)

    model = SignTransformer(
        vocab_size=decoder_vocab_size, ctc_vocab_size=ctc_vocab_size, use_bimanual_gate=True,
        dropout=DROPOUT,
    ).to(device)
    loaded, skipped, new_only = load_checkpoint_partial(
        model, RESUME_CHECKPOINT,
        prefixes_to_load=("encoder.", "ctc_head.", "boundary_head.", "bimanual_gate_head."),
    )
    print(f"  Checkpoint load: {len(loaded)} loaded, {len(skipped)} skipped, {len(new_only)} new-only "
          f"(from {RESUME_CHECKPOINT})")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    train_dataset = SignLanguageDataset(
        npz_dir=TRAIN_NPZ_DIR, vocab_path=VOCAB_MERGED_PATH, asllrp_token_to_id_path=ASLLRP_TOKEN_TO_ID_PATH,
        augment=False, include_transition_target=True,
    )
    val_probe_dataset = SignLanguageDataset(
        npz_dir=VAL_NPZ_DIR, vocab_path=VOCAB_MERGED_PATH, asllrp_token_to_id_path=ASLLRP_TOKEN_TO_ID_PATH,
    )

    shuffle_generator = torch.Generator().manual_seed(TRAIN_SEED)
    loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn,
                         shuffle=True, num_workers=0, generator=shuffle_generator)

    ckpt_dir = CHECKPOINT_SAVE_DIR / label
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    probe_log = []

    def do_probe(step):
        entry = {"step": step}
        entry["val_ctc_loss"] = probe_ctc_loss(model, val_probe_dataset, val_probe_idx, device, blank_id)
        entry["val_blob_f1"] = probe_blob_count_and_boundary_f1(model, val_probe_dataset, val_probe_idx, device)
        probe_log.append(entry)
        bf = entry["val_blob_f1"]
        print(f"  [PROBE step={step}] val_ctc_loss={entry['val_ctc_loss']:.4f} | "
              f"pred_blobs={bf['mean_pred_blob_count']:.2f} true_blobs={bf['mean_true_blob_count']:.2f} "
              f"iouF1={bf['iou']['f1']:.4f} (P={bf['iou']['precision']:.4f} R={bf['iou']['recall']:.4f}) "
              f"ctF1={bf['center_tolerance']['f1']:.4f} (P={bf['center_tolerance']['precision']:.4f} R={bf['center_tolerance']['recall']:.4f})")

    do_probe(0)

    ctc_losses, boundary_losses, transition_losses = [], [], []
    loss_log = []
    step = 0
    model.train()
    while step < NUM_STEPS:
        for batch in loader:
            if step >= NUM_STEPS:
                break
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

            ctc_log_probs, boundary_logits, gate_logits = model.forward_ctc_boundary_and_gate(
                batch["landmarks"], batch["encoder_attention_mask"]
            )
            total_loss, ctc_loss, boundary_loss, _ge, transition_loss = compute_combined_loss_with_gate_and_transition(
                ctc_log_probs, boundary_logits, gate_logits,
                batch["target_tokens"], batch["boundary_target"],
                batch["encoder_attention_mask"],
                bos_id, eos_id, pad_id, blank_id,
                boundary_loss_weight=BOUNDARY_LOSS_WEIGHT,
                gate_reg_weight=GATE_REG_WEIGHT,
                blank_logit_bias=BLANK_LOGIT_BIAS,
                transition_target=batch["transition_target"],
                transition_loss_weight=weight,
            )

            if not torch.isfinite(total_loss):
                print(f"  [WARNING] step {step}: non-finite loss, skipping batch")
                step += 1
                continue

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_MAX_NORM)
            optimizer.step()

            ctc_losses.append(ctc_loss.item())
            boundary_losses.append(boundary_loss.item())
            transition_losses.append(transition_loss.item())
            step += 1

            if step % LOSS_PROBE_EVERY_N_STEPS == 0 or step == NUM_STEPS:
                w = LOSS_PROBE_EVERY_N_STEPS
                recent_ctc = sum(ctc_losses[-w:]) / len(ctc_losses[-w:])
                recent_boundary = sum(boundary_losses[-w:]) / len(boundary_losses[-w:])
                recent_transition = sum(transition_losses[-w:]) / len(transition_losses[-w:])
                loss_log.append({"step": step, "mean_ctc_loss": recent_ctc, "mean_boundary_loss": recent_boundary,
                                  "mean_transition_loss": recent_transition})
                print(f"  [step={step:4d}] mean_ctc_loss={recent_ctc:.4f} mean_boundary_loss={recent_boundary:.4f} "
                      f"mean_transition_loss={recent_transition:.4f}")
                save_checkpoint(model, optimizer, step, ckpt_dir / f"step_{step:04d}.pt")

            if step in PROBE_STEPS:
                do_probe(step)

    return {
        "weight": weight, "loss_log": loss_log, "probe_log": probe_log,
        "checkpoint_dir": str(ckpt_dir), "train_seed": TRAIN_SEED,
    }


def main():
    device = get_device()
    with open(VOCAB_MERGED_PATH) as f:
        vocab_merged = json.load(f)
    with open(DECODER_VOCAB_PATH) as f:
        decoder_vocab = json.load(f)
    ctc_vocab_size = len(vocab_merged)
    blank_id = ctc_vocab_size
    bos_id, eos_id, pad_id = vocab_merged["<BOS>"], vocab_merged["<EOS>"], vocab_merged["<PAD>"]

    val_probe_idx = pick_fixed_indices(VAL_NPZ_DIR, VAL_PROBE_SEED, VAL_PROBE_SIZE)
    print(f"Val probe: {len(val_probe_idx)} samples (seed={VAL_PROBE_SEED})")
    print(f"Weights to sweep: {WEIGHTS}  (train_seed={TRAIN_SEED}, identical batch order across all four)")

    out_path = "logs/stageC_transition_loss_weight_sweep_results.json"
    results = []
    for weight in WEIGHTS:
        r = run_one_weight(weight, device, ctc_vocab_size, len(decoder_vocab), blank_id, bos_id, eos_id, pad_id,
                            val_probe_idx)
        results.append(r)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\n\nDone. Results in {out_path}, checkpoints under {CHECKPOINT_SAVE_DIR}/weight_<W>/")


if __name__ == "__main__":
    main()
