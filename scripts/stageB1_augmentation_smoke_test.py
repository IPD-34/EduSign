"""
scripts/stageB1_augmentation_smoke_test.py

Stage B1 smoke test: spatial-only landmark augmentation (src/models/
augmentation.py), isolated -- NOT combined with Stage A's ruled-out
levers (blank_logit_bias=0.0, boundary_loss_weight=1.0, both left at
plain defaults) so any effect found is attributable to augmentation
alone, same discipline as Stage A's own isolation of its two levers.

Two configs, otherwise identical:
  control    -- train dataset built with augment=False (default)
  augmented  -- train dataset built with augment=True

val/test datasets are ALWAYS built with augment=False (the class
default) in both configs -- augmentation is a training-time-only
regularizer; evaluating on artificially-perturbed data would measure
robustness-to-noise, not the actual question (does training on
augmented data improve performance on REAL, clean held-out data).

500 steps, seeded (train_seed=155, reused for DataLoader shuffle order
AND augmentation's own randomness isn't separately seeded -- augment.py
draws from the global default_rng() each call, matching its per-epoch-
freshness design; torch.manual_seed here only fixes batch ORDER, not
augmentation noise, which is intentionally left free-running per call,
same as it would be in a real run). Checkpoints saved every 100 steps
to checkpoints/stageB1_smoke/<config>/step_XXXX.pt, standard practice
per the Stage A blank-bias diagnostic's finding that un-persisted smoke
tests can't be diagnosed after the fact.

--- WHAT'S TRACKED ---
  - ctc_loss / boundary_loss trajectories (cheap, every 100 steps)
  - a REAL WER probe (not raw loss) on two small FIXED subsets --
    15 train samples (seed=155) and 15 val samples (seed=156), at
    steps {0, 250, 500} only (WER needs beam search -- the expensive
    part -- so probed less often than the cheap loss numbers, cheap
    discipline maintained). Reuses compute_wer_beam_search's
    ctc_prefix_beam_search/edit_distance UNCHANGED -- same beam_width/
    expand_top_k defaults as every other WER number this project has
    reported, so this IS comparable to Day 51/52's numbers in kind
    (not scale -- 15 samples is a smoke-test probe, not a real split).
  - the train-vs-val WER GAP at each of those 3 steps -- the actual
    thing augmentation is meant to move, per Day 52's finding
    (WER 0.892 train vs 0.959 test on the real, full splits).
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader

from src.models.sign_transformer import SignTransformer
from src.models.train_utils import load_checkpoint_partial, save_checkpoint, get_device
from src.models.dataset import SignLanguageDataset, collate_fn
from src.models.combined_loss import compute_combined_loss_with_gate
from scripts.compute_wer_beam_search import ctc_prefix_beam_search, edit_distance

VOCAB_MERGED_PATH = "data/vocab_merged.json"
DECODER_VOCAB_PATH = "data/vocab.json"
ASLLRP_TOKEN_TO_ID_PATH = "data/asllrp_token_to_id.json"
TRAIN_NPZ_DIR = "data/processed/asllrp/train"
VAL_NPZ_DIR = "data/processed/asllrp/val"
RESUME_CHECKPOINT = "checkpoints/month2_asllrp_gated/step_0008000.pt"  # current checkpoint of record

BATCH_SIZE = 4
LEARNING_RATE = 1e-4
GRAD_CLIP_MAX_NORM = 1.0
GATE_REG_WEIGHT = 0.0
BOUNDARY_LOSS_WEIGHT = 1.0   # Stage A's ruled-out lever, left at plain default -- isolation
BLANK_LOGIT_BIAS = 0.0       # Stage A's ruled-out lever, left at plain default -- isolation
NUM_STEPS = 500
LOSS_PROBE_EVERY_N_STEPS = 100
# Rerun (follow-up to the first B1 smoke test): 3 points (0/250/500) at
# n=15 gave a direction that FLIPPED between step 250 and 500 -- not
# enough to distinguish a real trend from noise. Widened to all 6
# loss-probe points (0/100/200/300/400/500) so a real trend, if one
# exists, has to show up as a consistent direction across several
# points, not be inferred from two endpoints.
WER_PROBE_STEPS = {0, 100, 200, 300, 400, 500}
BEAM_WIDTH = 10
EXPAND_TOP_K = 20

TRAIN_SEED = 155           # batch/shuffle order -- identical for both configs, UNCHANGED from the first run
TRAIN_WER_PROBE_SEED = 155
VAL_WER_PROBE_SEED = 156
# Widened from 15 -- the previous run's per-point noise floor (roughly
# 1/(total ref tokens) per single-utterance edit, ~0.01-0.013 at n=15)
# was large enough to plausibly explain the observed 0.04-0.06 swings by
# itself. 40 (roughly 2.7x) meaningfully cuts that floor while staying
# inside smoke-test compute budget (80 beam-search decodes per probe
# point per config, vs. Day 51's real 122-sample full-split runs) --
# went with 40 over the top of the stated 30-50 range specifically to
# get a clearly-smaller floor than the swings already observed, not
# just "somewhat smaller."
WER_PROBE_SIZE = 40
CHECKPOINT_SAVE_DIR = Path("checkpoints/stageB1_smoke_rerun")

CONFIGS = [
    ("control", False),
    ("augmented", True),
]


def pick_wer_probe_indices(npz_dir, seed, size):
    dataset = SignLanguageDataset(
        npz_dir=npz_dir, vocab_path=VOCAB_MERGED_PATH, asllrp_token_to_id_path=ASLLRP_TOKEN_TO_ID_PATH,
        augment=False,  # probing is ALWAYS on clean data, regardless of which config is training
    )
    import random
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(dataset)), min(size, len(dataset))))
    return dataset, indices


@torch.no_grad()
def wer_probe(model, probe_dataset, probe_indices, device, blank_id):
    model.eval()
    total_edits, total_ref = 0, 0
    for idx in probe_indices:
        item = probe_dataset[idx]
        landmarks = item["landmarks"].unsqueeze(0).to(device)
        T = landmarks.shape[1]
        mask = torch.ones(1, T, dtype=torch.bool, device=device)
        true_tokens = item["target_tokens"].tolist()
        true_ids = true_tokens[1:-1]

        ctc_log_probs, _bl, _gl = model.forward_ctc_boundary_and_gate(landmarks, mask)
        log_probs = ctc_log_probs[0].cpu()
        decoded_ids = ctc_prefix_beam_search(log_probs, blank_id, BEAM_WIDTH, EXPAND_TOP_K)
        total_edits += edit_distance(true_ids, decoded_ids)
        total_ref += len(true_ids)

    model.train()
    return total_edits / max(total_ref, 1)


def run_one_config(name, augment, train_probe_ds, train_probe_idx, val_probe_ds, val_probe_idx,
                    device, ctc_vocab_size, decoder_vocab_size, blank_id, bos_id, eos_id, pad_id):
    print(f"\n{'='*90}\nCONFIG: {name}  (augment={augment})\n{'='*90}")

    torch.manual_seed(TRAIN_SEED)

    model = SignTransformer(
        vocab_size=decoder_vocab_size, ctc_vocab_size=ctc_vocab_size, use_bimanual_gate=True,
    ).to(device)
    loaded, skipped, new_only = load_checkpoint_partial(
        model, RESUME_CHECKPOINT,
        prefixes_to_load=("encoder.", "ctc_head.", "boundary_head.", "bimanual_gate_head."),
    )
    print(f"  Checkpoint load: {len(loaded)} loaded, {len(skipped)} skipped, {len(new_only)} new-only")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    train_dataset = SignLanguageDataset(
        npz_dir=TRAIN_NPZ_DIR, vocab_path=VOCAB_MERGED_PATH, asllrp_token_to_id_path=ASLLRP_TOKEN_TO_ID_PATH,
        augment=augment,
    )
    shuffle_generator = torch.Generator().manual_seed(TRAIN_SEED)
    loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn,
        shuffle=True, num_workers=0, generator=shuffle_generator,
    )

    ckpt_dir = CHECKPOINT_SAVE_DIR / name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    probe_log = []

    def do_wer_probe(step):
        train_wer = wer_probe(model, train_probe_ds, train_probe_idx, device, blank_id)
        val_wer = wer_probe(model, val_probe_ds, val_probe_idx, device, blank_id)
        gap = val_wer - train_wer
        probe_log.append({"step": step, "train_wer": train_wer, "val_wer": val_wer, "gap": gap})
        print(f"  [WER PROBE step={step}] train_wer={train_wer:.4f}  val_wer={val_wer:.4f}  gap(val-train)={gap:+.4f}")

    do_wer_probe(0)

    ctc_losses, boundary_losses = [], []
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
            total_loss, ctc_loss, boundary_loss, _ge = compute_combined_loss_with_gate(
                ctc_log_probs, boundary_logits, gate_logits,
                batch["target_tokens"], batch["boundary_target"],
                batch["encoder_attention_mask"],
                bos_id, eos_id, pad_id, blank_id,
                boundary_loss_weight=BOUNDARY_LOSS_WEIGHT,
                gate_reg_weight=GATE_REG_WEIGHT,
                blank_logit_bias=BLANK_LOGIT_BIAS,
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
            step += 1

            if step % LOSS_PROBE_EVERY_N_STEPS == 0 or step == NUM_STEPS:
                recent_ctc = sum(ctc_losses[-LOSS_PROBE_EVERY_N_STEPS:]) / len(ctc_losses[-LOSS_PROBE_EVERY_N_STEPS:])
                recent_boundary = sum(boundary_losses[-LOSS_PROBE_EVERY_N_STEPS:]) / len(boundary_losses[-LOSS_PROBE_EVERY_N_STEPS:])
                loss_log.append({"step": step, "mean_ctc_loss": recent_ctc, "mean_boundary_loss": recent_boundary})
                print(f"  [step={step:4d}] mean_ctc_loss={recent_ctc:.4f}  mean_boundary_loss={recent_boundary:.4f}")

                ckpt_path = ckpt_dir / f"step_{step:04d}.pt"
                save_checkpoint(model, optimizer, step, ckpt_path)

            if step in WER_PROBE_STEPS:
                do_wer_probe(step)

    return {
        "name": name, "augment": augment,
        "loss_log": loss_log, "wer_probe_log": probe_log,
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

    train_probe_ds, train_probe_idx = pick_wer_probe_indices(TRAIN_NPZ_DIR, TRAIN_WER_PROBE_SEED, WER_PROBE_SIZE)
    val_probe_ds, val_probe_idx = pick_wer_probe_indices(VAL_NPZ_DIR, VAL_WER_PROBE_SEED, WER_PROBE_SIZE)
    print(f"Train WER probe: {len(train_probe_idx)} samples (seed={TRAIN_WER_PROBE_SEED}) from {TRAIN_NPZ_DIR}")
    print(f"Val WER probe:   {len(val_probe_idx)} samples (seed={VAL_WER_PROBE_SEED}) from {VAL_NPZ_DIR}")

    out_path = "logs/stageB1_smoke_test_rerun_results.json"  # separate from the first run's file -- preserve history
    results = []
    for name, augment in CONFIGS:
        r = run_one_config(name, augment, train_probe_ds, train_probe_idx, val_probe_ds, val_probe_idx,
                            device, ctc_vocab_size, len(decoder_vocab), blank_id, bos_id, eos_id, pad_id)
        results.append(r)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\n\nDone. Results in {out_path}, "
          f"checkpoints under {CHECKPOINT_SAVE_DIR}/<config_name>/")


if __name__ == "__main__":
    main()
