"""
scripts/stageB1_boundary_mechanisms_smoke_test.py

Follow-up to the boundary-F1 strictness-vs-genuine-failure diagnostic
(compute_boundary_f1.py's dual-criteria run on control/best_by_wer.pt):
that diagnostic found a real, non-metric segmentation problem -- 9/122
samples merge several adjacent signs into one wide predicted blob,
and even the loose center-tolerance criterion leaves most ground-truth
spans unmatched. This script tests THREE new, cheap, isolated
mechanisms aimed at that specific failure mode:

  a. REGULARIZATION       -- higher dropout + weight_decay
  b. TRANSITION-LOSS       -- a second, additive boundary-supervision
                              term at sign start/end frames specifically
                              (src/preprocessing/boundary_labels.py's
                              build_transition_targets(), combined_loss.
                              py's compute_combined_loss_with_gate_and_
                              transition())
  c. OVERSAMPLING          -- upweighted sampling of dense (>10-gloss)
                              utterances, the population where merged
                              blobs were concentrated

SMOKE-TEST SCALE ONLY -- no real training commitment made here. All
four configs (control + the three mechanisms, each isolated) resume
MODEL WEIGHTS ONLY (load_checkpoint_partial, fresh optimizer) from
checkpoints/stageB1_real/control/best_by_wer.pt (step 4000, WER
0.9173) -- the current best-known checkpoint -- same convention every
prior experimental branch in this project has used.

--- STEP COUNT: 1000, not Stage B1's 500 ---
Stage B1's own smoke test (500 steps) gave a result that flipped
direction between measurement points -- explicitly flagged there as
possibly too short to distinguish a real trend from noise. Doubling to
1000 steps (top of the task's suggested 800-1000 range) buys more
distance for a real trend to separate from noise, while staying inside
smoke-test compute budget (this is NOT a real training commitment --
that's an explicit follow-up decision, not made here).

--- REGULARIZATION LEVER CHOICE, WITH REASONING ---
Current model: dropout=0.1 (SignTransformer's own default, confirmed
by inspection -- not assumed). Current optimizer: AdamW's own default
weight_decay=0.01 (confirmed via torch.optim.AdamW's signature -- this
project's training scripts never override it, so 0.01 is what every
prior real run has actually used, not "no regularization").
regularization_alone doubles dropout to 0.2 and raises weight_decay
5x to 0.05. Both land inside the commonly-used range for small (tens
of millions of parameters) from-scratch transformers -- dropout
0.1-0.3 and weight_decay 0.01-0.1 are standard starting bracket values
in that regime (e.g. the original Transformer paper's own dropout=0.1
default, with 0.2-0.3 common for smaller/more overfitting-prone
setups; AdamW weight_decay commonly swept in the 0.01-0.1 decade for
transformer fine-tuning). Not a maximal or arbitrary pick -- a
one-notch-stronger point inside that standard bracket, chosen because
CTC training on ~1000 utterances is exactly the small-data, high-
overfitting-risk regime that bracket targets.

--- TRANSITION-LOSS WEIGHT CHOICE ---
transition_loss_weight=1.0 -- same order of magnitude as
boundary_loss_weight=1.0 (both terms are the same rough scale: BCE
loss for the flat target starts around O(0.6-0.9), MSE loss for the
peaked target against a naive sigmoid~=0.5 starting point is O(0.25);
weighting them equally keeps this an additive SECOND signal at
comparable strength to the first, not one drowning out the other by
construction).

--- OVERSAMPLING THRESHOLD/FACTOR CHOICE, VERIFIED AGAINST REAL DATA ---
Audited the actual train-split gloss-count distribution before
picking anything (990 utterances): median 8, 75th pct 11, 80th pct 12.
The task's suggested >10-gloss threshold sits right at the 75th-80th
percentile boundary -- a real "dense tail," not an arbitrary cut
(294/990 = 29.7% of train utterances clear it). Chose a 3x oversample
factor (top of the task's suggested 2x-3x range, not 2x) specifically
because the diagnostic's failure mode was CONCENTRATED in exactly this
population (every "genuine failure" example inspected had 5-8+
ground-truth signs) -- a mild 2x nudge seemed unlikely to counteract
that concentration; 3x raises dense utterances' share of total
sampling PRESSURE from 29.7% to 55.9% of weight mass (verified
directly, see scratchpad verification run), a real behavioral change,
not a token gesture.

--- WHAT'S TRACKED PER CONFIG ---
All four configs get the CHEAP signals every 100 steps (ctc_loss,
boundary_loss, transition_loss where applicable). Each mechanism ALSO
gets its own specific probe (see module docstring section above this
one and the task's own point 4):
  - regularization_alone: train-probe-loss vs val-probe-loss GAP
    trajectory (fixed 30-sample train probe seed=155, fixed 30-sample
    val probe seed=157 -- same val-probe seed/size convention as
    stageB1_real_training_run.py's own WER probe, for continuity)
  - transition_loss_alone: mean predicted-blob-count vs mean
    true-blob-count, PLUS both boundary-F1 criteria (strict IoU,
    loose center-tolerance), on the fixed 30-sample val probe
    (seed=157)
  - oversampling_alone: the SAME blob-count/dual-F1 probe, but on the
    DENSE-ONLY val subset (all 34 val utterances with >10 glosses --
    the full available population, not a further sample of it, since
    34 is already small) -- this is the specific population the
    mechanism targets
  - control: reconfirms the baseline direction of travel on the same
    probes (transition + blob-count probe, and dense-subset probe),
    for a fair floor to compare all three mechanisms against
"""
import sys
import json
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.models.sign_transformer import SignTransformer
from src.models.train_utils import load_checkpoint_partial, save_checkpoint, get_device
from src.models.dataset import SignLanguageDataset, collate_fn
from src.models.combined_loss import (
    compute_combined_loss_with_gate, compute_combined_loss_with_gate_and_transition,
)
from src.preprocessing.sample_writer import load_sample
from scripts.compute_boundary_f1 import (
    compute_sample_boundary_arrays, boundary_f1_for_sample, boundary_f1_for_sample_center_tolerance,
    extract_intervals, IOU_THRESHOLD, CENTER_TOLERANCE_TIMESTEPS, BOUNDARY_PROB_THRESHOLD,
)

VOCAB_MERGED_PATH = "data/vocab_merged.json"
DECODER_VOCAB_PATH = "data/vocab.json"
ASLLRP_TOKEN_TO_ID_PATH = "data/asllrp_token_to_id.json"
TRAIN_NPZ_DIR = "data/processed/asllrp/train"
VAL_NPZ_DIR = "data/processed/asllrp/val"
RESUME_CHECKPOINT = "checkpoints/stageB1_real/control/best_by_wer.pt"  # current best-known checkpoint

BATCH_SIZE = 4
LEARNING_RATE = 1e-4
GRAD_CLIP_MAX_NORM = 1.0
GATE_REG_WEIGHT = 0.0
BOUNDARY_LOSS_WEIGHT = 1.0   # held fixed -- isolation, same as every prior stage
BLANK_LOGIT_BIAS = 0.0       # held fixed -- isolation, same as every prior stage
NUM_STEPS = 1000
LOSS_PROBE_EVERY_N_STEPS = 100
PROBE_STEPS = {0, 200, 400, 600, 800, 1000}
CHECKPOINT_SAVE_DIR = Path("checkpoints/stageB1_boundary_mechanisms")

TRAIN_SEED = 160  # fresh seed for this stage -- new experimental branch, not a continuation

# Regularization lever
DEFAULT_DROPOUT = 0.1
REGULARIZED_DROPOUT = 0.2
DEFAULT_WEIGHT_DECAY = 0.01
REGULARIZED_WEIGHT_DECAY = 0.05

# Transition-loss lever
TRANSITION_LOSS_WEIGHT = 1.0

# Oversampling lever
OVERSAMPLE_GLOSS_THRESHOLD = 10
OVERSAMPLE_FACTOR = 3.0

TRAIN_LOSS_PROBE_SEED = 155
TRAIN_LOSS_PROBE_SIZE = 30
VAL_PROBE_SEED = 157
VAL_PROBE_SIZE = 30

CONFIGS = ["control", "regularization_alone", "transition_loss_alone", "oversampling_alone"]


def pick_fixed_indices(npz_dir, seed, size, filter_fn=None):
    """filter_fn(meta) -> bool, if given, restricts the population BEFORE
    sampling (used for the dense-only oversampling probe, where the
    entire filtered population -- 34 samples -- is used rather than a
    further sub-sample of it)."""
    paths = sorted(Path(npz_dir).glob("*.npz"))
    if filter_fn is not None:
        keep = [i for i, p in enumerate(paths) if filter_fn(load_sample(p).metadata)]
    else:
        keep = list(range(len(paths)))
    if size is None or size >= len(keep):
        return sorted(keep)
    rng = random.Random(seed)
    return sorted(rng.sample(keep, size))


@torch.no_grad()
def probe_ctc_loss(model, dataset, indices, device, blank_id, use_gate=True):
    """Mean per-token CTC loss on a fixed subset -- the train/val-gap
    signal regularization_alone specifically targets."""
    import torch.nn as nn
    was_training = model.training
    model.eval()
    ctc_loss_fn = nn.CTCLoss(blank=blank_id, zero_infinity=True, reduction="sum")
    total_loss, total_targets = 0.0, 0
    for idx in indices:
        item = dataset[idx]
        landmarks = item["landmarks"].unsqueeze(0).to(device)
        T = landmarks.shape[1]
        mask = torch.ones(1, T, dtype=torch.bool, device=device)
        if use_gate:
            ctc_log_probs, _bl, _gl = model.forward_ctc_boundary_and_gate(landmarks, mask)
        else:
            ctc_log_probs, _bl = model.forward_ctc_and_boundary(landmarks, mask)
        true_ids = item["target_tokens"].tolist()[1:-1]
        log_probs_t = ctc_log_probs.transpose(0, 1).cpu()
        loss = ctc_loss_fn(
            log_probs_t, torch.tensor(true_ids, dtype=torch.long),
            torch.tensor([T]), torch.tensor([len(true_ids)]),
        )
        if torch.isfinite(loss):
            total_loss += loss.item()
            total_targets += len(true_ids)
    if was_training:
        model.train()
    return total_loss / max(total_targets, 1)


@torch.no_grad()
def probe_blob_count_and_boundary_f1(model, dataset, indices, device, use_gate=True):
    """Mean predicted-blob-count vs mean true-blob-count, plus both
    boundary-F1 criteria (strict IoU, loose center-tolerance), on a
    fixed subset. Reuses compute_boundary_f1.py's hand-validated
    machinery UNCHANGED."""
    was_training = model.training
    model.eval()
    pred_blob_counts, true_blob_counts = [], []
    tp_iou = fp_iou = fn_iou = 0
    tp_ct = fp_ct = fn_ct = 0
    for idx in indices:
        item = dataset[idx]
        meta = load_sample(dataset.npz_paths[idx]).metadata
        gt_binary, pred_binary, _stats = compute_sample_boundary_arrays(
            model, item, meta, device, BOUNDARY_PROB_THRESHOLD, use_gate=use_gate
        )
        true_blob_counts.append(len(extract_intervals(gt_binary)))
        pred_blob_counts.append(len(extract_intervals(pred_binary)))

        r_iou = boundary_f1_for_sample(gt_binary, pred_binary, IOU_THRESHOLD)
        tp_iou += r_iou["tp"]; fp_iou += r_iou["fp"]; fn_iou += r_iou["fn"]
        r_ct = boundary_f1_for_sample_center_tolerance(gt_binary, pred_binary, CENTER_TOLERANCE_TIMESTEPS)
        tp_ct += r_ct["tp"]; fp_ct += r_ct["fp"]; fn_ct += r_ct["fn"]

    if was_training:
        model.train()

    def _prf1(tp, fp, fn):
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return p, r, f1

    p_iou, r_iou_, f1_iou = _prf1(tp_iou, fp_iou, fn_iou)
    p_ct, r_ct_, f1_ct = _prf1(tp_ct, fp_ct, fn_ct)
    return {
        "mean_pred_blob_count": float(np.mean(pred_blob_counts)),
        "mean_true_blob_count": float(np.mean(true_blob_counts)),
        "iou": {"precision": p_iou, "recall": r_iou_, "f1": f1_iou},
        "center_tolerance": {"precision": p_ct, "recall": r_ct_, "f1": f1_ct},
    }


def build_oversampling_sampler(dataset, gloss_threshold, oversample_factor, seed):
    weights = []
    for p in dataset.npz_paths:
        meta = load_sample(p).metadata
        n_glosses = len(meta.gloss.split("|"))
        weights.append(oversample_factor if n_glosses > gloss_threshold else 1.0)
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True, generator=generator)


def run_one_config(config_name, device, ctc_vocab_size, decoder_vocab_size, blank_id, bos_id, eos_id, pad_id,
                    train_loss_probe_idx, val_probe_idx, dense_val_probe_idx):
    print(f"\n{'='*90}\nCONFIG: {config_name}\n{'='*90}")
    torch.manual_seed(TRAIN_SEED)

    dropout = REGULARIZED_DROPOUT if config_name == "regularization_alone" else DEFAULT_DROPOUT
    weight_decay = REGULARIZED_WEIGHT_DECAY if config_name == "regularization_alone" else DEFAULT_WEIGHT_DECAY
    use_transition_loss = config_name == "transition_loss_alone"
    use_oversampling = config_name == "oversampling_alone"

    model = SignTransformer(
        vocab_size=decoder_vocab_size, ctc_vocab_size=ctc_vocab_size, use_bimanual_gate=True,
        dropout=dropout,
    ).to(device)
    loaded, skipped, new_only = load_checkpoint_partial(
        model, RESUME_CHECKPOINT,
        prefixes_to_load=("encoder.", "ctc_head.", "boundary_head.", "bimanual_gate_head."),
    )
    print(f"  Checkpoint load: {len(loaded)} loaded, {len(skipped)} skipped, {len(new_only)} new-only "
          f"(from {RESUME_CHECKPOINT})")
    print(f"  dropout={dropout} weight_decay={weight_decay} "
          f"transition_loss={'ON (weight=' + str(TRANSITION_LOSS_WEIGHT) + ')' if use_transition_loss else 'OFF'} "
          f"oversampling={'ON (threshold=' + str(OVERSAMPLE_GLOSS_THRESHOLD) + ', factor=' + str(OVERSAMPLE_FACTOR) + 'x)' if use_oversampling else 'OFF'}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=weight_decay)

    train_dataset = SignLanguageDataset(
        npz_dir=TRAIN_NPZ_DIR, vocab_path=VOCAB_MERGED_PATH, asllrp_token_to_id_path=ASLLRP_TOKEN_TO_ID_PATH,
        augment=False, include_transition_target=use_transition_loss,
    )
    train_loss_probe_dataset = SignLanguageDataset(
        npz_dir=TRAIN_NPZ_DIR, vocab_path=VOCAB_MERGED_PATH, asllrp_token_to_id_path=ASLLRP_TOKEN_TO_ID_PATH,
    )
    val_probe_dataset = SignLanguageDataset(
        npz_dir=VAL_NPZ_DIR, vocab_path=VOCAB_MERGED_PATH, asllrp_token_to_id_path=ASLLRP_TOKEN_TO_ID_PATH,
    )

    if use_oversampling:
        sampler = build_oversampling_sampler(train_dataset, OVERSAMPLE_GLOSS_THRESHOLD, OVERSAMPLE_FACTOR, TRAIN_SEED)
        loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn,
                             sampler=sampler, num_workers=0)
    else:
        shuffle_generator = torch.Generator().manual_seed(TRAIN_SEED)
        loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn,
                             shuffle=True, num_workers=0, generator=shuffle_generator)

    ckpt_dir = CHECKPOINT_SAVE_DIR / config_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    probe_log = []

    def do_probe(step):
        entry = {"step": step}
        entry["train_ctc_loss"] = probe_ctc_loss(model, train_loss_probe_dataset, train_loss_probe_idx, device, blank_id)
        entry["val_ctc_loss"] = probe_ctc_loss(model, val_probe_dataset, val_probe_idx, device, blank_id)
        entry["train_val_gap"] = entry["val_ctc_loss"] - entry["train_ctc_loss"]
        entry["val_blob_f1"] = probe_blob_count_and_boundary_f1(model, val_probe_dataset, val_probe_idx, device)
        entry["dense_val_blob_f1"] = probe_blob_count_and_boundary_f1(model, val_probe_dataset, dense_val_probe_idx, device)
        probe_log.append(entry)
        print(f"  [PROBE step={step}] train_ctc={entry['train_ctc_loss']:.4f} val_ctc={entry['val_ctc_loss']:.4f} "
              f"gap={entry['train_val_gap']:+.4f} | val: pred_blobs={entry['val_blob_f1']['mean_pred_blob_count']:.2f} "
              f"true_blobs={entry['val_blob_f1']['mean_true_blob_count']:.2f} "
              f"iouF1={entry['val_blob_f1']['iou']['f1']:.4f} ctF1={entry['val_blob_f1']['center_tolerance']['f1']:.4f} "
              f"| dense: pred_blobs={entry['dense_val_blob_f1']['mean_pred_blob_count']:.2f} "
              f"true_blobs={entry['dense_val_blob_f1']['mean_true_blob_count']:.2f} "
              f"iouF1={entry['dense_val_blob_f1']['iou']['f1']:.4f} ctF1={entry['dense_val_blob_f1']['center_tolerance']['f1']:.4f}")

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
            transition_target = batch.get("transition_target") if use_transition_loss else None
            total_loss, ctc_loss, boundary_loss, _ge, transition_loss = compute_combined_loss_with_gate_and_transition(
                ctc_log_probs, boundary_logits, gate_logits,
                batch["target_tokens"], batch["boundary_target"],
                batch["encoder_attention_mask"],
                bos_id, eos_id, pad_id, blank_id,
                boundary_loss_weight=BOUNDARY_LOSS_WEIGHT,
                gate_reg_weight=GATE_REG_WEIGHT,
                blank_logit_bias=BLANK_LOGIT_BIAS,
                transition_target=transition_target,
                transition_loss_weight=TRANSITION_LOSS_WEIGHT if use_transition_loss else 0.0,
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
        "config": config_name, "dropout": dropout, "weight_decay": weight_decay,
        "transition_loss_weight": TRANSITION_LOSS_WEIGHT if use_transition_loss else 0.0,
        "oversample_threshold": OVERSAMPLE_GLOSS_THRESHOLD if use_oversampling else None,
        "oversample_factor": OVERSAMPLE_FACTOR if use_oversampling else None,
        "loss_log": loss_log, "probe_log": probe_log,
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

    train_loss_probe_idx = pick_fixed_indices(TRAIN_NPZ_DIR, TRAIN_LOSS_PROBE_SEED, TRAIN_LOSS_PROBE_SIZE)
    val_probe_idx = pick_fixed_indices(VAL_NPZ_DIR, VAL_PROBE_SEED, VAL_PROBE_SIZE)
    dense_val_probe_idx = pick_fixed_indices(
        VAL_NPZ_DIR, seed=0, size=None,
        filter_fn=lambda meta: len(meta.gloss.split("|")) > OVERSAMPLE_GLOSS_THRESHOLD,
    )
    print(f"Train loss probe: {len(train_loss_probe_idx)} samples (seed={TRAIN_LOSS_PROBE_SEED})")
    print(f"Val probe:        {len(val_probe_idx)} samples (seed={VAL_PROBE_SEED})")
    print(f"Dense val probe:  {len(dense_val_probe_idx)} samples (ALL val utterances with >{OVERSAMPLE_GLOSS_THRESHOLD} glosses)")

    out_path = "logs/stageB1_boundary_mechanisms_smoke_test_results.json"
    results = []
    for config_name in CONFIGS:
        r = run_one_config(config_name, device, ctc_vocab_size, len(decoder_vocab), blank_id, bos_id, eos_id, pad_id,
                            train_loss_probe_idx, val_probe_idx, dense_val_probe_idx)
        results.append(r)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\n\nDone. Results in {out_path}, checkpoints under {CHECKPOINT_SAVE_DIR}/<config_name>/")


if __name__ == "__main__":
    main()
