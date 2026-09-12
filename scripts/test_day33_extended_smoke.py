"""
test_day33_extended_smoke.py

Day 33: two things before real Kaggle quota is spent.

1. Corpus-wide length-mismatch audit: for every ASLLRP utterance,
   check whether its real encoder timestep count (num_frames, post
   sampling) is >= its gloss count. CTC requires input_length >=
   target_length per sample -- violations produce inf loss under
   zero_infinity=False, silently corrupting that batch's gradient if
   not caught first.

2. Extended (~150-step) local run, tracking ctc_loss and boundary_loss
   SEPARATELY across the whole run -- not just the first 3 steps -- to
   see whether CTC's current ~100x magnitude dominance over boundary
   loss (found in Day 32's smoke test) persists, worsens, or naturally
   narrows as training proceeds.
"""

import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models.sign_transformer import SignTransformer
from src.models.train_utils import load_checkpoint_partial, get_device
from src.models.dataset import SignLanguageDataset, collate_fn
from src.models.combined_loss import compute_combined_loss

VOCAB_MERGED_PATH = "data/vocab_merged.json"
DECODER_VOCAB_PATH = "data/vocab.json"  # 1850 entries -- old WLASL decoder vocab
ASLLRP_TOKEN_TO_ID_PATH = "data/asllrp_token_to_id.json"
OLD_CHECKPOINT_PATH = "checkpoints/day12_latest.pt"
ASLLRP_NPZ_DIR = "data/processed/asllrp/train"
NUM_STEPS = 150
BOUNDARY_LOSS_WEIGHT = 1.0  # unchanged default -- this run's OWN job is
                             # to tell you whether this needs revisiting

# Added on the --reference-ratio 8.1 investigation: the original Day 33
# run never persisted its per-step data (confirmed -- this script only
# ever printed to console, no file write existed before this change), so
# the per-step ratio log below is written by THIS run, not recovered from
# the historical one. This script is also unseeded (no torch.manual_seed)
# and shuffles the DataLoader, so a fresh run will NOT reproduce Day 33's
# original numbers step-for-step -- it produces a new, independent
# 150-step run under the same procedure, useful for testing which
# aggregation formula could plausibly have produced "8.1x", not for
# recovering the literal original value.
PER_STEP_LOG_PATH = "logs/day33_extended_smoke_per_step_rerun.csv"
WARMUP_STEPS_EXCLUDED = 50  # per Day 33's own doc: "collapses to single-digit-x
                             # within the first 50 steps, then plateaus" --
                             # tested as one candidate steady-state window


def audit_length_mismatch():
    """
    Part 1: corpus-wide check, not a sample. For every ASLLRP .npz,
    compare num_frames (real encoder timesteps after sampling) against
    the number of glosses -- flags any utterance where CTC's
    input_length >= target_length requirement would be violated.
    """
    paths = list(Path("data/processed/asllrp").rglob("*.npz"))
    violations = []
    gloss_counts = []
    frame_counts = []

    for p in paths:
        d = np.load(p, allow_pickle=True)
        meta = json.loads(str(d["metadata"]))
        num_frames = meta["num_frames"]
        num_glosses = len(meta["gloss"].split("|"))
        gloss_counts.append(num_glosses)
        frame_counts.append(num_frames)
        if num_frames < num_glosses:
            violations.append((meta["video_id"], num_frames, num_glosses))

    print(f"=== Day 33 length-mismatch audit: {len(paths)} utterances ===")
    print(f"Violations (num_frames < num_glosses): {len(violations)} / {len(paths)} "
          f"({100*len(violations)/len(paths):.1f}%)")
    if violations:
        print("Sample violations (video_id, num_frames, num_glosses):")
        for v in violations[:10]:
            print(f"  {v}")
        # Check whether violations skew toward the shortest utterances,
        # per Day 33's plan's own stated hypothesis.
        violation_frames = [v[1] for v in violations]
        print(f"\nViolation num_frames: min={min(violation_frames)}, "
              f"max={max(violation_frames)}, mean={sum(violation_frames)/len(violation_frames):.1f}")
        print(f"Corpus-wide num_frames: min={min(frame_counts)}, mean={sum(frame_counts)/len(frame_counts):.1f}")
    print()
    return violations


def extended_smoke_run():
    """Part 2: ~150-step run, tracking both loss terms across the whole thing."""
    device = get_device()

    with open(VOCAB_MERGED_PATH) as f:
        vocab_merged = json.load(f)
    with open(DECODER_VOCAB_PATH) as f:
        decoder_vocab = json.load(f)
    ctc_vocab_size = len(vocab_merged)
    blank_id = ctc_vocab_size
    bos_id = vocab_merged["<BOS>"]
    eos_id = vocab_merged["<EOS>"]
    pad_id = vocab_merged["<PAD>"]

    model = SignTransformer(vocab_size=len(decoder_vocab), ctc_vocab_size=ctc_vocab_size).to(device)
    load_checkpoint_partial(model, OLD_CHECKPOINT_PATH, prefixes_to_load=("encoder.",))

    dataset = SignLanguageDataset(
        npz_dir=ASLLRP_NPZ_DIR,
        vocab_path=VOCAB_MERGED_PATH,
        asllrp_token_to_id_path=ASLLRP_TOKEN_TO_ID_PATH,
    )
    loader = DataLoader(
        dataset, batch_size=4, collate_fn=collate_fn,
        num_workers=2, persistent_workers=True, shuffle=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    ctc_losses, boundary_losses = [], []
    per_step_log = []  # (step, ctc_loss, boundary_loss, ratio) for every valid step
    skipped_inf_batches = 0
    step = 0

    print(f"=== Day 33 extended smoke run: {NUM_STEPS} steps ===\n")

    while step < NUM_STEPS:
        for batch in loader:
            if step >= NUM_STEPS:
                break

            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

            ctc_log_probs, boundary_logits = model.forward_ctc_and_boundary(
                batch["landmarks"], batch["encoder_attention_mask"]
            )

            total_loss, ctc_loss, boundary_loss = compute_combined_loss(
                ctc_log_probs, boundary_logits,
                batch["target_tokens"], batch["boundary_target"],
                batch["encoder_attention_mask"],
                bos_id, eos_id, pad_id, blank_id,
                boundary_loss_weight=BOUNDARY_LOSS_WEIGHT,
            )

            if not torch.isfinite(total_loss):
                # Expected, given Part 1's audit likely found real
                # violations -- count it, don't silently skip without
                # a trace, and don't let it corrupt the optimizer step.
                skipped_inf_batches += 1
                step += 1
                continue

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            ctc_losses.append(ctc_loss.item())
            boundary_losses.append(boundary_loss.item())
            per_step_log.append((
                step, ctc_loss.item(), boundary_loss.item(),
                ctc_loss.item() / max(boundary_loss.item(), 1e-8),
            ))

            if step % 25 == 0 or step == NUM_STEPS - 1:
                print(f"Step {step}: ctc_loss={ctc_loss.item():.4f}  "
                      f"boundary_loss={boundary_loss.item():.4f}  "
                      f"ratio={ctc_loss.item()/max(boundary_loss.item(), 1e-8):.1f}x")

            step += 1

    print(f"\n=== Summary ({len(ctc_losses)} valid steps, {skipped_inf_batches} skipped as non-finite) ===")
    print(f"ctc_loss:      start={ctc_losses[0]:.2f}  end={ctc_losses[-1]:.2f}  "
          f"mean_last_20={sum(ctc_losses[-20:])/len(ctc_losses[-20:]):.2f}")
    print(f"boundary_loss: start={boundary_losses[0]:.4f}  end={boundary_losses[-1]:.4f}  "
          f"mean_last_20={sum(boundary_losses[-20:])/len(boundary_losses[-20:]):.4f}")
    print(f"\nSkipped (non-finite) batches: {skipped_inf_batches}/{NUM_STEPS} "
          f"({100*skipped_inf_batches/NUM_STEPS:.1f}%) -- cross-check this against "
          f"Part 1's audit violation rate above; they should roughly agree.")

    # --- --reference-ratio 8.1 investigation (added, not part of Day 33's
    # original scope) -- save the full per-step log and test several
    # honest candidate aggregations against 8.1, rather than assuming
    # any one of them is "the" answer.
    Path(PER_STEP_LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(PER_STEP_LOG_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "ctc_loss", "boundary_loss", "ratio"])
        writer.writerows(per_step_log)
    print(f"\nFull per-step log written: {PER_STEP_LOG_PATH} ({len(per_step_log)} rows)")

    ratios = [r[3] for r in per_step_log]
    steady_ratios = [r[3] for r in per_step_log if r[0] >= WARMUP_STEPS_EXCLUDED]

    mean_all = sum(ratios) / len(ratios)
    median_all = sorted(ratios)[len(ratios) // 2]
    geomean_all = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
    ratio_of_means_all = sum(ctc_losses) / sum(boundary_losses)
    mean_steady = sum(steady_ratios) / len(steady_ratios)
    ratio_of_means_steady = (
        sum(ctc_losses[WARMUP_STEPS_EXCLUDED:]) / sum(boundary_losses[WARMUP_STEPS_EXCLUDED:])
    )
    mean_last_20 = sum(ratios[-20:]) / len(ratios[-20:])

    print("\n=== --reference-ratio 8.1 candidate check (THIS rerun, not the original Day 33 run) ===")
    print(f"  mean of all {len(ratios)} per-step ratios:            {mean_all:.2f}x")
    print(f"  median of all {len(ratios)} per-step ratios:          {median_all:.2f}x")
    print(f"  geometric mean of all {len(ratios)} per-step ratios:  {geomean_all:.2f}x")
    print(f"  sum(ctc_loss)/sum(boundary_loss), all steps:          {ratio_of_means_all:.2f}x")
    print(f"  mean of per-step ratios, steps >= {WARMUP_STEPS_EXCLUDED} "
          f"({len(steady_ratios)} steps):    {mean_steady:.2f}x")
    print(f"  sum(ctc_loss)/sum(boundary_loss), steps >= {WARMUP_STEPS_EXCLUDED}:     "
          f"{ratio_of_means_steady:.2f}x")
    print(f"  mean of last-20 per-step ratios:                      {mean_last_20:.2f}x")
    print("  (target: 8.1x -- reminder: this is a fresh, unseeded, reshuffled run, "
          "not the historical Day 33 run, so exact reproduction of that run's own "
          "number is not possible either way -- this only tests which formula lands "
          "near 8.1x under an equivalent procedure.)")


if __name__ == "__main__":
    violations = audit_length_mismatch()
    extended_smoke_run()