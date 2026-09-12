"""
scripts/day52_convergence_and_gate_recheck.py

Day 52: two real held-out-vs-train / checkpoint comparisons, both built
on top of Day 50/51's hand-validated, unmodified scoring math
(ctc_prefix_beam_search, edit_distance, boundary_f1_for_sample,
interval IoU matching) -- only the DATA SOURCE (train vs. test) and the
CHECKPOINT/GATE STATE being scored change; beam_width, expand_top_k,
IOU_THRESHOLD, and BOUNDARY_PROB_THRESHOLD are untouched from Day 51.

--- PART 1: TRAIN-VS-HELD-OUT GAP ---
Runs the same two scripts against a 122-sample train-split subset
(matching the test split's own size exactly, for a fair side-by-side --
not the full 990, to keep runtime bounded on top of Day 51's own full
122-sample runs; 122 is large enough to be a real aggregate, not a
cherry-picked handful). Sample chosen via a fixed seed (52, matching
this project's per-day deterministic-seed convention -- see
scripts/audit_sign_spans.py's seed=31), not the first N alphabetically,
to avoid any systematic bias from video_id ordering correlating with
extraction batch or utterance length.

--- PART 2: GATE-DECISION RECHECK ---
Runs the same two scripts against best_val.pt (step 4000, the pre-gate
checkpoint) on the IDENTICAL 122-file test split Day 51 used, with the
gate path bypassed (use_gate=False -> forward_ctc_and_boundary(),
exactly how that checkpoint was trained). Confound stated explicitly,
per the task: best_val.pt and best_val_gated.pt differ in BOTH training
step (4000 vs 2000) AND gate state -- this comparison does not isolate
the gate's effect the way Week 7's same-checkpoint ablation did. It
answers a different, real question: is the checkpoint Week 7 chose to
build the gate on top of itself a good checkpoint on real generalization
metrics, compared to the earlier one.
"""
import sys
import json
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.dataset import SignLanguageDataset

from scripts.compute_wer_beam_search import run_full_test_split as wer_run
from scripts.compute_boundary_f1 import run_full_test_split as boundary_f1_run

TRAIN_NPZ_DIR = "data/processed/asllrp/train"
TEST_NPZ_DIR = "data/processed/asllrp/test"
GATED_CHECKPOINT = "checkpoints/month2_asllrp/best_val_gated.pt"   # step 2000, gate active
PREGATE_CHECKPOINT = "checkpoints/month2_asllrp/best_val.pt"        # step 4000, gate inactive

TRAIN_SAMPLE_SIZE = 122  # matches the test split's own size exactly
TRAIN_SAMPLE_SEED = 52


def pick_train_indices():
    dataset = SignLanguageDataset(
        npz_dir=TRAIN_NPZ_DIR, vocab_path="data/vocab_merged.json",
        asllrp_token_to_id_path="data/asllrp_token_to_id.json",
    )
    n = len(dataset)
    rng = random.Random(TRAIN_SAMPLE_SEED)
    indices = rng.sample(range(n), min(TRAIN_SAMPLE_SIZE, n))
    return indices, n


def print_wer_row(label, r):
    n_empty = sum(1 for x in r["per_sample"] if len(x["decoded_ids"]) == 0)
    n_exact = sum(1 for x in r["per_sample"] if x["edits"] == 0)
    print(f"{label:<28} n={r['n_samples']:<5} corpus_WER={r['corpus_wer']:.4f}  "
          f"mean_utt_WER={r['mean_utterance_wer']:.4f}  empty_decode={n_empty}/{r['n_samples']}  "
          f"exact_match={n_exact}/{r['n_samples']}")


def print_bf1_row(label, r):
    per_sample_f1 = [x["f1"] for x in r["per_sample"]]
    mean_f1 = sum(per_sample_f1) / len(per_sample_f1) if per_sample_f1 else 0.0
    n_zero = sum(1 for f in per_sample_f1 if f == 0.0)
    print(f"{label:<28} n={r['n_samples']:<5} TP={r['total_tp']:<5} FP={r['total_fp']:<5} "
          f"FN={r['total_fn']:<5} P={r['micro_precision']:.4f}  R={r['micro_recall']:.4f}  "
          f"F1={r['micro_f1']:.4f}  mean_per_sample_F1={mean_f1:.4f}  F1==0:{n_zero}/{r['n_samples']}")


def main():
    print("=" * 78)
    print("PART 1: TRAIN-VS-HELD-OUT GAP")
    print(f"  train subset: {TRAIN_SAMPLE_SIZE} samples, seed={TRAIN_SAMPLE_SEED}, from {TRAIN_NPZ_DIR}")
    print(f"  checkpoint: {GATED_CHECKPOINT} (step 2000, gate active) -- same checkpoint as Day 51's test run")
    print("=" * 78)

    train_indices, n_train_total = pick_train_indices()
    print(f"  (sampled {len(train_indices)} of {n_train_total} train files)\n")

    train_wer = wer_run(checkpoint_path=GATED_CHECKPOINT, npz_dir=TRAIN_NPZ_DIR,
                         use_gate=True, indices=train_indices)
    train_bf1 = boundary_f1_run(checkpoint_path=GATED_CHECKPOINT, npz_dir=TRAIN_NPZ_DIR,
                                 use_gate=True, indices=train_indices)

    with open("logs/day51_wer_full_test_split.json") as f:
        test_wer_saved = json.load(f)
    with open("logs/day51_boundary_f1_full_test_split.json") as f:
        test_bf1_saved = json.load(f)

    print("\n--- WER: train (sample) vs. Day 51 held-out test (full) ---")
    print_wer_row("TRAIN (n=122, seed=52)", train_wer)
    print(f"{'TEST (Day 51, n=122)':<28} n={test_wer_saved['n_samples']:<5} "
          f"corpus_WER={test_wer_saved['corpus_wer']:.4f}  mean_utt_WER={test_wer_saved['mean_utterance_wer']:.4f}  "
          f"empty_decode={test_wer_saved['n_empty_decode']}/{test_wer_saved['n_samples']}  "
          f"exact_match={test_wer_saved['n_exact_match']}/{test_wer_saved['n_samples']}")
    gap_wer = test_wer_saved["corpus_wer"] - train_wer["corpus_wer"]
    print(f"  GAP (test - train) corpus WER = {gap_wer:+.4f}")

    print("\n--- Boundary-F1: train (sample) vs. Day 51 held-out test (full) ---")
    print_bf1_row("TRAIN (n=122, seed=52)", train_bf1)
    print(f"{'TEST (Day 51, n=122)':<28} n={test_bf1_saved['n_samples']:<5} "
          f"TP={test_bf1_saved['total_tp']:<5} FP={test_bf1_saved['total_fp']:<5} FN={test_bf1_saved['total_fn']:<5} "
          f"P={test_bf1_saved['micro_precision']:.4f}  R={test_bf1_saved['micro_recall']:.4f}  "
          f"F1={test_bf1_saved['micro_f1']:.4f}  mean_per_sample_F1={test_bf1_saved['mean_per_sample_f1']:.4f}  "
          f"F1==0:{test_bf1_saved['n_zero_f1']}/{test_bf1_saved['n_samples']}")
    gap_f1 = test_bf1_saved["micro_f1"] - train_bf1["micro_f1"]
    print(f"  GAP (test - train) micro F1 = {gap_f1:+.4f}")

    train_wer_out = {
        "checkpoint": GATED_CHECKPOINT, "checkpoint_step": 2000, "split": "train_sample",
        "n_train_total": n_train_total, "sample_size": len(train_indices), "sample_seed": TRAIN_SAMPLE_SEED,
        "corpus_wer": train_wer["corpus_wer"], "mean_utterance_wer": train_wer["mean_utterance_wer"],
        "n_samples": train_wer["n_samples"],
    }
    train_bf1_out = {
        "checkpoint": GATED_CHECKPOINT, "checkpoint_step": 2000, "split": "train_sample",
        "n_train_total": n_train_total, "sample_size": len(train_indices), "sample_seed": TRAIN_SAMPLE_SEED,
        "micro_precision": train_bf1["micro_precision"], "micro_recall": train_bf1["micro_recall"],
        "micro_f1": train_bf1["micro_f1"], "total_tp": train_bf1["total_tp"],
        "total_fp": train_bf1["total_fp"], "total_fn": train_bf1["total_fn"],
        "n_samples": train_bf1["n_samples"],
    }
    with open("logs/day52_train_sample_wer.json", "w") as f:
        json.dump(train_wer_out, f, indent=2)
    with open("logs/day52_train_sample_boundary_f1.json", "w") as f:
        json.dump(train_bf1_out, f, indent=2)

    print("\n\n" + "=" * 78)
    print("PART 2: GATE-DECISION RECHECK (best_val.pt, step 4000, gate INACTIVE)")
    print(f"  checkpoint: {PREGATE_CHECKPOINT} (step 4000, val_loss=5.5040 on CTC-only metric)")
    print(f"  split: {TEST_NPZ_DIR} -- IDENTICAL 122 files Day 51 used for best_val_gated.pt")
    print("  *** CONFOUND: best_val.pt vs best_val_gated.pt differ in BOTH training step ***")
    print("  *** (4000 vs 2000) AND gate state -- this does NOT isolate the gate's effect ***")
    print("  *** the way Week 7's same-checkpoint ablation did. ***")
    print("=" * 78)

    pregate_wer = wer_run(checkpoint_path=PREGATE_CHECKPOINT, npz_dir=TEST_NPZ_DIR, use_gate=False)
    pregate_bf1 = boundary_f1_run(checkpoint_path=PREGATE_CHECKPOINT, npz_dir=TEST_NPZ_DIR, use_gate=False)

    print("\n--- WER: best_val.pt (step 4000, gate OFF) vs best_val_gated.pt (step 2000, gate ON) ---")
    print_wer_row("best_val.pt (step 4000)", pregate_wer)
    print(f"{'best_val_gated.pt (step 2000)':<28} n={test_wer_saved['n_samples']:<5} "
          f"corpus_WER={test_wer_saved['corpus_wer']:.4f}  mean_utt_WER={test_wer_saved['mean_utterance_wer']:.4f}  "
          f"empty_decode={test_wer_saved['n_empty_decode']}/{test_wer_saved['n_samples']}  "
          f"exact_match={test_wer_saved['n_exact_match']}/{test_wer_saved['n_samples']}")
    diff_wer = test_wer_saved["corpus_wer"] - pregate_wer["corpus_wer"]
    print(f"  DIFF (gated - pregate) corpus WER = {diff_wer:+.4f}  "
          f"({'pregate better' if diff_wer > 0 else 'gated better' if diff_wer < 0 else 'tied'})")

    print("\n--- Boundary-F1: best_val.pt (step 4000, gate OFF) vs best_val_gated.pt (step 2000, gate ON) ---")
    print_bf1_row("best_val.pt (step 4000)", pregate_bf1)
    print(f"{'best_val_gated.pt (step 2000)':<28} n={test_bf1_saved['n_samples']:<5} "
          f"TP={test_bf1_saved['total_tp']:<5} FP={test_bf1_saved['total_fp']:<5} FN={test_bf1_saved['total_fn']:<5} "
          f"P={test_bf1_saved['micro_precision']:.4f}  R={test_bf1_saved['micro_recall']:.4f}  "
          f"F1={test_bf1_saved['micro_f1']:.4f}  mean_per_sample_F1={test_bf1_saved['mean_per_sample_f1']:.4f}  "
          f"F1==0:{test_bf1_saved['n_zero_f1']}/{test_bf1_saved['n_samples']}")
    diff_f1 = test_bf1_saved["micro_f1"] - pregate_bf1["micro_f1"]
    print(f"  DIFF (gated - pregate) micro F1 = {diff_f1:+.4f}  "
          f"({'pregate better' if diff_f1 < 0 else 'gated better' if diff_f1 > 0 else 'tied'})")

    with open("logs/day52_pregate_wer_test_split.json", "w") as f:
        json.dump({
            "checkpoint": PREGATE_CHECKPOINT, "checkpoint_step": 4000, "gate_active": False, "split": "test",
            "n_test_files": 122, "corpus_wer": pregate_wer["corpus_wer"],
            "mean_utterance_wer": pregate_wer["mean_utterance_wer"], "n_samples": pregate_wer["n_samples"],
        }, f, indent=2)
    with open("logs/day52_pregate_boundary_f1_test_split.json", "w") as f:
        json.dump({
            "checkpoint": PREGATE_CHECKPOINT, "checkpoint_step": 4000, "gate_active": False, "split": "test",
            "n_test_files": 122, "micro_precision": pregate_bf1["micro_precision"],
            "micro_recall": pregate_bf1["micro_recall"], "micro_f1": pregate_bf1["micro_f1"],
            "total_tp": pregate_bf1["total_tp"], "total_fp": pregate_bf1["total_fp"],
            "total_fn": pregate_bf1["total_fn"], "n_samples": pregate_bf1["n_samples"],
        }, f, indent=2)

    print("\nSaved: logs/day52_train_sample_wer.json, logs/day52_train_sample_boundary_f1.json,")
    print("       logs/day52_pregate_wer_test_split.json, logs/day52_pregate_boundary_f1_test_split.json")


if __name__ == "__main__":
    main()
