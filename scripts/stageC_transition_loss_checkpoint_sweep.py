"""
scripts/stageC_transition_loss_checkpoint_sweep.py

Post-hoc, full 122-sample TEST-split evaluation across EVERY checkpoint
saved by the Stage C transition-loss real run
(checkpoints/stageC_transition_loss_real/) -- both full WER
(compute_wer_beam_search.run_full_test_split) and BOTH boundary-F1
criteria (compute_boundary_f1.run_full_test_split_dual_criteria),
UNMODIFIED.

WHY THE FULL SET, NOT A SCREENING SUBSET FIRST: this project's own
standing lesson (val CTC loss, then a 30-sample WER probe, then a
55-sample screen -- each shown to mis-rank checkpoints at least once
relative to the full 122-sample split) means NO proxy is trusted to
pick the final "best" checkpoint. The Day 54 screening-then-confirm
two-step exists to make a LARGE checkpoint set affordable; this run
only produces a small, bounded number of checkpoints (checkpoint-every-
2000-steps, no cleanup, early-stopped run -- typically 6-8 total, same
order as the two prior real runs' own 12000-step early stops), so
running the full 122-sample evaluation on literally every one of them
directly is itself affordable -- no screening step needed or used.

Evaluation-only. No training, no checkpoint-of-record changes.
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.compute_wer_beam_search import run_full_test_split as wer_run
from scripts.compute_boundary_f1 import run_full_test_split_dual_criteria as boundary_dual_run

TEST_NPZ_DIR = "data/processed/asllrp/test"
RUN_DIR = Path("checkpoints/stageC_transition_loss_real")

BASELINE = {
    "label": "control/best_by_wer.pt (current best, Stage B1)",
    "path": "checkpoints/stageB1_real/control/best_by_wer.pt",
    "wer": 0.9173, "iou_f1": 0.1342, "ct_f1": 0.3498,
}


def discover_checkpoints():
    checkpoints = []
    for p in sorted(RUN_DIR.glob("step_*.pt")):
        checkpoints.append((p.stem, p))
    for name in ("best_val.pt", "best_by_wer.pt"):
        p = RUN_DIR / name
        if p.exists():
            checkpoints.append((p.stem, p))
    return checkpoints


def main():
    checkpoints = discover_checkpoints()
    print(f"Found {len(checkpoints)} checkpoints under {RUN_DIR}: {[c[0] for c in checkpoints]}\n")

    results = []
    # Also re-score the baseline on THIS run's exact machinery, as a live
    # sanity check that nothing about the eval path itself has drifted
    # (same discipline as the boundary-F1 dual-criteria task's own
    # "reconfirm the existing number reproduces exactly" step).
    print(f"--- BASELINE RECONFIRM: {BASELINE['label']} ---")
    wer_b = wer_run(checkpoint_path=BASELINE["path"], npz_dir=TEST_NPZ_DIR, use_gate=True)
    bf1_b = boundary_dual_run(checkpoint_path=BASELINE["path"], npz_dir=TEST_NPZ_DIR, use_gate=True)
    print(f"  reconfirmed: WER={wer_b['corpus_wer']:.4f} (expected {BASELINE['wer']}) "
          f"iouF1={bf1_b['iou']['micro_f1']:.4f} (expected {BASELINE['iou_f1']}) "
          f"ctF1={bf1_b['center_tolerance']['micro_f1']:.4f} (expected {BASELINE['ct_f1']})\n")

    for label, path in checkpoints:
        print(f"--- {label} ({path}) ---")
        wer = wer_run(checkpoint_path=str(path), npz_dir=TEST_NPZ_DIR, use_gate=True)
        bf1 = boundary_dual_run(checkpoint_path=str(path), npz_dir=TEST_NPZ_DIR, use_gate=True)
        row = {
            "label": label, "path": str(path),
            "corpus_wer": wer["corpus_wer"], "mean_utterance_wer": wer["mean_utterance_wer"],
            "n_empty_decode": sum(1 for r in wer["per_sample"] if len(r["decoded_ids"]) == 0),
            "iou_f1": bf1["iou"]["micro_f1"], "iou_precision": bf1["iou"]["micro_precision"], "iou_recall": bf1["iou"]["micro_recall"],
            "ct_f1": bf1["center_tolerance"]["micro_f1"], "ct_precision": bf1["center_tolerance"]["micro_precision"],
            "ct_recall": bf1["center_tolerance"]["micro_recall"],
            "n_samples": wer["n_samples"],
        }
        results.append(row)
        print(f"  corpus_WER={row['corpus_wer']:.4f}  iouF1={row['iou_f1']:.4f}  ctF1={row['ct_f1']:.4f}  "
              f"empty_decode={row['n_empty_decode']}/{row['n_samples']}\n")

    print("\n" + "=" * 110)
    print(f"FULL 122-SAMPLE SWEEP -- ranked by corpus WER (ascending = better)")
    print("=" * 110)
    ranked = sorted(results, key=lambda r: r["corpus_wer"])
    header = f"{'checkpoint':<20} {'corpus_WER':>11} {'iouF1':>8} {'ctF1':>8} {'iouP':>7} {'iouR':>7} {'ctP':>7} {'ctR':>7} {'empty':>7}"
    print(header)
    for r in ranked:
        print(f"{r['label']:<20} {r['corpus_wer']:>11.4f} {r['iou_f1']:>8.4f} {r['ct_f1']:>8.4f} "
              f"{r['iou_precision']:>7.4f} {r['iou_recall']:>7.4f} {r['ct_precision']:>7.4f} {r['ct_recall']:>7.4f} "
              f"{r['n_empty_decode']:>7}/{r['n_samples']}")

    print(f"\nBASELINE (control/best_by_wer.pt): WER={BASELINE['wer']}  iouF1={BASELINE['iou_f1']}  ctF1={BASELINE['ct_f1']}")

    out_path = Path("logs") / "stageC_transition_loss_checkpoint_sweep.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "baseline_reconfirm": {
                "wer": wer_b["corpus_wer"], "iou_f1": bf1_b["iou"]["micro_f1"], "ct_f1": bf1_b["center_tolerance"]["micro_f1"],
            },
            "results": results,
        }, f, indent=2)
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()
