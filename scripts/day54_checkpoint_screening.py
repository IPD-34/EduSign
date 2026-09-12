"""
scripts/day54_checkpoint_screening.py

Checks whether the current checkpoint-of-record (best_val_gated.pt,
selected purely by lowest val CTC loss) is actually the best-decoding
checkpoint currently on disk -- Day 52 found train ctc_loss collapsing
below 1.0 while decode-level WER stayed ~90%+ even on TRAINING data,
i.e. per-token CTC loss and real sequence-decode accuracy may not track
each other at this stage. If true, some OTHER already-existing
checkpoint could decode better right now, at zero new training cost.

Evaluation-only. No training, no architecture changes. Reuses Day 50/51's
compute_wer_beam_search.py / compute_boundary_f1.py UNMODIFIED (their
scoring math -- beam search, edit distance, IoU matching -- is untouched;
only the Day-52-added npz_dir/use_gate/indices parameters are used to
point them at different checkpoints/subsets).

--- SCREENING SUBSET: fixed, shared across every checkpoint ---
25 test-split files, chosen via a fixed seed (54) over the 122-file test
split, listed explicitly below so this is reproducible and so the same
apples-to-apples subset is used for every checkpoint (no re-sampling per
checkpoint).
"""
import sys
import json
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.dataset import SignLanguageDataset
from src.preprocessing.sample_writer import load_sample

from scripts.compute_wer_beam_search import run_full_test_split as wer_run
from scripts.compute_boundary_f1 import run_full_test_split as boundary_f1_run

TEST_NPZ_DIR = "data/processed/asllrp/test"
SCREEN_SEED = 54
SCREEN_SIZE = 25

CHECKPOINTS = [
    # (label, path, gate_dir_run, use_gate)
    ("best_val.pt",              "checkpoints/month2_asllrp/best_val.pt",              "ungated", False),
    ("best_val_gated.pt",        "checkpoints/month2_asllrp/best_val_gated.pt",        "gated",   True),
    ("step_0010000 (ungated)",   "checkpoints/month2_asllrp/step_0010000.pt",          "ungated", False),
    ("step_0012000 (ungated)",   "checkpoints/month2_asllrp/step_0012000.pt",          "ungated", False),
    ("step_0014000 (ungated)",   "checkpoints/month2_asllrp/step_0014000.pt",          "ungated", False),
    ("step_0008000 (gated)",     "checkpoints/month2_asllrp_gated/step_0008000.pt",    "gated",   True),
    ("step_0010000 (gated)",     "checkpoints/month2_asllrp_gated/step_0010000.pt",    "gated",   True),
    ("step_0012000 (gated)",     "checkpoints/month2_asllrp_gated/step_0012000.pt",    "gated",   True),
]


def pick_screen_indices():
    dataset = SignLanguageDataset(
        npz_dir=TEST_NPZ_DIR, vocab_path="data/vocab_merged.json",
        asllrp_token_to_id_path="data/asllrp_token_to_id.json",
    )
    n = len(dataset)
    rng = random.Random(SCREEN_SEED)
    indices = sorted(rng.sample(range(n), min(SCREEN_SIZE, n)))
    video_ids = [load_sample(dataset.npz_paths[i]).metadata.video_id for i in indices]
    return indices, video_ids, n


def main():
    indices, video_ids, n_total = pick_screen_indices()
    print(f"Screening subset: {len(indices)} of {n_total} test files, seed={SCREEN_SEED}")
    print("video_ids:", video_ids)
    print()

    results = []
    for label, path, run_type, use_gate in CHECKPOINTS:
        print(f"--- {label} ({path}, use_gate={use_gate}) ---")
        wer = wer_run(checkpoint_path=path, npz_dir=TEST_NPZ_DIR, use_gate=use_gate, indices=indices)
        bf1 = boundary_f1_run(checkpoint_path=path, npz_dir=TEST_NPZ_DIR, use_gate=use_gate, indices=indices)
        row = {
            "label": label, "path": path, "run_type": run_type, "use_gate": use_gate,
            "corpus_wer": wer["corpus_wer"], "mean_utterance_wer": wer["mean_utterance_wer"],
            "n_empty_decode": sum(1 for r in wer["per_sample"] if len(r["decoded_ids"]) == 0),
            "micro_precision": bf1["micro_precision"], "micro_recall": bf1["micro_recall"],
            "micro_f1": bf1["micro_f1"],
            "n_samples": wer["n_samples"],
        }
        results.append(row)
        print(f"  corpus_WER={row['corpus_wer']:.4f}  micro_F1={row['micro_f1']:.4f} "
              f"(P={row['micro_precision']:.4f} R={row['micro_recall']:.4f})  "
              f"empty_decode={row['n_empty_decode']}/{row['n_samples']}\n")

    print("\n" + "=" * 100)
    print(f"SCREENING TABLE -- n={len(indices)} shared subset, ranked by corpus WER (ascending = better)")
    print("=" * 100)
    ranked = sorted(results, key=lambda r: r["corpus_wer"])
    header = f"{'checkpoint':<28} {'run':<8} {'gate':<6} {'corpus_WER':>11} {'micro_F1':>9} {'P':>8} {'R':>8} {'empty':>7}"
    print(header)
    for r in ranked:
        print(f"{r['label']:<28} {r['run_type']:<8} {str(r['use_gate']):<6} {r['corpus_wer']:>11.4f} "
              f"{r['micro_f1']:>9.4f} {r['micro_precision']:>8.4f} {r['micro_recall']:>8.4f} "
              f"{r['n_empty_decode']:>7}/{r['n_samples']}")

    out_path = Path("logs") / "day54_checkpoint_screening.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "screen_seed": SCREEN_SEED, "screen_size": len(indices), "screen_indices": indices,
            "screen_video_ids": video_ids, "results": results,
        }, f, indent=2)
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()
