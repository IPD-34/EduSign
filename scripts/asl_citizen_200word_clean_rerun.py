"""
scripts/asl_citizen_200word_clean_rerun.py

Clean rerun of the 200-word point from the locked vocab-scaling curve,
now that extraction is fully complete (6744/6744, 0/200 val zero-class
gaps -- confirmed, vs. 11/200 when the original 200-word point trained
mid-extraction). Same locked setup as the original curve (architecture,
NUM_EPOCHS=30, batch size, optimizer, augmentation policy) -- only the
underlying data completeness differs. Purpose: confirm whether the
original 25.53% test_top1 / +11.25% val_test_gap moved now that
best-checkpoint selection isn't compromised by missing val data.

Saved separately (NOT logs/asl_citizen_vocab_scaling_results.json) so
the original locked curve's results are preserved untouched.
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.asl_citizen_vocab_scaling import load_gloss_ranking, run_one_vocab_size
from src.models.train_utils import get_device


def main():
    device = get_device()
    print(f"Device: {device}")
    ranked, train_c, val_c, test_c = load_gloss_ranking()

    result = run_one_vocab_size(200, ranked, device)  # defaults: NUM_EPOCHS=30, locked setup

    out_path = "logs/asl_citizen_200word_clean_rerun_result.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {out_path}")

    original = {"test_top1": 0.2553, "test_top5": 0.5609, "val_test_gap": 0.1125}
    print("\n=== COMPARISON: original (mid-extraction, 11/200 val gap) vs clean rerun (0/200 val gap) ===")
    print(f"original: test_top1={original['test_top1']:.4f}  test_top5={original['test_top5']:.4f}  "
          f"val_test_gap={original['val_test_gap']:+.4f}")
    print(f"clean   : test_top1={result['test_top1']:.4f}  test_top5={result['test_top5']:.4f}  "
          f"val_test_gap={result['val_test_gap']:+.4f}")
    print(f"delta   : test_top1={result['test_top1']-original['test_top1']:+.4f}  "
          f"val_test_gap={result['val_test_gap']-original['val_test_gap']:+.4f}")


if __name__ == "__main__":
    main()
