"""
scripts/asl_citizen_40word_extended.py

Exploratory side experiment, NOT part of the locked Step 2/3 scaling
curve: the official 40-word point (asl_citizen_vocab_scaling.py, fixed
at NUM_EPOCHS=30 across every vocab size for fair comparison) hadn't
plateaued -- val_top1 was still climbing at epoch 30 (40.7% @ ep20 ->
38.5% @ ep25 (noise) -> 45.0% @ ep30, best_val_top1=49.45%). This asks
a different question than the locked curve does: not "how does
accuracy change with vocab size under a fixed budget" but "how high
can the 40-word point go if given more epochs" -- answered separately
so it never touches or invalidates the official results file
(logs/asl_citizen_vocab_scaling_results.json) or the in-progress
100/200-word locked runs.

Same architecture, same data, same everything else as the locked
40-word run -- only num_epochs differs (100 instead of 30).
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.asl_citizen_vocab_scaling import load_gloss_ranking, run_one_vocab_size
from src.models.train_utils import get_device

EXTENDED_EPOCHS = 100


def main():
    device = get_device()
    print(f"Device: {device}")
    ranked, train_c, val_c, test_c = load_gloss_ranking()

    result = run_one_vocab_size(40, ranked, device, num_epochs=EXTENDED_EPOCHS)

    out_path = "logs/asl_citizen_40word_extended_result.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {out_path}")

    # explicit comparison against the locked 30-epoch point, if available
    locked_path = Path("logs/asl_citizen_vocab_scaling_results.json")
    if locked_path.exists():
        with open(locked_path) as f:
            locked_results = json.load(f)
        locked_40 = next((r for r in locked_results if r["vocab_size"] == 40), None)
        if locked_40:
            print("\n=== COMPARISON: 30 epochs (locked) vs 100 epochs (extended) ===")
            print(f"30ep : test_top1={locked_40['test_top1']:.4f}  test_top5={locked_40['test_top5']:.4f}  "
                  f"best_val_top1={locked_40['best_val_top1']:.4f}")
            print(f"100ep: test_top1={result['test_top1']:.4f}  test_top5={result['test_top5']:.4f}  "
                  f"best_val_top1={result['best_val_top1']:.4f}")


if __name__ == "__main__":
    main()
