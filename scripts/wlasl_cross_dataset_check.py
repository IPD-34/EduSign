"""
scripts/wlasl_cross_dataset_check.py

WLASL cross-dataset generalization check. WLASL fails the 10+/split
floor on 100% of its own words (never meant to be pooled into
training) -- its role here is a held-out generalization test: train
on ASL Citizen only, then check whether accuracy holds up on a
DIFFERENT source's landmarks for the same words, or collapses (partial
overfitting to ASL Citizen's specific collection conditions).

PREREQUISITE FINDING, checked before writing this: no ASL-Citizen-
trained checkpoint exists anywhere in this project -- every prior
vocab-scaling run tracked best-epoch weights only in memory (to
restore before its own test eval), never called torch.save. Also,
yesterday's already-trained 40/100/200-word models only cover 88 of
the 1,158 words that actually overlap with WLASL (they were built from
ASL Citizen's top-K most FREQUENT words, not chosen for WLASL overlap).
So this trains one new model -- SAME locked setup as every other run
in this project (architecture, pretrained-encoder init, 30 epochs,
batch/LR/optimizer, augmentation=off) -- on exactly the 1,158
WLASL-overlapping words, and this time actually saves the checkpoint.

Step 1 (overlap count, 1,158/2,674 floor-clearing Citizen words) was
computed separately, see logs/_tmp_citizen_floor_wlasl_overlap.json.
"""
import sys
import json
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader

from scripts.asl_citizen_vocab_scaling import (
    run_one_vocab_size, IsolatedSignDataset, collate_fn, CTCIsolatedSignModel,
    evaluate, peek_gloss_split, PROCESSED_DIR as CITIZEN_PROCESSED_DIR,
)
from src.models.train_utils import get_device

WLASL_PROCESSED_DIR = Path("data/processed/wlasl")
CHECKPOINT_PATH = "checkpoints/asl_citizen_wlasl_overlap_1158/model.pt"


class WLASLEvalDataset(torch.utils.data.Dataset):
    """ALL of WLASL's data (train+val+test) for the target vocabulary --
    none of it was used in training (training was ASL-Citizen-only), so
    every WLASL example is equally valid as a held-out generalization
    probe regardless of WLASL's own internal split assignment. Reuses
    the SAME word_to_label mapping the model was trained with (loaded
    from the checkpoint) -- label indices must match training exactly."""
    def __init__(self, wlasl_dir, word_to_label):
        self.paths = []
        self.glosses = []
        for split in ("train", "val", "test"):
            for p in (wlasl_dir / split).glob("*.npz"):
                gloss, _ = peek_gloss_split(p)
                if gloss in word_to_label:
                    self.paths.append(p)
                    self.glosses.append(gloss)
        self.word_to_label = word_to_label

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        from src.preprocessing.sample_writer import load_sample
        from src.models.feature_utils import flatten_frame_features
        sample = load_sample(self.paths[idx])
        landmarks = torch.tensor(flatten_frame_features(sample), dtype=torch.float32)
        label = self.word_to_label[sample.metadata.gloss]
        return {"landmarks": landmarks, "label": label}


def main():
    device = get_device()
    print(f"Device: {device}")

    with open("logs/_tmp_citizen_floor_wlasl_overlap.json") as f:
        overlap_words = json.load(f)
    k = len(overlap_words)
    print(f"Training vocabulary: {k} words (ASL Citizen floor-clearing words that also exist in WLASL)")

    # Step 2a: train on ASL Citizen (locked setup, unchanged) for exactly
    # this vocabulary, saving the checkpoint this time.
    result = run_one_vocab_size(k, sorted(overlap_words), device, save_checkpoint_path=CHECKPOINT_PATH)

    print("\n=== ASL CITIZEN (own held-out test set) ===")
    print(f"test_top1={result['test_top1']:.4f}  test_top5={result['test_top5']:.4f}  "
          f"n_test={result['test_n']}  chance={result['chance_accuracy']:.4f}")

    # Step 2b: evaluate the SAME trained model on WLASL's landmarks for
    # the SAME vocabulary.
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    word_to_label = ckpt["word_to_label"]
    model = CTCIsolatedSignModel(vocab_size=ckpt["vocab_size"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    wlasl_ds = WLASLEvalDataset(WLASL_PROCESSED_DIR, word_to_label)
    print(f"\nWLASL eval set (all splits pooled, overlapping words only): {len(wlasl_ds)} clips")
    wlasl_class_counts = Counter(word_to_label[g] for g in wlasl_ds.glosses)
    n_classes_present = len(wlasl_class_counts)
    print(f"Classes actually represented in WLASL data: {n_classes_present}/{k}")

    wlasl_loader = DataLoader(wlasl_ds, batch_size=16, shuffle=False, collate_fn=collate_fn)
    wlasl_top1, wlasl_top5, wlasl_n = evaluate(model, wlasl_loader, device)

    print("\n=== WLASL (cross-dataset, same model, same vocabulary) ===")
    print(f"top1={wlasl_top1:.4f}  top5={wlasl_top5:.4f}  n={wlasl_n}")

    gap_top1 = result["test_top1"] - wlasl_top1
    gap_top5 = result["test_top5"] - wlasl_top5
    print("\n=== CROSS-DATASET GENERALIZATION GAP ===")
    print(f"top1: Citizen={result['test_top1']:.4f}  WLASL={wlasl_top1:.4f}  drop={gap_top1:+.4f}")
    print(f"top5: Citizen={result['test_top5']:.4f}  WLASL={wlasl_top5:.4f}  drop={gap_top5:+.4f}")

    out = {
        "vocab_size": k, "checkpoint": CHECKPOINT_PATH,
        "citizen_test": {"top1": result["test_top1"], "top5": result["test_top5"], "n": result["test_n"]},
        "wlasl_eval": {"top1": wlasl_top1, "top5": wlasl_top5, "n": wlasl_n, "n_classes_present": n_classes_present},
        "gap": {"top1": gap_top1, "top5": gap_top5},
    }
    with open("logs/wlasl_cross_dataset_check_result.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved logs/wlasl_cross_dataset_check_result.json")


if __name__ == "__main__":
    main()
