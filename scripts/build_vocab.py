"""
build_vocab.py

Scans processed WLASL samples for unique gloss labels and builds a
vocabulary file (gloss <-> integer id), plus special tokens needed
for sequence generation (PAD, BOS, EOS, UNK).

Run once, before Day 9's dataloader/training work. Re-run only if
new gloss labels are added to the dataset (e.g. ASLLRP integration
later, which uses "|"-joined multi-gloss sequences — this vocab
builder already handles that by splitting on "|").
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.config import WLASL_PROCESSED_DIR
from src.preprocessing.sample_writer import load_sample

SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]


def main():
    npz_files = list(WLASL_PROCESSED_DIR.glob("*/*.npz"))
    print(f"Scanning {len(npz_files)} processed samples for gloss labels...")

    unique_glosses = set()
    for i, path in enumerate(npz_files, start=1):
        sample = load_sample(path)
        # gloss may be "|"-joined (ASLLRP-style) or a single label (WLASL)
        for g in sample.metadata.gloss.split("|"):
            if g:
                unique_glosses.add(g)

        if i % 2000 == 0:
            print(f"  Scanned {i}/{len(npz_files)}...")

    sorted_glosses = sorted(unique_glosses)
    vocab = SPECIAL_TOKENS + sorted_glosses

    token_to_id = {tok: idx for idx, tok in enumerate(vocab)}

    output_path = Path("data/vocab.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(token_to_id, f, indent=2)

    print(f"\nVocab size (incl. special tokens): {len(vocab)}")
    print(f"Unique glosses found: {len(sorted_glosses)}")
    print(f"Written to: {output_path}")


if __name__ == "__main__":
    main()