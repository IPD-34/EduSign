"""
test_day10_dataloader.py

Day 10 validation: DataLoader iterates over a real WLASL subset,
batches are correctly padded/masked, and feeding a batch through
Day 9's model produces output with no shape mismatches.

Also does the manual mask-sum spot-check the plan calls for:
each sample's attention_mask.sum() should equal its known num_frames
from the .npz metadata -- a mismatch means the padding logic is wrong,
not the model.
"""

import json
import sys
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.config import WLASL_PROCESSED_DIR
from src.models.dataset import SignLanguageDataset, collate_fn
from src.models.sign_transformer import SignTransformer


def main():
    with open("data/vocab.json", "r", encoding="utf-8") as f:
        token_to_id = json.load(f)
    vocab_size = len(token_to_id)
    pad_id = token_to_id["<PAD>"]

    dataset = SignLanguageDataset(WLASL_PROCESSED_DIR / "train")
    print(f"Dataset size: {len(dataset)}")

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=partial(collate_fn, pad_id=pad_id),
    )

    batch = next(iter(dataloader))
    print(f"landmarks shape: {batch['landmarks'].shape}")               # (B, T_max, feature_dim)
    print(f"encoder_attention_mask shape: {batch['encoder_attention_mask'].shape}")  # (B, T_max)
    print(f"target_tokens shape: {batch['target_tokens'].shape}")
    print(f"target_padding_mask shape: {batch['target_padding_mask'].shape}")

    mask_sums = batch["encoder_attention_mask"].sum(dim=1).tolist()
    print(f"\nattention_mask.sum() per sample: {mask_sums}")
    print(f"known num_frames per sample:      {batch['num_frames']}")

    mismatches = [
        (vid, m, n) for vid, m, n in zip(batch["video_ids"], mask_sums, batch["num_frames"]) if m != n
    ]
    if mismatches:
        print(f"\n[MISMATCH] {len(mismatches)} samples where mask sum != num_frames:")
        for vid, m, n in mismatches:
            print(f"  {vid}: mask_sum={m}, num_frames={n}")
    else:
        print("\nAll mask sums match known num_frames. Padding logic verified correct.")

    # Feed batch through Day 9's model
    model = SignTransformer(vocab_size=vocab_size, pad_token_id=pad_id)
    logits = model(
        batch["landmarks"],
        batch["encoder_attention_mask"],
        batch["target_tokens"],
        batch["target_padding_mask"],
    )
    print(f"\nlogits shape: {logits.shape}")  # expect (B, tgt_len_max, vocab_size)


if __name__ == "__main__":
    main()