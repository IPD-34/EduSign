"""
test_day9_smoke.py

Day 9 validation: one real preprocessed sample through the full
encoder-decoder, forward pass + dummy backward pass. Confirms shapes
are correct and gradients flow -- not real training, just structural
validation before investing in Day 10's dataloader/Day 11's training loop.
"""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.config import WLASL_PROCESSED_DIR
from src.preprocessing.sample_writer import load_sample
from src.models.sign_transformer import SignTransformer
from src.models.feature_utils import flatten_frame_features


def main():
    # Load vocab
    with open("data/vocab.json", "r", encoding="utf-8") as f:
        token_to_id = json.load(f)
    vocab_size = len(token_to_id)
    print(f"Vocab size: {vocab_size}")

    # Load one real sample
    npz_files = list((WLASL_PROCESSED_DIR / "train").glob("*.npz"))
    if not npz_files:
        raise RuntimeError("No processed WLASL train samples found.")
    sample = load_sample(npz_files[0])
    print(f"Loaded sample: {sample.metadata.video_id}, gloss='{sample.metadata.gloss}', T={sample.metadata.num_frames}")

    # Build model input
    features = flatten_frame_features(sample)  # (T, RAW_FEATURE_DIM)
    landmarks = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # (1, T, RAW_FEATURE_DIM)
    encoder_attention_mask = torch.ones(1, features.shape[0], dtype=torch.bool)  # no padding, single real sample

    # Build target token sequence: <BOS> gloss <EOS>
    bos_id = token_to_id["<BOS>"]
    eos_id = token_to_id["<EOS>"]
    unk_id = token_to_id["<UNK>"]
    gloss_ids = [token_to_id.get(g, unk_id) for g in sample.metadata.gloss.split("|")]
    target_tokens = torch.tensor([[bos_id] + gloss_ids + [eos_id]], dtype=torch.long)  # (1, tgt_len)

    print(f"Landmarks shape: {landmarks.shape}")
    print(f"Target tokens shape: {target_tokens.shape}")

    # Build model
    model = SignTransformer(vocab_size=vocab_size)

    # Forward pass
    logits = model(landmarks, encoder_attention_mask, target_tokens)
    print(f"Logits shape: {logits.shape}")  # expect (1, tgt_len, vocab_size)

    # Dummy backward pass
    dummy_loss = logits.sum()
    dummy_loss.backward()
    print("Backward pass completed without error.")

    # Confirm gradients actually flowed
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    print(f"Gradients flowed through model: {has_grad}")


if __name__ == "__main__":
    main()