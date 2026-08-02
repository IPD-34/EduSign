"""
dataset.py

PyTorch Dataset + collate_fn for variable-length landmark sequences.

Key distinction (per the interface spec, and worth being careful about):
padding mask != presence flag.
  - A PADDED frame (added just to make the batch rectangular) is fully
    masked out of attention -- the model never sees it.
  - A PRESENT-BUT-OCCLUDED real frame (presence=False for some channel,
    e.g. a hand briefly out of frame) is a genuine frame in the real
    sequence -- it still participates in attention, just with
    zero-filled coordinates for whichever channel was occluded.
Confusing these two silently degrades training without crashing --
exactly the failure mode Day 10 is meant to guard against.
"""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.preprocessing.sample_writer import load_sample
from src.models.feature_utils import flatten_frame_features


class SignLanguageDataset(Dataset):
    def __init__(self, npz_dir, vocab_path="data/vocab.json"):
        self.npz_paths = sorted(Path(npz_dir).glob("*.npz"))
        if not self.npz_paths:
            raise RuntimeError(f"No .npz files found in {npz_dir}")

        with open(vocab_path, "r", encoding="utf-8") as f:
            self.token_to_id = json.load(f)

        self.bos_id = self.token_to_id["<BOS>"]
        self.eos_id = self.token_to_id["<EOS>"]
        self.unk_id = self.token_to_id["<UNK>"]
        self.pad_id = self.token_to_id["<PAD>"]

    def __len__(self):
        return len(self.npz_paths)

    def __getitem__(self, idx):
        sample = load_sample(self.npz_paths[idx])

        features = flatten_frame_features(sample)  # (T, RAW_FEATURE_DIM)
        landmarks = torch.tensor(features, dtype=torch.float32)

        gloss_ids = [self.token_to_id.get(g, self.unk_id) for g in sample.metadata.gloss.split("|")]
        target_tokens = torch.tensor([self.bos_id] + gloss_ids + [self.eos_id], dtype=torch.long)

        return {
            "landmarks": landmarks,          # (T, RAW_FEATURE_DIM)
            "target_tokens": target_tokens,  # (tgt_len,)
            "video_id": sample.metadata.video_id,
            "num_frames": sample.metadata.num_frames,  # for mask verification
        }


def collate_fn(batch, pad_id=0):
    """
    Pads variable-length landmark sequences and target token sequences
    to the batch's max length. Produces:
      - encoder_attention_mask: True/1 = real frame, False/0 = padding
      - target_padding_mask: True/1 = real token, False/0 = padding
    """
    batch_size = len(batch)

    max_seq_len = max(item["landmarks"].shape[0] for item in batch)
    max_tgt_len = max(item["target_tokens"].shape[0] for item in batch)
    feature_dim = batch[0]["landmarks"].shape[1]

    landmarks_padded = torch.zeros(batch_size, max_seq_len, feature_dim, dtype=torch.float32)
    encoder_attention_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.bool)

    target_tokens_padded = torch.full((batch_size, max_tgt_len), pad_id, dtype=torch.long)
    target_padding_mask = torch.zeros(batch_size, max_tgt_len, dtype=torch.bool)

    video_ids = []
    num_frames_list = []

    for i, item in enumerate(batch):
        seq_len = item["landmarks"].shape[0]
        tgt_len = item["target_tokens"].shape[0]

        landmarks_padded[i, :seq_len, :] = item["landmarks"]
        encoder_attention_mask[i, :seq_len] = True

        target_tokens_padded[i, :tgt_len] = item["target_tokens"]
        target_padding_mask[i, :tgt_len] = True

        video_ids.append(item["video_id"])
        num_frames_list.append(item["num_frames"])

    return {
        "landmarks": landmarks_padded,
        "encoder_attention_mask": encoder_attention_mask,
        "target_tokens": target_tokens_padded,
        "target_padding_mask": target_padding_mask,
        "video_ids": video_ids,
        "num_frames": num_frames_list,
    }