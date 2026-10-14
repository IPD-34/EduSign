"""
test_dataset_attention_masking.py

Tests for the fix to a real bug: a genuinely never-observed leading-dropout
timestep (no valid face reference yet, per hold_last_value_sample +
normalize_landmarks_sample) was left as a NaN-turned-zero placeholder in
the model input with NOTHING telling the encoder's self-attention to
ignore it -- confirmed directly to poison 100% of that sample's
ctc_log_probs via attention mixing, on 30/41 real ASLLRP samples.

Fix: src/models/dataset.py's SignLanguageDataset.__getitem__ now derives a
per-timestep landmark_validity mask (OR-ed across all four groups' NaN
pattern -- checked empirically that only face validity ever actually
drives it today, see that file's inline comment) and nan_to_num's the
placeholder AFTER deriving it; collate_fn now builds encoder_attention_mask
from each item's landmark_validity (ANDed implicitly with padding, since
padded positions are never written and stay at their zero-initialized
False) instead of unconditionally True for every real frame.

No model/checkpoint is loaded here -- these tests check the masking
plumbing itself (collate_fn's combination logic, and that real samples
produce NaN-free landmarks with a validity mask matching the known
leading-dropout pattern), not model output. The end-to-end "does the
model actually produce clean output now" check was done manually against
best_val.pt (0/41 NaN, confirmed) -- exercising that here would mean
loading a 70MB checkpoint into every test run, which the rest of this
suite deliberately avoids.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import pytest

from src.models.dataset import SignLanguageDataset, collate_fn

_NPZ_DIRS = ["data/processed/asllrp/train", "data/processed/asllrp/val"]


def _real_samples_available() -> bool:
    return any(Path(d).exists() and any(Path(d).glob("*.npz")) for d in _NPZ_DIRS)


def _make_item(seq_len: int, feature_dim: int, valid_mask: list[bool], tgt_len: int = 3) -> dict:
    return {
        "landmarks": torch.zeros(seq_len, feature_dim, dtype=torch.float32),
        "target_tokens": torch.zeros(tgt_len, dtype=torch.long),
        "boundary_target": torch.zeros(seq_len, dtype=torch.float32),
        "landmark_validity": torch.tensor(valid_mask, dtype=torch.bool),
        "video_id": "hand_built",
        "num_frames": seq_len,
        "source": "synthetic",
    }


# ---------------------------------------------------------------------
# collate_fn: encoder_attention_mask must combine padding AND validity
# ---------------------------------------------------------------------


def test_collate_fn_masks_out_invalid_timestep_within_real_region():
    item = _make_item(seq_len=4, feature_dim=8, valid_mask=[False, True, True, True])

    batch = collate_fn([item])

    # Frame 0 is within the real (non-padded) region but marked invalid --
    # must be False in encoder_attention_mask, not just "real so True".
    np.testing.assert_array_equal(
        batch["encoder_attention_mask"][0].tolist(), [False, True, True, True]
    )


def test_collate_fn_combines_validity_with_padding_correctly():
    # Two items: one shorter (will be padded), one with an invalid leading frame.
    short_item = _make_item(seq_len=2, feature_dim=8, valid_mask=[True, True])
    long_item_with_invalid = _make_item(seq_len=4, feature_dim=8, valid_mask=[False, True, True, True])

    batch = collate_fn([short_item, long_item_with_invalid])

    # short_item: real region [0,1] all valid, padded region [2,3] must be False.
    np.testing.assert_array_equal(batch["encoder_attention_mask"][0].tolist(), [True, True, False, False])
    # long_item: frame 0 invalid (not padding -- it's within its own real length).
    np.testing.assert_array_equal(batch["encoder_attention_mask"][1].tolist(), [False, True, True, True])


def test_collate_fn_all_valid_matches_old_behavior():
    """Sanity check: when every frame is valid (the common case), behavior
    is identical to the pre-fix 'always True for real frames' logic."""
    item = _make_item(seq_len=3, feature_dim=8, valid_mask=[True, True, True])

    batch = collate_fn([item])

    np.testing.assert_array_equal(batch["encoder_attention_mask"][0].tolist(), [True, True, True])


# ---------------------------------------------------------------------
# End-to-end (no model) over all 41 real ASLLRP samples
# ---------------------------------------------------------------------


pytestmark_real = pytest.mark.skipif(
    not _real_samples_available(), reason="real ASLLRP samples not found under data/processed/asllrp/"
)


@pytestmark_real
def test_real_samples_produce_no_nan_landmarks():
    for npz_dir in _NPZ_DIRS:
        ds = SignLanguageDataset(npz_dir=npz_dir)
        for i in range(len(ds)):
            item = ds[i]
            assert not torch.isnan(item["landmarks"]).any(), (
                f"{item['video_id']}: NaN found in landmarks after nan_to_num -- "
                f"placeholder substitution should make this impossible"
            )


@pytestmark_real
def test_real_samples_landmark_validity_matches_known_leading_dropout_pattern():
    """
    Regression pin for the specific finding this fix addresses: across all
    41 real samples, invalid-timestep count per sample should be small
    (leading dropout only, 0-2 frames per sample per the investigation),
    never a large fraction of the sequence -- a large invalid count would
    indicate something other than leading dropout is now being masked,
    worth a fresh look rather than assuming it's fine.
    """
    total_invalid = 0
    total_frames = 0
    samples_with_invalid = 0

    for npz_dir in _NPZ_DIRS:
        ds = SignLanguageDataset(npz_dir=npz_dir)
        for i in range(len(ds)):
            item = ds[i]
            validity = item["landmark_validity"]
            invalid_count = int((~validity).sum())
            total_invalid += invalid_count
            total_frames += validity.shape[0]
            if invalid_count > 0:
                samples_with_invalid += 1
                assert invalid_count <= 2, (
                    f"{item['video_id']}: {invalid_count} invalid timesteps -- "
                    f"expected only small leading-dropout counts (<=2)"
                )

    assert total_frames > 0
    # Known finding: ~30/41 samples have exactly 1 (one has 2) leading
    # invalid frame; total invalid should be a tiny fraction of all frames.
    assert total_invalid / total_frames < 0.01
