"""
dataset.py

PyTorch Dataset + collate_fn for variable-length landmark sequences.

Day 32 additions:
  1. ASLLRP gloss-to-ID lookup now routes through asllrp_token_to_id.json
     (Day 29's normalized mapping) when source == "asllrp", instead of a
     raw string lookup against the merged vocab directly. The raw lookup
     was flagged as a blocker back on Day 30: ASLLRP's raw gloss strings
     (IX-3p:i, DCL:B-L"...", fs-JESSICA, etc.) don't match vocab_merged.json's
     keys directly -- almost everything silently fell through to <UNK>.
  2. Per-timestep boundary targets (Day 31/32) are now built here, using
     each sample's frame_indices + sign_spans metadata (added during the
     Day 32 ASLLRP re-extraction) and boundary_labels.build_boundary_targets().
     WLASL samples have no sign_spans concept -- boundary_target is an
     all-zero array of the right length for those, not a crash or a None.

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

import numpy as np
import torch
from torch.utils.data import Dataset

from src.preprocessing.sample_writer import load_sample
from src.models.feature_utils import flatten_frame_features
from src.preprocessing.boundary_labels import build_boundary_targets, build_transition_targets
from src.models.augmentation import apply_spatial_augmentation


class SignLanguageDataset(Dataset):
    def __init__(
        self,
        npz_dir,
        vocab_path="data/vocab.json",
        asllrp_token_to_id_path="data/asllrp_token_to_id.json",
        augment=False,
        include_transition_target=False,
    ):
        # Stage B1: opt-in, additive, default OFF -- byte-identical
        # __getitem__ behavior when unset, same discipline as every other
        # feature this project has added. Caller's responsibility (not
        # this class's) to only ever pass augment=True for a TRAIN split
        # dataset -- see src/models/augmentation.py for the transform
        # itself and scripts/stageB1_augmentation_smoke_test.py for where
        # that responsibility is actually exercised (train=True, val/
        # test=False, always).
        self.augment = augment

        # Boundary-mechanisms smoke test: opt-in, additive, default OFF --
        # byte-identical __getitem__ output (same dict keys as before)
        # when unset, same discipline as `augment` above. See
        # boundary_labels.build_transition_targets' docstring for what
        # this is and why; collate_fn only pads/returns this key when
        # it is actually present on the batch's items, so a caller that
        # never sets this flag sees no change to collate_fn's output
        # either.
        self.include_transition_target = include_transition_target

        self.npz_paths = sorted(Path(npz_dir).glob("*.npz"))
        if not self.npz_paths:
            raise RuntimeError(f"No .npz files found in {npz_dir}")

        with open(vocab_path, "r", encoding="utf-8") as f:
            self.token_to_id = json.load(f)

        self.bos_id = self.token_to_id["<BOS>"]
        self.eos_id = self.token_to_id["<EOS>"]
        self.unk_id = self.token_to_id["<UNK>"]
        self.pad_id = self.token_to_id["<PAD>"]

        # Day 32 fix: ASLLRP gloss strings need Day 29's normalization
        # (case-fold, suffix-collapse, fingerspelling/classifier
        # placeholder routing) before vocab lookup -- a raw .get() call
        # against vocab_merged.json's keys directly silently resolves
        # almost everything to <UNK>, since ASLLRP's raw strings
        # (IX-3p:i, DCL:B-L"...", fs-JESSICA) don't match those keys.
        # This mapping is optional at construction time so WLASL-only
        # datasets (which have no ASLLRP samples) don't need the file.
        self.asllrp_token_to_id = None
        if Path(asllrp_token_to_id_path).exists():
            with open(asllrp_token_to_id_path, "r", encoding="utf-8") as f:
                self.asllrp_token_to_id = json.load(f)

    def __len__(self):
        return len(self.npz_paths)

    def _gloss_to_id(self, raw_gloss: str, source: str) -> int:
        """
        Routes gloss-string-to-ID lookup through the correct mapping
        depending on source. ASLLRP raw strings MUST go through
        asllrp_token_to_id.json's normalization; WLASL strings are
        already clean lowercase words matching vocab.json's keys
        directly, so they use the plain lookup as before.
        """
        if source == "asllrp" and self.asllrp_token_to_id is not None:
            # Day 29's precomputed mapping already resolves case,
            # numbered-suffix collapsing, and fingerspelling/classifier
            # placeholder routing -- if a raw string isn't in this
            # mapping at all, that's a real gap worth knowing about
            # (not silently defaulting to <UNK> without a trace), so
            # this intentionally does NOT catch a KeyError here.
            token_or_id = self.asllrp_token_to_id.get(raw_gloss)
            if token_or_id is None:
                return self.unk_id
            # asllrp_token_to_id.json maps raw string -> final numeric
            # ID directly (per Day 29's doc: "precomputed raw-gloss-
            # string -> final-ID lookup"). If it instead maps to a
            # normalized token STRING rather than an int, this needs
            # one more indirection through self.token_to_id -- confirm
            # against the real file contents before trusting this line
            # as-is (see note below).
            return token_or_id if isinstance(token_or_id, int) else self.token_to_id.get(token_or_id, self.unk_id)
        return self.token_to_id.get(raw_gloss, self.unk_id)

    def __getitem__(self, idx):
        sample = load_sample(self.npz_paths[idx])

        # Stage B1: spatial-only augmentation, opt-in. Applied AFTER
        # load_sample() (so disk contents are never touched) and BEFORE
        # flatten_frame_features() (so it's real coordinate values that
        # change, not the flattened feature vector after the fact --
        # equivalent either way, but this order keeps augmentation.py
        # working on the same structured, semantically-labeled arrays
        # the rest of the preprocessing pipeline uses). Frame count,
        # metadata (including sign_spans/frame_indices), and gloss are
        # passed through completely untouched -- see augmentation.py's
        # docstring for why that's true by construction, not just
        # asserted. Fresh randomness every call (no caching here or in
        # augmentation.py), so this is naturally per-epoch, not fixed.
        if self.augment:
            sample = apply_spatial_augmentation(sample)

        meta = sample.metadata

        features = flatten_frame_features(sample)  # (T, RAW_FEATURE_DIM)
        landmarks = torch.tensor(features, dtype=torch.float32)
        T = landmarks.shape[0]

        raw_glosses = meta.gloss.split("|")
        gloss_ids = [self._gloss_to_id(g, meta.source) for g in raw_glosses]
        target_tokens = torch.tensor([self.bos_id] + gloss_ids + [self.eos_id], dtype=torch.long)

        # Day 32: per-timestep boundary target. WLASL has no sign_spans
        # concept (frame_indices/sign_spans are None for those samples,
        # since they were never populated for WLASL's isolated-sign
        # format) -- boundary_target is an all-zero array of length T
        # in that case, not a crash and not fabricated ground truth.
        if meta.source == "asllrp" and meta.frame_indices is not None and meta.sign_spans is not None:
            sign_spans_as_tuples = [
                tuple(s) if s is not None else None for s in meta.sign_spans
            ]
            boundary_target, _stats = build_boundary_targets(
                sign_spans=sign_spans_as_tuples,
                sampled_frame_indices=meta.frame_indices,
            )
        else:
            boundary_target = np.zeros(T, dtype=np.float32)

        item = {
            "landmarks": landmarks,          # (T, RAW_FEATURE_DIM)
            "target_tokens": target_tokens,  # (tgt_len,)
            "boundary_target": torch.tensor(boundary_target, dtype=torch.float32),  # (T,)
            "video_id": meta.video_id,
            "num_frames": meta.num_frames,   # for mask verification
            "source": meta.source,
        }

        # Boundary-mechanisms smoke test: opt-in second target, same
        # source data (sign_spans/frame_indices) and same WLASL
        # fallback-to-zeros behavior as boundary_target above -- see
        # build_transition_targets' docstring for the mechanism.
        if self.include_transition_target:
            if meta.source == "asllrp" and meta.frame_indices is not None and meta.sign_spans is not None:
                transition_target, _tstats = build_transition_targets(
                    sign_spans=sign_spans_as_tuples,
                    sampled_frame_indices=meta.frame_indices,
                )
            else:
                transition_target = np.zeros(T, dtype=np.float32)
            item["transition_target"] = torch.tensor(transition_target, dtype=torch.float32)

        return item


def collate_fn(batch, pad_id=0):
    """
    Pads variable-length landmark sequences, target token sequences,
    and per-timestep boundary targets to the batch's max length.
    Produces:
      - encoder_attention_mask: True/1 = real frame, False/0 = padding
      - target_padding_mask: True/1 = real token, False/0 = padding
      - boundary_target: padded to max_seq_len, aligned 1:1 with
        landmarks/encoder_attention_mask, NOT with target_tokens --
        this is a per-encoder-timestep label, not a per-gloss-token one.
    """
    batch_size = len(batch)

    max_seq_len = max(item["landmarks"].shape[0] for item in batch)
    max_tgt_len = max(item["target_tokens"].shape[0] for item in batch)
    feature_dim = batch[0]["landmarks"].shape[1]

    landmarks_padded = torch.zeros(batch_size, max_seq_len, feature_dim, dtype=torch.float32)
    encoder_attention_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.bool)
    boundary_target_padded = torch.zeros(batch_size, max_seq_len, dtype=torch.float32)

    target_tokens_padded = torch.full((batch_size, max_tgt_len), pad_id, dtype=torch.long)
    target_padding_mask = torch.zeros(batch_size, max_tgt_len, dtype=torch.bool)

    video_ids = []
    num_frames_list = []
    sources = []

    # Boundary-mechanisms smoke test: opt-in, additive. Only allocated/
    # returned when the dataset actually produced this key (i.e.
    # include_transition_target=True) -- a caller that never sets that
    # flag gets a return dict with EXACTLY the same keys as before this
    # change, byte-identical.
    has_transition_target = "transition_target" in batch[0]
    transition_target_padded = (
        torch.zeros(batch_size, max_seq_len, dtype=torch.float32) if has_transition_target else None
    )

    for i, item in enumerate(batch):
        seq_len = item["landmarks"].shape[0]
        tgt_len = item["target_tokens"].shape[0]

        landmarks_padded[i, :seq_len, :] = item["landmarks"]
        encoder_attention_mask[i, :seq_len] = True
        boundary_target_padded[i, :seq_len] = item["boundary_target"]

        target_tokens_padded[i, :tgt_len] = item["target_tokens"]
        target_padding_mask[i, :tgt_len] = True

        if has_transition_target:
            transition_target_padded[i, :seq_len] = item["transition_target"]

        video_ids.append(item["video_id"])
        num_frames_list.append(item["num_frames"])
        sources.append(item["source"])

    out = {
        "landmarks": landmarks_padded,
        "encoder_attention_mask": encoder_attention_mask,
        "target_tokens": target_tokens_padded,
        "target_padding_mask": target_padding_mask,
        "boundary_target": boundary_target_padded,
        "video_ids": video_ids,
        "num_frames": num_frames_list,
        "sources": sources,
    }
    if has_transition_target:
        out["transition_target"] = transition_target_padded
    return out