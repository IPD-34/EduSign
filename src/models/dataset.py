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

Robustness-layer wiring fix: __getitem__ now runs hold_last_value_sample()
then normalize_landmarks_sample() on the loaded sample BEFORE
flatten_frame_features(), in that order (hold-last-value must run first --
normalization needs valid coordinates to compute head_center/head_size
from). Previously this module flattened the raw loaded sample directly;
robustness/hold_last_value.py and robustness/normalisation.py existed and
were tested but were never actually reached by the live training/inference
path.

Wiring this in first surfaced a real bug (now fixed): normalize_landmarks_sample
used to decide frame validity from sample.presence.face alone, which
hold_last_value_sample deliberately never mutates -- so a hold-filled frame
was still marked NaN despite holding a real, carried-forward value, which
then poisoned 100% of a sample's model output via self-attention mixing on
31/41 real ASLLRP samples. Fixed via
hold_last_value.compute_fill_validity() + normalize_landmarks_sample's new
face_validity parameter -- see the inline comment below and both modules'
docstrings for the full explanation.

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
from src.preprocessing.boundary_labels import build_boundary_targets
from robustness.hold_last_value import compute_fill_validity, hold_last_value_sample
from robustness.normalisation import normalize_landmarks_sample


class SignLanguageDataset(Dataset):
    def __init__(
        self,
        npz_dir,
        vocab_path="data/vocab.json",
        asllrp_token_to_id_path="data/asllrp_token_to_id.json",
    ):
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
        meta = sample.metadata

        # Wires the Month 1 robustness layer into the live data path:
        # previously flatten_frame_features ran on the raw loaded sample
        # (zero-filled at missing frames, un-normalized coordinates) --
        # hold_last_value_sample() must run BEFORE normalize_landmarks_sample()
        # here, since normalization needs valid (filled, non-zero) coordinates
        # to compute head_center/head_size from; running it in the other
        # order would compute the normalization reference off already-zeroed
        # frames.
        #
        # BUG FIX: normalize_landmarks_sample used to decide "does this frame
        # have a usable face reference" from sample.presence.face alone --
        # which hold_last_value_sample deliberately never changes (see that
        # module's docstring), so a hold-filled frame was STILL marked NaN
        # even though it now holds a real, carried-forward face value. On
        # real ASLLRP data this poisoned 31/41 samples down to 100% NaN
        # model output via self-attention mixing. Fix: compute each frame's
        # POST-FILL validity (originally present, OR successfully held --
        # see compute_fill_validity) and pass that to normalize_landmarks_sample
        # via face_validity, instead of letting it fall back to raw presence.
        # A frame is still correctly left NaN if hold-last-value itself
        # couldn't fill it (leading dropout before any valid frame exists).
        filled = hold_last_value_sample(sample)
        face_fill_validity = compute_fill_validity(sample.presence.face)
        normalized = normalize_landmarks_sample(filled, face_validity=face_fill_validity)

        # REMAINING BUG (fixed here): a frame with no valid observation to
        # hold from AT ALL yet -- leading dropout before the first face was
        # ever seen -- is still correctly left NaN by normalize_landmarks_sample
        # (there is no reference to normalize against). But the encoder's
        # self-attention has no notion of "this timestep is undefined" --
        # SignEncoder.forward already accepts and applies an attention_mask
        # (wired to nn.TransformerEncoder's src_key_padding_mask), it was
        # just never told about anything besides padding. A single NaN
        # timestep poisons 100% of that sample's output via attention
        # mixing (confirmed directly: ~0.5% NaN input -> 100% NaN
        # ctc_log_probs). Fix: mark these timesteps invalid in the SAME
        # mask that already excludes padding (encoder_attention_mask,
        # built in collate_fn below) -- standard key-padding-mask usage,
        # no changes needed to sign_transformer.py/encoder.py itself.
        #
        # Checked empirically across all 41 real samples (mismatches: 0/41):
        # NaN currently only ever comes from face validity (normalize_landmarks_sample
        # applies one shared valid_frames mask to all four groups) -- pose/
        # hand dropout never independently produces NaN today. OR-ed across
        # all four groups anyway (rather than checking pose alone) so this
        # stays correct if that ever changes.
        invalid_timestep = (
            np.isnan(normalized.pose[:, 0, 0])
            | np.isnan(normalized.face[:, 0, 0])
            | np.isnan(normalized.left_hand[:, 0, 0])
            | np.isnan(normalized.right_hand[:, 0, 0])
        )
        landmark_validity = ~invalid_timestep  # (T,) bool -- True = safe to attend to/from

        features = flatten_frame_features(normalized)  # (T, RAW_FEATURE_DIM)
        # Safe placeholder for masked-out timesteps, applied AFTER
        # landmark_validity is derived from the NaN pattern above (this
        # must not run before that, or there would be nothing left to
        # detect). Zeroing is not enough on its own without the mask above:
        # a masked KEY's attention weight is exactly 0.0, but
        # 0.0 * NaN == NaN in IEEE float, which would still poison the
        # weighted sum for every other (valid) position. The mask makes
        # this timestep's OWN output irrelevant (nothing downstream should
        # read it); the zeroing makes sure it can't corrupt anyone else's.
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
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

        return {
            "landmarks": landmarks,          # (T, RAW_FEATURE_DIM)
            "target_tokens": target_tokens,  # (tgt_len,)
            "boundary_target": torch.tensor(boundary_target, dtype=torch.float32),  # (T,)
            "landmark_validity": torch.tensor(landmark_validity, dtype=torch.bool),  # (T,)
            "video_id": meta.video_id,
            "num_frames": meta.num_frames,   # for mask verification
            "source": meta.source,
        }


def collate_fn(batch, pad_id=0):
    """
    Pads variable-length landmark sequences, target token sequences,
    and per-timestep boundary targets to the batch's max length.
    Produces:
      - encoder_attention_mask: True/1 = attend to this timestep,
        False/0 = do not (padding OR a genuinely never-observed timestep
        -- see landmark_validity below). Fed directly into
        SignEncoder's src_key_padding_mask, so both padding and
        never-observed frames are excluded from self-attention the same
        way -- neither attended TO nor meaningfully attended FROM.
      - target_padding_mask: True/1 = real token, False/0 = padding
      - boundary_target: padded to max_seq_len, aligned 1:1 with
        landmarks/encoder_attention_mask, NOT with target_tokens --
        this is a per-encoder-timestep label, not a per-gloss-token one.

    encoder_attention_mask used to be True for every real (non-padded)
    frame unconditionally -- but a frame with genuinely no valid
    observation yet (leading dropout before hold_last_value_sample has
    anything to hold from) is left NaN by normalize_landmarks_sample, and
    that landmarks tensor is now nan_to_num'd to 0.0 in __getitem__ (a
    real, un-masked zero would look like legitimate data). Excluding it
    from encoder_attention_mask here -- via each item's own
    landmark_validity -- is what actually keeps that placeholder zero
    from being attended to. Without this, self-attention's global mixing
    turns a single such frame into 100% NaN output for the whole sample
    (confirmed directly against best_val.pt).
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

    for i, item in enumerate(batch):
        seq_len = item["landmarks"].shape[0]
        tgt_len = item["target_tokens"].shape[0]

        landmarks_padded[i, :seq_len, :] = item["landmarks"]
        encoder_attention_mask[i, :seq_len] = item["landmark_validity"]
        boundary_target_padded[i, :seq_len] = item["boundary_target"]

        target_tokens_padded[i, :tgt_len] = item["target_tokens"]
        target_padding_mask[i, :tgt_len] = True

        video_ids.append(item["video_id"])
        num_frames_list.append(item["num_frames"])
        sources.append(item["source"])

    return {
        "landmarks": landmarks_padded,
        "encoder_attention_mask": encoder_attention_mask,
        "target_tokens": target_tokens_padded,
        "target_padding_mask": target_padding_mask,
        "boundary_target": boundary_target_padded,
        "video_ids": video_ids,
        "num_frames": num_frames_list,
        "sources": sources,
    }