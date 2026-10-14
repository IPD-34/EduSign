"""
make_sample.py

Synthetic-data helpers for tests and timing scripts that need a
ProcessedSample without depending on real WLASL/ASLLRP preprocessing
(which is not available yet).

Not a pytest conftest / fixture module -- just plain functions that
tests and scripts import directly, e.g.:

    from robustness.tests.fixtures.make_sample import make_synthetic_sample

Data contract reminders (see src/preprocessing/sample_schema.py):
- Four separate landmark groups: pose (T, 33, 4), left_hand (T, 21, 3),
  right_hand (T, 21, 3), face (T, 478, 3). pose's 4th channel is
  visibility; the other three groups have only (x, y, z).
- Missingness is tracked explicitly per group via LandmarkPresence
  boolean (T,) arrays -- never via NaN-scanning.
- A frame where a group is absent (presence == False) is zero-filled
  for that group at that frame, not NaN-filled.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import numpy as np

from src.preprocessing.sample_schema import (
    LandmarkPresence,
    ProcessedSample,
    SampleMetadata,
)
from src.preprocessing.sample_writer import load_sample, write_sample

# (group_name -> (num_landmarks, num_channels))
_GROUP_SHAPES: Dict[str, tuple] = {
    "pose": (33, 4),
    "left_hand": (21, 3),
    "right_hand": (21, 3),
    "face": (478, 3),
}

DropoutSpec = Union[float, Sequence[int], None]


def _resolve_missing_mask(
    rng: np.random.Generator,
    num_frames: int,
    dropout: DropoutSpec,
) -> np.ndarray:
    """
    Build a (T,) boolean "is missing" mask for one landmark group from a
    dropout spec, which may be:
      - None (default): no dropout, nothing missing.
      - float: independent per-frame dropout probability in [0, 1].
      - a sequence of explicit frame indices to mark as missing.

    Returns a boolean array where True means "missing at this frame".
    """
    missing = np.zeros(num_frames, dtype=bool)

    if dropout is None:
        return missing

    if isinstance(dropout, (int, float)) and not isinstance(dropout, bool):
        prob = float(dropout)
        if not 0.0 <= prob <= 1.0:
            raise ValueError(f"dropout_prob must be in [0, 1], got {prob}")
        if prob > 0.0:
            missing = rng.random(num_frames) < prob
        return missing

    # Otherwise treat as an explicit sequence of missing-frame indices.
    indices = np.asarray(list(dropout), dtype=np.int64)
    if indices.size and (indices.min() < 0 or indices.max() >= num_frames):
        raise ValueError(
            f"missing-frame indices must be in [0, {num_frames - 1}], "
            f"got range [{indices.min()}, {indices.max()}]"
        )
    missing[indices] = True
    return missing


def make_synthetic_sample(
    num_frames: int = 30,
    *,
    seed: Optional[int] = None,
    video_id: str = "synthetic_0001",
    gloss: str = "PLACEHOLDER",
    fps: float = 30.0,
    pose_dropout: DropoutSpec = None,
    left_hand_dropout: DropoutSpec = None,
    right_hand_dropout: DropoutSpec = None,
    face_dropout: DropoutSpec = None,
) -> ProcessedSample:
    """
    Build a synthetic ProcessedSample with random landmark data and a
    controllable per-group dropout (presence) pattern.

    Parameters
    ----------
    num_frames:
        T, the number of frames in the synthetic sample.
    seed:
        Seed for np.random.default_rng, for reproducibility. None means
        an unseeded (non-reproducible) generator.
    video_id, gloss, fps:
        Placeholder metadata values; split is always "train" and source
        is always "synthetic" per the contract for this helper.
    pose_dropout, left_hand_dropout, right_hand_dropout, face_dropout:
        Per-group dropout spec, each either:
          - None (default): that group is fully present every frame.
          - a float in [0, 1]: independent per-frame dropout probability.
          - a sequence of explicit frame indices to mark as missing.

    Returns
    -------
    ProcessedSample with random pose/left_hand/right_hand/face arrays,
    zero-filled at frames where the corresponding group is marked
    missing, plus a matching LandmarkPresence and SampleMetadata.
    """
    if num_frames <= 0:
        raise ValueError(f"num_frames must be > 0, got {num_frames}")

    rng = np.random.default_rng(seed)

    dropout_by_group = {
        "pose": pose_dropout,
        "left_hand": left_hand_dropout,
        "right_hand": right_hand_dropout,
        "face": face_dropout,
    }

    landmarks: Dict[str, np.ndarray] = {}
    presence: Dict[str, np.ndarray] = {}

    for group, (num_landmarks, num_channels) in _GROUP_SHAPES.items():
        arr = rng.random((num_frames, num_landmarks, num_channels)).astype(np.float32)

        if group == "pose":
            # Channels 0:3 are (x, y, z) spatial; channel 3 is visibility,
            # which must stay in [0, 1] (rng.random already yields that,
            # but this is spelled out explicitly for clarity/robustness
            # to future changes in how the other channels are generated).
            arr[:, :, 3] = rng.random((num_frames, num_landmarks)).astype(np.float32)

        missing_mask = _resolve_missing_mask(rng, num_frames, dropout_by_group[group])
        present_mask = ~missing_mask

        # Per the real data contract: an absent frame is zero-filled,
        # not NaN-filled.
        arr[missing_mask] = 0.0

        landmarks[group] = arr
        presence[group] = present_mask

    metadata = SampleMetadata(
        video_id=video_id,
        gloss=gloss,
        split="train",
        source="synthetic",
        fps=fps,
        num_frames=num_frames,
        translation=None,
    )

    landmark_presence = LandmarkPresence(
        pose=presence["pose"],
        left_hand=presence["left_hand"],
        right_hand=presence["right_hand"],
        face=presence["face"],
    )

    return ProcessedSample(
        metadata=metadata,
        pose=landmarks["pose"],
        left_hand=landmarks["left_hand"],
        right_hand=landmarks["right_hand"],
        face=landmarks["face"],
        presence=landmark_presence,
    )


_DEFAULT_REAL_STATS_PATH = "data/real_dropout_stats.json"


def _load_real_stats(real_stats_path: Union[str, Path]) -> dict:
    path = Path(real_stats_path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Generate it first with "
            "`uv run python -m scripts.extract_real_dropout_stats` "
            "(reads the real ASLLRP samples under data/processed/asllrp/)."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _sample_missing_mask_from_real_stats(
    rng: np.random.Generator,
    num_frames: int,
    run_lengths_pool: Sequence[int],
    target_fraction_missing: float,
    fraction_co_occurring: float,
    placed_spans: Dict[str, list],
    this_group: str,
) -> np.ndarray:
    """
    Builds one group's (T,) missing mask by bootstrap-resampling burst
    lengths from that group's REAL run_lengths pool (rather than drawing
    from a synthetic geometric/uniform assumption), placing bursts until
    the group's real aggregate missing-fraction is (approximately)
    reached.

    Co-occurrence: with probability `fraction_co_occurring` (the real,
    measured fraction of missing spans that overlap another channel's
    missing span -- see data/real_dropout_stats.json's co_occurrence
    block), a new burst is anchored to start at the same frame as an
    already-placed burst on a different channel, instead of a fully
    independent random position. This approximates the real
    single-channel-vs-multi-channel dropout mix; it does not attempt to
    exactly reproduce the full joint (which channels, how many at once)
    distribution -- see this module's docstring note on
    generate_realistic_dropout for what is and isn't modeled.
    """
    mask = np.zeros(num_frames, dtype=bool)
    if not run_lengths_pool or target_fraction_missing <= 0.0 or num_frames <= 0:
        return mask

    target_missing = int(round(target_fraction_missing * num_frames))
    if target_missing <= 0:
        return mask

    max_attempts = target_missing * 20 + 50  # generous cap: avoids an infinite loop
    # on a short sequence where placed bursts keep overlapping themselves.
    attempts = 0
    while int(mask.sum()) < target_missing and attempts < max_attempts:
        attempts += 1
        length = min(int(rng.choice(run_lengths_pool)), num_frames)

        start = None
        if rng.random() < fraction_co_occurring:
            other_groups = [g for g in placed_spans if g != this_group and placed_spans[g]]
            if other_groups:
                other_group = other_groups[rng.integers(len(other_groups))]
                other_spans = placed_spans[other_group]
                anchor_start, _anchor_end = other_spans[rng.integers(len(other_spans))]
                start = min(anchor_start, max(0, num_frames - length))

        if start is None:
            start = int(rng.integers(0, num_frames - length + 1)) if num_frames > length else 0

        end = start + length - 1
        mask[start : end + 1] = True
        placed_spans[this_group].append((start, end))

    return mask


def generate_realistic_dropout(
    num_frames: int = 30,
    *,
    seed: Optional[int] = None,
    video_id: str = "realistic_0001",
    gloss: str = "PLACEHOLDER",
    fps: float = 30.0,
    real_stats: Optional[dict] = None,
    real_stats_path: Union[str, Path] = _DEFAULT_REAL_STATS_PATH,
) -> ProcessedSample:
    """
    Build a synthetic ProcessedSample whose dropout pattern is sampled
    from the REAL per-channel run-length and cross-channel co-occurrence
    distributions measured from the actual 41 ASLLRP train+val samples
    (data/real_dropout_stats.json, produced by
    scripts/extract_real_dropout_stats.py) -- rather than the arbitrary
    per-frame-independent-probability or explicit-index dropout specs
    make_synthetic_sample uses.

    make_synthetic_sample is left completely unchanged; this is an
    additional, separate generator with its own (smaller) internal
    helpers, reusing only the read-only _GROUP_SHAPES constant.

    What this DOES reproduce from the real data:
      - per-channel burst-length distribution: burst lengths are
        bootstrap-resampled from that channel's actual observed
        run_lengths list, not drawn from a synthetic distribution
        (geometric, uniform, etc.) -- so e.g. left_hand's real long tail
        (bursts up to 22 frames) can appear here too, at roughly its real
        rate, alongside its much more common 1-2 frame blips.
      - per-channel aggregate missing-fraction, targeted via the total
        number of bootstrapped missing frames placed.
      - the overall rate at which missing spans co-occur across channels
        vs. stay isolated to one channel (real: ~35% co-occurring, see
        data/real_dropout_stats.json's co_occurrence.fraction_co_occurring),
        approximated by anchoring a configurable fraction of new bursts to
        start at the same frame as an already-placed burst on a different
        channel.

    What this does NOT attempt to reproduce exactly (documented
    simplifications, not silently smoothed over):
      - the real per-sample heterogeneity (some real samples are fully
        clean for pose/face, others carry most of the missing frames) --
        this generator targets the pooled aggregate fraction uniformly
        for every sample it produces, it does not model "some samples are
        much worse than others."
      - the exact joint distribution of HOW MANY channels drop out at
        once (real data shows a 0/1/2/4-simultaneous split with a notable
        4-at-once tail from full-occlusion frames, but never exactly 3) --
        the co-occurrence mechanism here approximates the overall
        isolated-vs-shared rate, not that specific joint shape.

    Parameters
    ----------
    num_frames, seed, video_id, gloss, fps:
        Same meaning as in make_synthetic_sample. split is always "train"
        and source is "synthetic_realistic" (distinct from
        make_synthetic_sample's "synthetic", so callers/inspectors can
        tell which generator produced a given sample).
    real_stats:
        Pre-loaded contents of data/real_dropout_stats.json, if the
        caller already has it (e.g. a test running this many times).
        Takes precedence over real_stats_path when given.
    real_stats_path:
        Path to data/real_dropout_stats.json, loaded if `real_stats` is
        not given. Raises FileNotFoundError (with the command to
        regenerate it) if missing.

    Returns
    -------
    ProcessedSample, same shape/dtype contract as make_synthetic_sample:
    random landmark data, zero-filled at frames marked missing by the
    sampled masks, matching LandmarkPresence and SampleMetadata.
    """
    if num_frames <= 0:
        raise ValueError(f"num_frames must be > 0, got {num_frames}")

    if real_stats is None:
        real_stats = _load_real_stats(real_stats_path)

    rng = np.random.default_rng(seed)

    per_channel_stats = real_stats["per_channel"]
    fraction_co_occurring = real_stats["co_occurrence"].get("fraction_co_occurring") or 0.0

    placed_spans: Dict[str, list] = {group: [] for group in _GROUP_SHAPES}
    missing_masks: Dict[str, np.ndarray] = {}
    for group in _GROUP_SHAPES:
        stats = per_channel_stats[group]
        missing_masks[group] = _sample_missing_mask_from_real_stats(
            rng,
            num_frames,
            stats["run_lengths"],
            stats["fraction_missing"],
            fraction_co_occurring,
            placed_spans,
            group,
        )

    landmarks: Dict[str, np.ndarray] = {}
    presence: Dict[str, np.ndarray] = {}

    for group, (num_landmarks, num_channels) in _GROUP_SHAPES.items():
        arr = rng.random((num_frames, num_landmarks, num_channels)).astype(np.float32)

        if group == "pose":
            arr[:, :, 3] = rng.random((num_frames, num_landmarks)).astype(np.float32)

        missing_mask = missing_masks[group]
        present_mask = ~missing_mask

        # Per the real data contract: an absent frame is zero-filled, not NaN-filled.
        arr[missing_mask] = 0.0

        landmarks[group] = arr
        presence[group] = present_mask

    metadata = SampleMetadata(
        video_id=video_id,
        gloss=gloss,
        split="train",
        source="synthetic_realistic",
        fps=fps,
        num_frames=num_frames,
        translation=None,
    )

    landmark_presence = LandmarkPresence(
        pose=presence["pose"],
        left_hand=presence["left_hand"],
        right_hand=presence["right_hand"],
        face=presence["face"],
    )

    return ProcessedSample(
        metadata=metadata,
        pose=landmarks["pose"],
        left_hand=landmarks["left_hand"],
        right_hand=landmarks["right_hand"],
        face=landmarks["face"],
        presence=landmark_presence,
    )


def round_trip_sample(
    sample: ProcessedSample,
    tmp_path: Union[str, Path],
    *,
    filename: str = "synthetic_sample.npz",
) -> ProcessedSample:
    """
    Write `sample` to disk via write_sample and immediately read it back
    via load_sample. Useful for tests/scripts (e.g. the latency timer)
    that need a real .npz file on disk rather than an in-memory sample.

    Parameters
    ----------
    sample:
        The ProcessedSample to round-trip.
    tmp_path:
        Directory to write the .npz file into (e.g. pytest's tmp_path
        fixture, passed in by the caller -- this module does not depend
        on pytest itself).
    filename:
        Name of the .npz file to write within tmp_path.

    Returns
    -------
    The ProcessedSample as read back from disk.
    """
    output_path = Path(tmp_path) / filename
    write_sample(sample, output_path)
    return load_sample(output_path)