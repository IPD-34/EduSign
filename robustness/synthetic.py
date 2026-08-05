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