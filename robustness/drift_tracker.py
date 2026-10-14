"""
drift_tracker.py

Month 3 (built ahead of schedule): a drift-aware alternative to
normalisation.py's stateless, single-reference-per-frame normalization.

normalisation.py re-derives head_center/head_size from scratch on every
single frame, purely from that frame's own face landmarks -- there is no
memory across frames. That's simple and correct, but it means a frame
where the face is absent goes fully NaN (across ALL FOUR groups, per
that module's docstring), even on a frame where pose/hands are still
present and could, in principle, still be normalized using a recently-
seen head reference.

This module adds an EWMA-based DriftTracker that maintains a running
estimate of head_center/head_size ACROSS a sequence, updated only on
frames where the face is actually present (never updated toward a
zero-filled/absent-frame value), and HELD unchanged on frames where the
face is absent -- the same "hold, don't fabricate" discipline
hold_last_value.py already uses for landmark values themselves, applied
here to the normalization REFERENCE instead.

Deliberately reuses normalisation.py's own compute_head_center,
compute_head_size, _normalize_spatial_only, and _normalize_pose rather
than reimplementing per-frame head geometry or the centering/scaling
math -- the only new logic here is the EWMA state-tracking across time;
the per-frame math and the pose-visibility-channel handling stay
identical to Month 1's contract, on purpose.

Does not assume anything about ProcessedSample/LandmarkPresence beyond
what normalisation.py already assumes -- no new schema.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from robustness.normalisation import (
    _DEFAULT_MIN_HEAD_SIZE,
    _SPATIAL_ONLY_GROUPS,
    _normalize_pose,
    _normalize_spatial_only,
    compute_head_center,
    compute_head_size,
)
from src.preprocessing.sample_schema import LandmarkPresence, ProcessedSample

DEFAULT_ALPHA = 0.1


class DriftTracker:
    """
    EWMA tracker over head_center (x, y) and head_size (scalar), fed one
    frame's raw, possibly-undefined observation at a time via update().

    alpha is the standard EWMA smoothing factor: estimate_t = alpha *
    observation_t + (1 - alpha) * estimate_{t-1}. Larger alpha tracks
    recent frames more closely (less smoothing, more responsive to real
    head movement); smaller alpha smooths more aggressively (more
    resistant to single-frame jitter, slower to react to genuine drift).
    """

    def __init__(self, alpha: float = DEFAULT_ALPHA):
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.alpha = alpha
        self._center: Optional[np.ndarray] = None
        self._size: Optional[float] = None

    @property
    def initialized(self) -> bool:
        """True once at least one valid observation has been seen."""
        return self._center is not None

    def update(
        self, head_center_frame: np.ndarray, head_size_frame: float
    ) -> tuple[Optional[np.ndarray], Optional[float]]:
        """
        Feed one frame's raw (x, y) head_center and scalar head_size
        observation -- as produced by normalisation.compute_head_center /
        compute_head_size, which are NaN for a frame where the face is
        absent or the computed size is degenerate.

        If the observation is valid (neither NaN): initializes the EWMA
        on the first valid observation seen, or updates it thereafter.
        If invalid: the estimate is left completely unchanged -- this is
        the "hold, don't fabricate" rule; an absent-face frame must never
        pull the tracked reference toward a zero-filled or undefined
        value.

        Returns (center, size): the tracker's CURRENT estimate after this
        call (a *copy* of the internal center array, so callers can't
        accidentally mutate tracker state through the return value).
        Both are None if no valid observation has ever been seen -- there
        is nothing to hold from yet, the same state hold_last_value.py
        represents by leaving a frame zero-filled.
        """
        is_valid = not (np.isnan(head_center_frame).any() or np.isnan(head_size_frame))

        if is_valid:
            if not self.initialized:
                self._center = np.array(head_center_frame, dtype=np.float64)
                self._size = float(head_size_frame)
            else:
                self._center = self.alpha * np.asarray(head_center_frame, dtype=np.float64) + (
                    1.0 - self.alpha
                ) * self._center
                self._size = self.alpha * float(head_size_frame) + (1.0 - self.alpha) * self._size

        center_out = None if self._center is None else self._center.copy()
        return center_out, self._size


def track_sequence(
    face: np.ndarray,
    presence_face: np.ndarray,
    alpha: float = DEFAULT_ALPHA,
    min_head_size: float = _DEFAULT_MIN_HEAD_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Runs a fresh DriftTracker sequentially across one sample's face group.

    face: (T, 478, 3). presence_face: (T,) bool.

    Returns (tracked_center, tracked_size):
      tracked_center: (T, 2) float64 -- NaN for every frame before the
          tracker's first valid observation; the EWMA estimate at and
          after that point, including frames where THIS frame's face is
          absent (the tracker holds through those).
      tracked_size: (T,) float64 -- same convention.
    """
    raw_center = compute_head_center(face, presence_face)  # (T, 2), NaN where undefined
    raw_size = compute_head_size(face, presence_face, min_head_size=min_head_size)  # (T,)

    T = face.shape[0]
    tracked_center = np.full((T, 2), np.nan, dtype=np.float64)
    tracked_size = np.full(T, np.nan, dtype=np.float64)

    tracker = DriftTracker(alpha=alpha)
    for t in range(T):
        center, size = tracker.update(raw_center[t], raw_size[t])
        if center is not None:
            tracked_center[t] = center
            tracked_size[t] = size

    return tracked_center, tracked_size


def normalize_with_drift_tracking(
    sample: ProcessedSample,
    *,
    alpha: float = DEFAULT_ALPHA,
    min_head_size: float = _DEFAULT_MIN_HEAD_SIZE,
) -> ProcessedSample:
    """
    Drop-in alternative to normalisation.normalize_landmarks_sample: same
    output contract (a ProcessedSample, same shapes/dtypes, NaN marks
    "no reference available for this frame"), but the head_center/
    head_size reference is an EWMA tracked ACROSS the sequence (see
    DriftTracker/track_sequence) instead of re-derived independently on
    every frame.

    Behavioral difference from the stateless version, stated explicitly
    (this is the actual point of building this module): once the tracker
    has seen at least one valid face frame, EVERY subsequent frame gets a
    usable normalization reference -- including a frame where THIS
    frame's face is absent but pose/hands are still present -- because
    the EWMA holds its last estimate through occlusion instead of going
    undefined. normalize_landmarks_sample marks every face-absent frame
    NaN across all four groups regardless of whether other groups are
    present that frame. Frames before the tracker's first valid
    observation are still NaN under BOTH versions -- there is no
    reference of any kind to fall back on there.

    Does not mutate the input sample.
    """
    tracked_center, tracked_size = track_sequence(
        sample.face, sample.presence.face, alpha=alpha, min_head_size=min_head_size
    )

    valid_frames = ~np.isnan(tracked_size)

    normalized = {
        group: _normalize_spatial_only(
            getattr(sample, group), valid_frames, tracked_center, tracked_size
        )
        for group in _SPATIAL_ONLY_GROUPS
    }
    normalized["pose"] = _normalize_pose(sample.pose, valid_frames, tracked_center, tracked_size)

    return ProcessedSample(
        metadata=sample.metadata,
        pose=normalized["pose"],
        left_hand=normalized["left_hand"],
        right_hand=normalized["right_hand"],
        face=normalized["face"],
        presence=LandmarkPresence(
            pose=sample.presence.pose,
            left_hand=sample.presence.left_hand,
            right_hand=sample.presence.right_hand,
            face=sample.presence.face,
        ),
    )
