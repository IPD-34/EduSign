"""
normalisation.py

Face/head-size-based normalisation for a ProcessedSample's landmark
groups (pose, face, left_hand, right_hand).

For each frame where the face group is present:
    1. head_center = mean (x, y) over ALL face points that frame
       (presence is whole-frame, not per-point, so no per-point masking
       is needed within a present frame)
    2. head_size    = bounding-box diagonal (x, y) over those same points
    3. normalized   = (landmarks - head_center) / head_size for the
       spatial (x, y) channels; z is scaled by head_size but not
       re-centered, since MediaPipe's z is already roughly zero-centered
       on the hips/torso.

If a frame's face group is absent (presence.face is False), or the
computed head size is degenerate (zero / near-zero), normalisation is
undefined for that frame and the whole frame -- across ALL FOUR groups,
not just face -- is written out as NaN rather than risking a division
blow-up or silently-wrong scaling.

pose's 4th channel (visibility) is never touched by any of this: it is
copied through unchanged, never centered and never divided by head size.

This module does not attempt to guess a substitute for missing frames
(e.g. holding a previous head size) -- that is a hold-last-value
concern, not a normalisation concern. Callers that want that behavior
should run hold_last_value_sample() first.
"""

from __future__ import annotations

import numpy as np

from src.preprocessing.sample_schema import LandmarkPresence, ProcessedSample

_DEFAULT_MIN_HEAD_SIZE = 1e-4

# Groups with only spatial (x, y, z) channels -- everything except pose,
# whose 4th channel (visibility) needs separate handling.
_SPATIAL_ONLY_GROUPS = ("left_hand", "right_hand", "face")


def compute_head_center(face: np.ndarray, presence_face: np.ndarray) -> np.ndarray:
    """
    face: (T, 478, 3) array.
    presence_face: (T,) bool array; True where the face group is present
        (valid) that frame.

    Returns (T, 2) array of per-frame (x, y) centers, computed as the
    mean over ALL face points in frames where presence_face is True (no
    per-point masking is needed within a present frame -- presence is
    whole-frame). NaN for frames where presence_face is False.
    """
    T = face.shape[0]
    center = np.full((T, 2), np.nan, dtype=face.dtype)
    if not presence_face.any():
        return center

    mean_xy = face[:, :, :2].mean(axis=1)  # (T, 2), computed for every frame
    center[presence_face] = mean_xy[presence_face]
    return center


def compute_head_size(
    face: np.ndarray,
    presence_face: np.ndarray,
    min_head_size: float = _DEFAULT_MIN_HEAD_SIZE,
) -> np.ndarray:
    """
    face: (T, 478, 3) array.
    presence_face: (T,) bool array; True where the face group is present
        (valid) that frame.
    min_head_size:
        Head sizes below this are treated as degenerate.

    Returns (T,) array of per-frame bounding-box diagonals over (x, y),
    computed over ALL face points in present frames. NaN for frames
    where presence_face is False, or where the computed diagonal is
    below `min_head_size`.
    """
    T = face.shape[0]
    diag = np.full(T, np.nan, dtype=face.dtype)
    if not presence_face.any():
        return diag

    xy = face[:, :, :2]
    mins = xy.min(axis=1)  # (T, 2)
    maxs = xy.max(axis=1)  # (T, 2)
    span = maxs - mins  # (T, 2)
    computed = np.sqrt(np.sum(span**2, axis=1))  # (T,)

    diag[presence_face] = computed[presence_face]
    diag[diag < min_head_size] = np.nan
    return diag


def _normalize_spatial_only(
    landmarks: np.ndarray,
    valid_frames: np.ndarray,
    head_center: np.ndarray,
    head_size: np.ndarray,
) -> np.ndarray:
    """
    Normalize a 3-channel (x, y, z) group: center (x, y) on head_center
    and scale all three channels by head_size. Frames outside
    valid_frames are output as NaN.
    """
    out = np.full_like(landmarks, np.nan)
    if not valid_frames.any():
        return out

    hs = head_size[valid_frames].reshape(-1, 1, 1)  # (T_valid, 1, 1)
    center_xy = head_center[valid_frames].reshape(-1, 1, 2)  # (T_valid, 1, 2)

    frame_data = landmarks[valid_frames]  # (T_valid, N, 3)
    normed = frame_data.copy()
    normed[:, :, :2] = (frame_data[:, :, :2] - center_xy) / hs
    normed[:, :, 2] = frame_data[:, :, 2] / hs[:, :, 0]

    out[valid_frames] = normed
    return out


def _normalize_pose(
    pose: np.ndarray,
    valid_frames: np.ndarray,
    head_center: np.ndarray,
    head_size: np.ndarray,
) -> np.ndarray:
    """
    Normalize pose's spatial channels (x, y, z) exactly like the other
    groups, but copy the 4th (visibility) channel through UNCHANGED --
    it must never be centered or divided by head_size. Frames outside
    valid_frames still get NaN for the spatial channels, but visibility
    is preserved as-is even for those frames (there is nothing to
    normalize about visibility, so "undefined normalisation" does not
    apply to it).
    """
    out = np.full_like(pose, np.nan)
    out[:, :, 3] = pose[:, :, 3]  # visibility passes through unchanged, always

    if not valid_frames.any():
        return out

    hs = head_size[valid_frames].reshape(-1, 1, 1)
    center_xy = head_center[valid_frames].reshape(-1, 1, 2)

    frame_data = pose[valid_frames, :, :3]  # (T_valid, N, 3) spatial only
    normed = frame_data.copy()
    normed[:, :, :2] = (frame_data[:, :, :2] - center_xy) / hs
    normed[:, :, 2] = frame_data[:, :, 2] / hs[:, :, 0]

    out[valid_frames, :, :3] = normed
    return out


def normalize_landmarks_sample(
    sample: ProcessedSample,
    *,
    min_head_size: float = _DEFAULT_MIN_HEAD_SIZE,
) -> ProcessedSample:
    """
    Normalise all four landmark groups (pose, face, left_hand,
    right_hand) of a ProcessedSample by per-frame head center and head
    size, computed from the face group.

    Does not mutate the input sample. Does not fill any missing frames
    -- run hold_last_value_sample() first if that behavior is wanted.

    Parameters
    ----------
    sample:
        The ProcessedSample to normalize.
    min_head_size:
        Head sizes below this are treated as degenerate; frames with a
        degenerate or undefined head size are output as NaN across ALL
        FOUR groups (except pose's visibility channel, which is always
        preserved).

    Returns
    -------
    A new ProcessedSample with the same metadata and presence as the
    input. Frames where head size was valid have normalized (x, y, z)
    values (pose's visibility channel untouched); frames where head
    size was NaN/degenerate have NaN spatial values (visibility still
    preserved for pose).
    """
    head_center = compute_head_center(sample.face, sample.presence.face)  # (T, 2)
    head_size = compute_head_size(
        sample.face, sample.presence.face, min_head_size=min_head_size
    )  # (T,)

    valid_frames = ~np.isnan(head_size)

    normalized = {
        group: _normalize_spatial_only(
            getattr(sample, group), valid_frames, head_center, head_size
        )
        for group in _SPATIAL_ONLY_GROUPS
    }
    normalized["pose"] = _normalize_pose(
        sample.pose, valid_frames, head_center, head_size
    )

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