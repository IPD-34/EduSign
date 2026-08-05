"""
hold_last_value.py

Configurable hold-last-value interpolation for the landmark arrays inside
a ProcessedSample (pose, face, left_hand, right_hand).

Missingness is NOT detected via NaN-scanning. Presence is explicit and is
tracked per FRAME per GROUP (never per individual landmark within a
group) via the boolean (T,) arrays in LandmarkPresence: a frame is either
present or absent for an entire group at once.

Fill rule: for each landmark independently, walk forward through time.
When a missing frame is found (per the group's presence array), fill it
with the most recent valid value for that landmark, as long as no more
than `max_hold_frames` consecutive frames have been held since that valid
value. Once the hold budget is exceeded, the landmark is left as-is
(whatever zero-filled value the frame already had) until a new valid
value appears. Landmarks that have never had a valid value yet are left
unfilled (nothing to hold from). Frames that are already valid are left
completely unchanged.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from src.preprocessing.sample_schema import LandmarkPresence, ProcessedSample

_GROUPS = ("pose", "left_hand", "right_hand", "face")


def _hold_last_value_array(
    landmarks: np.ndarray,
    presence: np.ndarray,
    max_hold_frames: Optional[int] = None,
) -> np.ndarray:
    """
    Fill missing frames of one landmark group by holding the last valid
    value.

    Parameters
    ----------
    landmarks:
        (T, N, C) array for a single landmark group (C is 4 for pose,
        3 for face/left_hand/right_hand). Not mutated.
    presence:
        (T,) bool array. presence[t] is True if this group is present
        (valid) at frame t, False if it's missing at frame t. This is
        per-frame per-group -- not per-landmark; when a frame is
        missing, it's missing for every landmark in the group.
    max_hold_frames:
        Maximum number of consecutive frames a value may be held for.
        `None` (default) means hold indefinitely until a new valid value
        appears. `0` means never hold (equivalent to returning a copy).

    Returns
    -------
    (T, N, C) array, same shape/dtype as input, with eligible missing
    frames filled.
    """
    if landmarks.ndim != 3:
        raise ValueError(f"Expected shape (T, N, C), got {landmarks.shape}")
    T, N, C = landmarks.shape
    if presence.ndim != 1 or presence.shape[0] != T:
        raise ValueError(
            f"Expected presence shape ({T},), got {presence.shape}"
        )
    if max_hold_frames is not None and max_hold_frames < 0:
        raise ValueError("max_hold_frames must be >= 0 or None")

    out = landmarks.copy()

    last_valid = np.zeros((N, C), dtype=landmarks.dtype)
    has_valid = False
    hold_count = 0

    for t in range(T):
        if presence[t]:
            # Frame is valid: update the "last seen" cache and reset the
            # hold counter. Already-valid frames are left untouched.
            last_valid = landmarks[t]
            has_valid = True
            hold_count = 0
            continue

        # Frame is missing for this whole group.
        if not has_valid:
            # Nothing to hold from yet; leave as-is.
            continue

        hold_count += 1
        if max_hold_frames is None or hold_count <= max_hold_frames:
            out[t] = last_valid
        # else: hold budget exceeded, leave frame as-is (zero-filled).

    return out


def hold_last_value_sample(
    sample: ProcessedSample,
    max_hold_frames: Optional[int] = None,
) -> ProcessedSample:
    """
    Apply hold-last-value filling to every landmark group (pose, face,
    left_hand, right_hand) of a ProcessedSample, each using that group's
    own presence.<group> array.

    Does not mutate the input sample.

    Presence is left UNCHANGED in the returned sample: hold-last-value
    only interpolates a value for a frame that was originally missing,
    it does not make that frame "truly" present. Callers that need to
    know which frames were originally missing should keep consulting
    the (unchanged) presence arrays, not infer it from the landmark
    values.

    Parameters
    ----------
    sample:
        The ProcessedSample to fill. Not mutated.
    max_hold_frames:
        Maximum number of consecutive frames a value may be held for,
        applied independently to each of the four groups. `None`
        (default) means hold indefinitely; `0` means never hold.

    Returns
    -------
    A new ProcessedSample with the same metadata and presence as the
    input, but with each landmark group passed through
    _hold_last_value_array using its own presence array.
    """
    filled = {
        group: _hold_last_value_array(
            getattr(sample, group),
            getattr(sample.presence, group),
            max_hold_frames=max_hold_frames,
        )
        for group in _GROUPS
    }

    return ProcessedSample(
        metadata=sample.metadata,
        pose=filled["pose"],
        left_hand=filled["left_hand"],
        right_hand=filled["right_hand"],
        face=filled["face"],
        presence=LandmarkPresence(
            pose=sample.presence.pose,
            left_hand=sample.presence.left_hand,
            right_hand=sample.presence.right_hand,
            face=sample.presence.face,
        ),
    )