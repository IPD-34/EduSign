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

--- Fill-validity mask (added after a real bug: normalize_landmarks_sample
was NaN-ing every hold-filled frame anyway) ---
hold_last_value_sample() deliberately leaves `presence` UNCHANGED in its
returned sample (see that function's docstring) -- a frame this module
filled is still not "truly present" as far as the original sensor data
is concerned, and callers that care about that distinction (e.g. dropout
characterization scripts) must keep working from the unchanged presence
arrays. But that leaves no way for a caller who DOES want to know "does
this frame have a usable value after filling" (e.g. normalisation.py's
head_center/head_size reference, which is happy to use a held/carried-
forward face value) to ask that question -- compute_fill_validity() and
compute_fill_validity_sample() below answer exactly that, as a SEPARATE
concept from `presence`, not a replacement for it.

Deliberately implemented as standalone functions of (presence,
max_hold_frames) alone, NOT as an added return value on
hold_last_value_sample() / _hold_last_value_array(): the fill decision
(is frame t held, and within budget) never actually depends on the
landmark values themselves, only on the presence array and the hold
bookkeeping -- so it can be computed independently, without changing
either function's existing return signature and without touching the
~15 existing call sites (including hold_last_value_sample's own
presence-unchanged contract and its dedicated test) that depend on that
signature staying a single ProcessedSample.
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


def compute_fill_validity(
    presence: np.ndarray,
    max_hold_frames: Optional[int] = None,
) -> np.ndarray:
    """
    Returns a (T,) bool array: True at frame t if hold_last_value_sample
    would give that frame a legitimate value -- either originally present
    (presence[t] is True), or successfully held from an earlier valid
    frame within max_hold_frames. False only for frames with no valid
    observation to hold from yet (leading dropout before any valid frame
    appears at all) or where the hold budget was exceeded (frame is left
    at its original zero-filled value) -- the same two "still garbage"
    cases _hold_last_value_array itself leaves untouched.

    Mirrors _hold_last_value_array's own has_valid/hold_count bookkeeping
    exactly, but needs only `presence` (and max_hold_frames) -- never the
    actual landmark values -- since the fill DECISION at each frame
    depends only on those two things.
    """
    T = presence.shape[0]
    if max_hold_frames is not None and max_hold_frames < 0:
        raise ValueError("max_hold_frames must be >= 0 or None")

    validity = np.zeros(T, dtype=bool)
    has_valid = False
    hold_count = 0

    for t in range(T):
        if presence[t]:
            has_valid = True
            hold_count = 0
            validity[t] = True
            continue

        if not has_valid:
            continue  # nothing to hold from yet -- stays False

        hold_count += 1
        if max_hold_frames is None or hold_count <= max_hold_frames:
            validity[t] = True
        # else: hold budget exceeded -- stays False

    return validity


def compute_fill_validity_sample(
    sample: ProcessedSample,
    max_hold_frames: Optional[int] = None,
) -> LandmarkPresence:
    """
    compute_fill_validity() applied to all four of sample.presence's
    groups. Returns a LandmarkPresence-shaped result -- reusing that
    existing type because the shape (four (T,) bool arrays, one per
    group) is identical; this is NOT sample.presence itself and must
    never be assigned back into a ProcessedSample's `presence` field --
    it answers a different question ("usable after filling") than
    presence answers ("originally observed").
    """
    return LandmarkPresence(
        pose=compute_fill_validity(sample.presence.pose, max_hold_frames),
        left_hand=compute_fill_validity(sample.presence.left_hand, max_hold_frames),
        right_hand=compute_fill_validity(sample.presence.right_hand, max_hold_frames),
        face=compute_fill_validity(sample.presence.face, max_hold_frames),
    )


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