"""
test_hold_last_value.py

Tests for robustness.hold_last_value: both the low-level
_hold_last_value_array (operates on a single group's (T, N, C) array +
(T,) presence array) and the top-level hold_last_value_sample (operates
on a full ProcessedSample, filling all four groups).

Missingness in this contract is never NaN-based: it's an explicit
per-frame presence=False, and absent frames are zero-filled in the
input (not NaN-filled). There is no real preprocessed data yet this
month, so tests either build small hand-crafted samples/arrays for
precise edge-case control, or use make_synthetic_sample for broader
shape/dtype/behavioral checks.
"""

from __future__ import annotations

import numpy as np
import pytest

from robustness.hold_last_value import (
    _hold_last_value_array,
    compute_fill_validity,
    compute_fill_validity_sample,
    hold_last_value_sample,
)
from robustness.tests.fixtures.make_sample import make_synthetic_sample
from src.preprocessing.sample_schema import (
    LandmarkPresence,
    ProcessedSample,
    SampleMetadata,
)

_GROUPS = ("pose", "left_hand", "right_hand", "face")


# ---------------------------------------------------------------------
# Low-level _hold_last_value_array tests
# ---------------------------------------------------------------------


def test_fills_from_last_valid():
    # T=5, N=1, C=3. Valid at frames 0 and 3; frames 1, 2, 4 missing.
    landmarks = np.zeros((5, 1, 3), dtype=np.float32)
    landmarks[0] = [1.0, 1.0, 1.0]
    landmarks[3] = [2.0, 2.0, 2.0]
    presence = np.array([True, False, False, True, False])

    out = _hold_last_value_array(landmarks, presence, max_hold_frames=None)

    np.testing.assert_array_equal(out[0, 0], [1.0, 1.0, 1.0])
    np.testing.assert_array_equal(out[1, 0], [1.0, 1.0, 1.0])  # held from frame 0
    np.testing.assert_array_equal(out[2, 0], [1.0, 1.0, 1.0])  # held from frame 0
    np.testing.assert_array_equal(out[3, 0], [2.0, 2.0, 2.0])
    np.testing.assert_array_equal(out[4, 0], [2.0, 2.0, 2.0])  # held from frame 3


def test_stays_unfilled_when_no_valid_seen_yet():
    # Frames 0, 1 missing before any valid value has appeared; frame 2 valid.
    landmarks = np.zeros((3, 1, 3), dtype=np.float32)
    landmarks[2] = [5.0, 5.0, 5.0]
    presence = np.array([False, False, True])

    out = _hold_last_value_array(landmarks, presence, max_hold_frames=None)

    # Nothing to hold from yet -- frames 0 and 1 stay as their original
    # (zero-filled) values, unchanged.
    np.testing.assert_array_equal(out[0, 0], [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(out[1, 0], [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(out[2, 0], [5.0, 5.0, 5.0])


def test_max_hold_frames_budget_respected():
    # Valid at frame 0, then three consecutive missing frames.
    landmarks = np.zeros((4, 1, 3), dtype=np.float32)
    landmarks[0] = [1.0, 1.0, 1.0]
    presence = np.array([True, False, False, False])

    out = _hold_last_value_array(landmarks, presence, max_hold_frames=1)

    np.testing.assert_array_equal(out[0, 0], [1.0, 1.0, 1.0])
    np.testing.assert_array_equal(out[1, 0], [1.0, 1.0, 1.0])  # hold_count 1 <= 1
    np.testing.assert_array_equal(out[2, 0], [0.0, 0.0, 0.0])  # hold_count 2 > 1
    np.testing.assert_array_equal(out[3, 0], [0.0, 0.0, 0.0])  # hold_count 3 > 1


def test_max_hold_frames_zero_never_holds():
    landmarks = np.zeros((2, 1, 3), dtype=np.float32)
    landmarks[0] = [1.0, 1.0, 1.0]
    presence = np.array([True, False])

    out = _hold_last_value_array(landmarks, presence, max_hold_frames=0)

    np.testing.assert_array_equal(out[0, 0], [1.0, 1.0, 1.0])
    np.testing.assert_array_equal(out[1, 0], [0.0, 0.0, 0.0])  # never held


def test_hold_resets_after_new_valid_value():
    # valid, missing, valid, missing -- with max_hold_frames=1, both
    # missing frames should get filled because the hold counter resets
    # each time a new valid value is seen.
    landmarks = np.zeros((4, 1, 3), dtype=np.float32)
    landmarks[0] = [1.0, 1.0, 1.0]
    landmarks[2] = [2.0, 2.0, 2.0]
    presence = np.array([True, False, True, False])

    out = _hold_last_value_array(landmarks, presence, max_hold_frames=1)

    np.testing.assert_array_equal(out[0, 0], [1.0, 1.0, 1.0])
    np.testing.assert_array_equal(out[1, 0], [1.0, 1.0, 1.0])  # held from frame 0
    np.testing.assert_array_equal(out[2, 0], [2.0, 2.0, 2.0])
    np.testing.assert_array_equal(out[3, 0], [2.0, 2.0, 2.0])  # held from frame 2, fresh budget


def test_negative_max_hold_frames_raises():
    landmarks = np.zeros((2, 1, 3), dtype=np.float32)
    presence = np.array([True, False])

    with pytest.raises(ValueError):
        _hold_last_value_array(landmarks, presence, max_hold_frames=-1)


def test_bad_landmarks_shape_raises():
    # ndim != 3
    landmarks = np.zeros((2, 3), dtype=np.float32)
    presence = np.array([True, False])

    with pytest.raises(ValueError):
        _hold_last_value_array(landmarks, presence)


def test_presence_shape_mismatch_raises():
    landmarks = np.zeros((3, 1, 3), dtype=np.float32)
    presence = np.array([True, False])  # wrong length (2 != 3)

    with pytest.raises(ValueError):
        _hold_last_value_array(landmarks, presence)


def test_does_not_mutate_input_array():
    landmarks = np.zeros((3, 1, 3), dtype=np.float32)
    landmarks[0] = [1.0, 1.0, 1.0]
    presence = np.array([True, False, False])
    original = landmarks.copy()

    _hold_last_value_array(landmarks, presence, max_hold_frames=None)

    np.testing.assert_array_equal(landmarks, original)


# ---------------------------------------------------------------------
# Top-level hold_last_value_sample tests
# ---------------------------------------------------------------------


def _make_hand_built_sample() -> ProcessedSample:
    """
    A small (T=4) hand-built ProcessedSample with a nontrivial, distinct
    dropout pattern per group, for precise control in top-level tests.
    """
    T = 4
    pose = np.zeros((T, 33, 4), dtype=np.float32)
    left_hand = np.zeros((T, 21, 3), dtype=np.float32)
    right_hand = np.zeros((T, 21, 3), dtype=np.float32)
    face = np.zeros((T, 478, 3), dtype=np.float32)

    # pose: valid at frame 0, missing 1-3.
    pose[0] = 1.0
    pose_presence = np.array([True, False, False, False])

    # left_hand: valid at frames 0 and 2.
    left_hand[0] = 2.0
    left_hand[2] = 3.0
    left_hand_presence = np.array([True, False, True, False])

    # right_hand: always present (no dropout).
    right_hand[:] = 4.0
    right_hand_presence = np.array([True, True, True, True])

    # face: never present.
    face_presence = np.array([False, False, False, False])

    metadata = SampleMetadata(
        video_id="handbuilt_0001",
        gloss="TEST",
        split="train",
        source="synthetic",
        fps=30.0,
        num_frames=T,
        translation=None,
    )
    presence = LandmarkPresence(
        pose=pose_presence,
        left_hand=left_hand_presence,
        right_hand=right_hand_presence,
        face=face_presence,
    )
    return ProcessedSample(
        metadata=metadata,
        pose=pose,
        left_hand=left_hand,
        right_hand=right_hand,
        face=face,
        presence=presence,
    )


def test_hold_last_value_sample_fills_each_group_independently():
    sample = _make_hand_built_sample()

    out = hold_last_value_sample(sample, max_hold_frames=None)

    # pose: held from frame 0 for the rest.
    np.testing.assert_array_equal(out.pose[1, :, :3], np.ones((33, 3)))
    np.testing.assert_array_equal(out.pose[3, :, :3], np.ones((33, 3)))
    # pose visibility channel (index 3) untouched by hold-last-value too
    # (it's just another channel here -- normalisation is what special-
    # cases it, hold-last-value fills it like any other channel).
    np.testing.assert_array_equal(out.pose[1, :, 3], np.ones(33))

    # left_hand: frame 1 held from frame 0, frame 3 held from frame 2.
    np.testing.assert_array_equal(out.left_hand[1], np.full((21, 3), 2.0))
    np.testing.assert_array_equal(out.left_hand[3], np.full((21, 3), 3.0))

    # right_hand: never missing, unchanged everywhere.
    np.testing.assert_array_equal(out.right_hand, sample.right_hand)

    # face: never present -- nothing to hold from, stays zero-filled.
    np.testing.assert_array_equal(out.face, np.zeros_like(sample.face))


def test_hold_last_value_sample_presence_unchanged():
    sample = _make_hand_built_sample()

    out = hold_last_value_sample(sample, max_hold_frames=None)

    for group in _GROUPS:
        original_presence = getattr(sample.presence, group)
        output_presence = getattr(out.presence, group)
        np.testing.assert_array_equal(output_presence, original_presence)


def test_hold_last_value_sample_does_not_mutate_input():
    sample = _make_hand_built_sample()
    original_pose = sample.pose.copy()
    original_left_hand = sample.left_hand.copy()

    hold_last_value_sample(sample, max_hold_frames=None)

    np.testing.assert_array_equal(sample.pose, original_pose)
    np.testing.assert_array_equal(sample.left_hand, original_left_hand)


def test_hold_last_value_sample_all_four_groups_synthetic():
    sample = make_synthetic_sample(
        num_frames=50,
        seed=0,
        pose_dropout=0.3,
        left_hand_dropout=0.3,
        right_hand_dropout=0.3,
        face_dropout=0.3,
    )

    out = hold_last_value_sample(sample, max_hold_frames=None)

    for group in _GROUPS:
        original = getattr(sample, group)
        filled = getattr(out, group)
        assert filled.shape == original.shape
        assert filled.dtype == original.dtype


def test_hold_last_value_sample_fill_actually_changes_values():
    # With a nontrivial dropout pattern and unlimited hold budget, at
    # least some previously-missing (zero-filled) frames should no
    # longer be all-zero after filling, as long as there's at least one
    # valid frame earlier in the sequence for them to hold from.
    sample = make_synthetic_sample(
        num_frames=50,
        seed=1,
        left_hand_dropout=[1, 2, 3, 4, 5],  # frame 0 stays present (no dropout there)
    )

    out = hold_last_value_sample(sample, max_hold_frames=None)

    missing_frames = ~sample.presence.left_hand
    assert missing_frames.any()

    original_missing = sample.left_hand[missing_frames]
    filled_missing = out.left_hand[missing_frames]

    # The originally-missing frames were zero-filled; after holding from
    # frame 0 (which is present and, with overwhelming probability for
    # random float data, not all-zero), they should no longer match the
    # original all-zero values.
    assert not np.array_equal(original_missing, filled_missing)


# ---------------------------------------------------------------------
# compute_fill_validity / compute_fill_validity_sample
#
# Added alongside the fix for a real bug: normalize_landmarks_sample used
# to decide frame validity from `presence` alone, which hold_last_value_sample
# deliberately never changes -- so a hold-filled frame was still marked NaN
# by normalisation despite holding a real, carried-forward value. These
# functions answer "does frame t have a usable value after filling",
# entirely separate from `presence` (which still means "was this frame
# actually observed").
# ---------------------------------------------------------------------


def test_compute_fill_validity_matches_hold_last_value_array_decisions():
    """
    compute_fill_validity's has_valid/hold_count bookkeeping must exactly
    match _hold_last_value_array's own fill decisions -- checked directly
    against test_max_hold_frames_budget_respected's exact scenario:
    valid at frame 0, then three consecutive missing frames, budget=1.
    """
    presence = np.array([True, False, False, False])

    validity = compute_fill_validity(presence, max_hold_frames=1)

    # Frame 0: present -> valid. Frame 1: held (hold_count=1<=1) -> valid.
    # Frames 2, 3: hold budget exceeded -> NOT valid (still garbage zero-fill).
    np.testing.assert_array_equal(validity, [True, True, False, False])


def test_compute_fill_validity_unlimited_hold():
    presence = np.array([True, False, False, True, False])

    validity = compute_fill_validity(presence, max_hold_frames=None)

    # Every frame has a valid value with unlimited hold: frames 1, 2 held
    # from frame 0; frame 4 held from frame 3.
    np.testing.assert_array_equal(validity, [True, True, True, True, True])


def test_compute_fill_validity_false_before_first_valid_observation():
    presence = np.array([False, False, True, False])

    validity = compute_fill_validity(presence, max_hold_frames=None)

    # Frames 0, 1: nothing to hold from yet -- NOT valid, matching
    # _hold_last_value_array leaving those frames as their original
    # zero-filled values, not fabricating one.
    np.testing.assert_array_equal(validity, [False, False, True, True])


def test_compute_fill_validity_max_hold_zero_matches_presence():
    presence = np.array([True, False, True, False])

    validity = compute_fill_validity(presence, max_hold_frames=0)

    # max_hold_frames=0 means "never hold" -- validity should exactly
    # equal presence itself (nothing is ever successfully filled).
    np.testing.assert_array_equal(validity, presence)


def test_compute_fill_validity_negative_max_hold_raises():
    with pytest.raises(ValueError):
        compute_fill_validity(np.array([True, False]), max_hold_frames=-1)


def test_compute_fill_validity_sample_applies_per_group_independently():
    sample = _make_hand_built_sample()

    validity = compute_fill_validity_sample(sample, max_hold_frames=None)

    # From _make_hand_built_sample: pose valid at 0 only -> held for the
    # rest (unlimited hold) -> all True. left_hand valid at 0, 2 -> all
    # True (held). right_hand always present -> all True. face never
    # present -> all False (nothing to ever hold from).
    np.testing.assert_array_equal(validity.pose, [True, True, True, True])
    np.testing.assert_array_equal(validity.left_hand, [True, True, True, True])
    np.testing.assert_array_equal(validity.right_hand, [True, True, True, True])
    np.testing.assert_array_equal(validity.face, [False, False, False, False])


def test_compute_fill_validity_does_not_mutate_presence():
    """compute_fill_validity_sample must return a NEW object, never the
    sample's own `presence` -- conflating the two would silently break the
    documented "presence is never mutated by hold-last-value" contract
    that hold_last_value_sample's own tests (above) depend on."""
    sample = _make_hand_built_sample()
    original_presence_face = sample.presence.face.copy()

    validity = compute_fill_validity_sample(sample, max_hold_frames=None)

    np.testing.assert_array_equal(sample.presence.face, original_presence_face)
    assert validity is not sample.presence