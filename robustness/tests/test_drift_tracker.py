"""
test_drift_tracker.py

Tests for robustness.drift_tracker: the EWMA math itself on hand-built
known values, the hold-through-occlusion behavior, the key behavioral
difference from the stateless normalize_landmarks_sample (a frame with
face absent but pose present), and the same no-NaN/no-discontinuity bar
Month 1's test_normalisation.py applies, run against the real 41 ASLLRP
samples.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from robustness.drift_tracker import (
    DEFAULT_ALPHA,
    DriftTracker,
    normalize_with_drift_tracking,
    track_sequence,
)
from robustness.normalisation import normalize_landmarks_sample
from robustness.tests.fixtures.make_sample import make_synthetic_sample
from src.preprocessing.sample_writer import load_sample

_GROUPS = ("pose", "left_hand", "right_hand", "face")
_NPZ_DIRS = ["data/processed/asllrp/train", "data/processed/asllrp/val"]


def _real_samples_available() -> bool:
    return any(Path(d).exists() and any(Path(d).glob("*.npz")) for d in _NPZ_DIRS)


# ---------------------------------------------------------------------
# DriftTracker.update: EWMA math on known values
# ---------------------------------------------------------------------


def test_first_valid_observation_initializes_exactly():
    tracker = DriftTracker(alpha=0.3)

    center, size = tracker.update(np.array([1.0, 2.0]), 5.0)

    np.testing.assert_array_equal(center, [1.0, 2.0])
    assert size == 5.0
    assert tracker.initialized


def test_second_valid_observation_applies_ewma_formula():
    tracker = DriftTracker(alpha=0.5)
    tracker.update(np.array([0.0, 0.0]), 10.0)

    center, size = tracker.update(np.array([2.0, 4.0]), 20.0)

    # estimate = alpha * obs + (1 - alpha) * previous
    np.testing.assert_allclose(center, [0.5 * 2.0 + 0.5 * 0.0, 0.5 * 4.0 + 0.5 * 0.0])
    assert size == pytest.approx(0.5 * 20.0 + 0.5 * 10.0)


def test_ewma_formula_with_asymmetric_alpha():
    tracker = DriftTracker(alpha=0.2)
    tracker.update(np.array([10.0, 10.0]), 100.0)

    center, size = tracker.update(np.array([0.0, 0.0]), 0.0)

    np.testing.assert_allclose(center, [0.2 * 0.0 + 0.8 * 10.0, 0.2 * 0.0 + 0.8 * 10.0])
    assert size == pytest.approx(0.2 * 0.0 + 0.8 * 100.0)


def test_invalid_observation_holds_previous_estimate_unchanged():
    tracker = DriftTracker(alpha=0.5)
    tracker.update(np.array([1.0, 2.0]), 5.0)

    center, size = tracker.update(np.array([np.nan, np.nan]), np.nan)

    np.testing.assert_array_equal(center, [1.0, 2.0])
    assert size == 5.0


def test_nothing_to_hold_before_first_valid_observation():
    tracker = DriftTracker(alpha=0.5)

    center, size = tracker.update(np.array([np.nan, np.nan]), np.nan)

    assert center is None
    assert size is None
    assert not tracker.initialized


def test_partial_nan_center_treated_as_invalid():
    # Only one of the two center coordinates is NaN -- still invalid as a
    # whole observation (can't half-update a 2D center).
    tracker = DriftTracker(alpha=0.5)
    tracker.update(np.array([1.0, 2.0]), 5.0)

    center, size = tracker.update(np.array([np.nan, 3.0]), 6.0)

    np.testing.assert_array_equal(center, [1.0, 2.0])
    assert size == 5.0


def test_update_return_value_is_a_copy_not_internal_state():
    tracker = DriftTracker(alpha=0.5)
    center, _size = tracker.update(np.array([1.0, 2.0]), 5.0)

    center[0] = 999.0  # mutate the returned array

    center2, _ = tracker.update(np.array([1.0, 2.0]), 5.0)
    assert center2[0] != 999.0  # internal state must be unaffected


def test_invalid_alpha_raises():
    with pytest.raises(ValueError):
        DriftTracker(alpha=0.0)
    with pytest.raises(ValueError):
        DriftTracker(alpha=1.5)
    with pytest.raises(ValueError):
        DriftTracker(alpha=-0.1)


# ---------------------------------------------------------------------
# track_sequence: hold-through-occlusion behavior over a real sequence
# ---------------------------------------------------------------------


def test_track_sequence_holds_through_missing_face_frames():
    T = 5
    face = np.zeros((T, 478, 3), dtype=np.float64)
    # Frames 0, 1: a valid square face at two different positions.
    # Frames 2, 3: face absent (zero-filled, presence False).
    # Frame 4: valid again, at a third position.
    def _square_at(cx, cy):
        f = np.full((478, 3), 0.0)
        f[:, :2] = [cx, cy]
        f[0] = [cx - 1, cy - 1, 0]
        f[1] = [cx + 1, cy - 1, 0]
        f[2] = [cx + 1, cy + 1, 0]
        f[3] = [cx - 1, cy + 1, 0]
        return f

    face[0] = _square_at(0.0, 0.0)
    face[1] = _square_at(2.0, 0.0)
    face[4] = _square_at(4.0, 0.0)
    presence_face = np.array([True, True, False, False, True])

    tracked_center, tracked_size = track_sequence(face, presence_face, alpha=0.5)

    # Frames 2, 3 must exactly equal frame 1's tracked estimate (held).
    np.testing.assert_array_equal(tracked_center[2], tracked_center[1])
    np.testing.assert_array_equal(tracked_center[3], tracked_center[1])
    assert tracked_size[2] == tracked_size[1]
    assert tracked_size[3] == tracked_size[1]

    # Frame 4 (a new valid observation) must differ from the held value.
    assert not np.array_equal(tracked_center[4], tracked_center[1])


def test_track_sequence_all_frames_absent_stays_nan():
    T = 4
    face = np.zeros((T, 478, 3), dtype=np.float64)
    presence_face = np.zeros(T, dtype=bool)

    tracked_center, tracked_size = track_sequence(face, presence_face)

    assert np.isnan(tracked_center).all()
    assert np.isnan(tracked_size).all()


# ---------------------------------------------------------------------
# The key behavioral difference: face absent, pose present
# ---------------------------------------------------------------------


def test_drift_tracking_normalizes_pose_when_face_absent_but_pose_present():
    """
    This is the actual point of building drift_tracker.py: a frame where
    the face is absent but pose is still present goes NaN under the
    stateless normalize_landmarks_sample, but gets a real normalized
    value under normalize_with_drift_tracking, once the tracker has seen
    at least one valid face frame earlier in the sequence.
    """
    sample = make_synthetic_sample(
        num_frames=10,
        seed=3,
        face_dropout=[5, 6, 7],  # face absent at frames 5-7 only
    )
    # pose is fully present throughout (no pose_dropout passed).

    stateless_out = normalize_landmarks_sample(sample)
    drift_out = normalize_with_drift_tracking(sample, alpha=0.3)

    for t in (5, 6, 7):
        assert np.isnan(stateless_out.pose[t, :, :3]).all(), (
            f"frame {t}: stateless normalization should be NaN when face is absent"
        )
        assert not np.isnan(drift_out.pose[t, :, :3]).any(), (
            f"frame {t}: drift-tracked normalization should still produce a "
            f"value, held from the last valid face observation"
        )


def test_drift_tracking_still_nan_before_first_valid_face_frame():
    sample = make_synthetic_sample(
        num_frames=6,
        seed=4,
        face_dropout=[0, 1, 2],  # face absent for the first 3 frames
    )

    drift_out = normalize_with_drift_tracking(sample, alpha=0.3)

    for t in (0, 1, 2):
        assert np.isnan(drift_out.pose[t, :, :3]).all(), (
            f"frame {t}: no valid reference exists yet, should still be NaN "
            f"under drift tracking too"
        )


# ---------------------------------------------------------------------
# Same bar as Month 1's test_normalisation.py: no NaN/Inf beyond the
# expected "no reference yet" marker, no shape drift, run against the
# real 41 ASLLRP samples.
# ---------------------------------------------------------------------


pytestmark_real = pytest.mark.skipif(
    not _real_samples_available(), reason="real ASLLRP samples not found under data/processed/asllrp/"
)


def _iter_real_samples():
    for npz_dir in _NPZ_DIRS:
        for npz_path in sorted(Path(npz_dir).glob("*.npz")):
            yield load_sample(npz_path)


@pytestmark_real
@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.3, 0.7])
def test_drift_tracking_no_inf_on_real_samples(alpha):
    for sample in _iter_real_samples():
        out = normalize_with_drift_tracking(sample, alpha=alpha)
        for group in _GROUPS:
            arr = getattr(out, group)
            assert np.all(np.isnan(arr) | np.isfinite(arr)), (
                f"{sample.metadata.video_id}/{group}: +/-inf found "
                f"(alpha={alpha})"
            )


@pytestmark_real
def test_drift_tracking_preserves_shapes_on_real_samples():
    for sample in _iter_real_samples():
        out = normalize_with_drift_tracking(sample)
        for group in _GROUPS:
            assert getattr(out, group).shape == getattr(sample, group).shape


@pytestmark_real
def test_drift_tracking_does_not_mutate_input_on_real_samples():
    for sample in _iter_real_samples():
        original_pose = sample.pose.copy()
        original_face = sample.face.copy()

        normalize_with_drift_tracking(sample)

        np.testing.assert_array_equal(sample.pose, original_pose)
        np.testing.assert_array_equal(sample.face, original_face)


@pytestmark_real
def test_drift_tracking_covers_at_least_as_many_frames_as_stateless_on_real_samples():
    """
    'No discontinuous jump' framed as a coverage monotonicity property
    here: drift tracking should never produce FEWER valid (non-NaN)
    frames than the stateless version -- it's designed to hold through
    gaps the stateless version can't, never to lose coverage.
    """
    for sample in _iter_real_samples():
        stateless_out = normalize_landmarks_sample(sample)
        drift_out = normalize_with_drift_tracking(sample, alpha=DEFAULT_ALPHA)

        stateless_valid = ~np.isnan(stateless_out.pose[:, 0, 0])
        drift_valid = ~np.isnan(drift_out.pose[:, 0, 0])

        assert drift_valid.sum() >= stateless_valid.sum(), (
            f"{sample.metadata.video_id}: drift tracking covered fewer valid "
            f"frames ({drift_valid.sum()}) than stateless "
            f"({stateless_valid.sum()})"
        )
