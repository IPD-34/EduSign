"""
test_normalisation.py

Tests for robustness.normalisation: compute_head_center, compute_head_size,
and the top-level normalize_landmarks_sample.

There is no metadata["segments"] dict in this contract -- normalisation
reads sample.face / sample.presence.face directly. Real face arrays are
(T, 478, 3); for the low-level compute_head_center/compute_head_size math
tests we use a smaller hand-built face-like array for clarity (the
functions themselves don't care about N, only about axis=1 being "points"
and axis=-1 being (x, y[, z])) -- but the full-sample tests use a real
478-point face array, as real callers will.
"""

from __future__ import annotations

import numpy as np
import pytest

from robustness.normalisation import (
    compute_head_center,
    compute_head_size,
    normalize_landmarks_sample,
)
from robustness.tests.fixtures.make_sample import make_synthetic_sample
from src.preprocessing.sample_schema import (
    LandmarkPresence,
    ProcessedSample,
    SampleMetadata,
)

_NUM_FACE_POINTS = 478


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _known_square_face(T: int = 1) -> np.ndarray:
    """
    Build a (T, 478, 3) face array where:
      - four "corner" points sit at (0,0), (2,0), (2,2), (0,2), z=0
      - every other point sits exactly at (1,1,0), the centroid of those
        corners

    This gives an exact, hand-verifiable geometry every frame:
      - head_center = mean over all points = (1, 1)  (since the "filler"
        points are placed exactly at the corners' centroid, adding them
        doesn't move the overall mean)
      - bounding box = [0, 2] x [0, 2], so head_size = sqrt(2^2 + 2^2)
        = sqrt(8), unaffected by the filler points because they all lie
        strictly inside that box
    """
    face = np.full((T, _NUM_FACE_POINTS, 3), 0.0, dtype=np.float64)
    face[:, :, :2] = 1.0  # filler points at (1, 1, 0)
    face[:, 0] = [0.0, 0.0, 0.0]
    face[:, 1] = [2.0, 0.0, 0.0]
    face[:, 2] = [2.0, 2.0, 0.0]
    face[:, 3] = [0.0, 2.0, 0.0]
    return face


_EXPECTED_HEAD_CENTER = np.array([1.0, 1.0])
_EXPECTED_HEAD_SIZE = np.sqrt(8.0)


def _make_known_sample(
    *,
    pose_point: tuple = (1.0, 1.0, 0.0),
    left_hand_point: tuple = (3.0, 3.0, 4.0),
    right_hand_point: tuple = (1.0, 1.0, 0.0),
    face_presence: bool = True,
    face: np.ndarray = None,
    pose_visibility: float = 0.42,
) -> ProcessedSample:
    """
    Build a single-frame (T=1) ProcessedSample with the known-square face
    geometry (unless a custom `face` array is passed), and every other
    group's single landmark set to a caller-chosen point so the
    centering/scaling math can be checked against hand-computed values.
    """
    T = 1
    if face is None:
        face = _known_square_face(T)

    pose = np.zeros((T, 33, 4), dtype=np.float64)
    pose[0, :, :3] = pose_point
    pose[0, :, 3] = pose_visibility  # arbitrary, known visibility value

    left_hand = np.zeros((T, 21, 3), dtype=np.float64)
    left_hand[0, :, :] = left_hand_point

    right_hand = np.zeros((T, 21, 3), dtype=np.float64)
    right_hand[0, :, :] = right_hand_point

    metadata = SampleMetadata(
        video_id="known_geometry_0001",
        gloss="TEST",
        split="train",
        source="synthetic",
        fps=30.0,
        num_frames=T,
        translation=None,
    )
    presence = LandmarkPresence(
        pose=np.array([True]),
        left_hand=np.array([True]),
        right_hand=np.array([True]),
        face=np.array([face_presence]),
    )
    return ProcessedSample(
        metadata=metadata,
        pose=pose,
        left_hand=left_hand,
        right_hand=right_hand,
        face=face,
        presence=presence,
    )


# ---------------------------------------------------------------------
# compute_head_center / compute_head_size
# ---------------------------------------------------------------------


def test_compute_head_center_known_geometry():
    face = _known_square_face(T=1)
    presence_face = np.array([True])

    center = compute_head_center(face, presence_face)

    np.testing.assert_allclose(center[0], _EXPECTED_HEAD_CENTER)


def test_compute_head_size_known_geometry():
    face = _known_square_face(T=1)
    presence_face = np.array([True])

    size = compute_head_size(face, presence_face)

    np.testing.assert_allclose(size[0], _EXPECTED_HEAD_SIZE)


def test_compute_head_center_nan_when_face_absent():
    face = np.zeros((2, 5, 3))
    presence_face = np.array([True, False])

    center = compute_head_center(face, presence_face)

    assert not np.isnan(center[0]).any()
    assert np.isnan(center[1]).all()


def test_compute_head_size_nan_when_face_absent():
    face = np.zeros((2, 5, 3))
    face[0] = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.5, 0.5, 0]]
    presence_face = np.array([True, False])

    size = compute_head_size(face, presence_face)

    assert not np.isnan(size[0])
    assert np.isnan(size[1])


def test_compute_head_size_nan_when_degenerate():
    # All points collapsed to a single location -> zero diagonal.
    face = np.full((1, 5, 3), 3.0)
    presence_face = np.array([True])

    size = compute_head_size(face, presence_face, min_head_size=1e-4)

    assert np.isnan(size[0])


# ---------------------------------------------------------------------
# normalize_landmarks_sample: shapes
# ---------------------------------------------------------------------


def test_normalize_preserves_shapes_for_all_groups():
    sample = _make_known_sample()

    out = normalize_landmarks_sample(sample)

    assert out.pose.shape == sample.pose.shape
    assert out.left_hand.shape == sample.left_hand.shape
    assert out.right_hand.shape == sample.right_hand.shape
    assert out.face.shape == sample.face.shape


# ---------------------------------------------------------------------
# Centering + scaling math
# ---------------------------------------------------------------------


def test_pose_landmark_at_head_center_maps_to_origin():
    sample = _make_known_sample(pose_point=(1.0, 1.0, 0.0))

    out = normalize_landmarks_sample(sample)

    np.testing.assert_allclose(out.pose[0, :, :2], 0.0, atol=1e-10)


def test_left_hand_offset_scales_by_head_size():
    # left_hand point sits at head_center + (2, 2), z = 4.
    sample = _make_known_sample(left_hand_point=(3.0, 3.0, 4.0))

    out = normalize_landmarks_sample(sample)

    expected_xy = 2.0 / _EXPECTED_HEAD_SIZE
    expected_z = 4.0 / _EXPECTED_HEAD_SIZE

    np.testing.assert_allclose(out.left_hand[0, :, 0], expected_xy)
    np.testing.assert_allclose(out.left_hand[0, :, 1], expected_xy)
    np.testing.assert_allclose(out.left_hand[0, :, 2], expected_z)


def test_right_hand_offset_scales_by_head_size():
    sample = _make_known_sample(right_hand_point=(1.0, 1.0, -4.0))

    out = normalize_landmarks_sample(sample)

    np.testing.assert_allclose(out.right_hand[0, :, :2], 0.0, atol=1e-10)
    np.testing.assert_allclose(out.right_hand[0, :, 2], -4.0 / _EXPECTED_HEAD_SIZE)


def test_face_corner_point_scales_correctly():
    sample = _make_known_sample()

    out = normalize_landmarks_sample(sample)

    # Corner point 1 was at (2, 0, 0) -> offset from center (1,1) is (1,-1).
    expected = np.array([1.0, -1.0]) / _EXPECTED_HEAD_SIZE
    np.testing.assert_allclose(out.face[0, 1, :2], expected)


# ---------------------------------------------------------------------
# Pose visibility channel: must never be touched
# ---------------------------------------------------------------------


def test_pose_visibility_channel_untouched_on_valid_frame():
    sample = _make_known_sample(pose_visibility=0.42)

    out = normalize_landmarks_sample(sample)

    np.testing.assert_array_equal(out.pose[0, :, 3], sample.pose[0, :, 3])
    assert (out.pose[0, :, 3] == 0.42).all()


def test_pose_visibility_channel_untouched_when_face_absent():
    sample = _make_known_sample(pose_visibility=0.77, face_presence=False)

    out = normalize_landmarks_sample(sample)

    # Spatial channels are undefined (NaN) for this frame...
    assert np.isnan(out.pose[0, :, :3]).all()
    # ...but visibility is preserved regardless.
    np.testing.assert_array_equal(out.pose[0, :, 3], sample.pose[0, :, 3])


# ---------------------------------------------------------------------
# Undefined-normalisation frames -> NaN across all four groups
# ---------------------------------------------------------------------


def test_face_absent_frame_produces_nan_across_all_groups():
    sample = _make_known_sample(face_presence=False)

    out = normalize_landmarks_sample(sample)

    assert np.isnan(out.pose[0, :, :3]).all()
    assert np.isnan(out.left_hand[0]).all()
    assert np.isnan(out.right_hand[0]).all()
    assert np.isnan(out.face[0]).all()


def test_degenerate_head_size_produces_nan_across_all_groups():
    # All face points collapsed to the same location -> zero head size.
    degenerate_face = np.full((1, _NUM_FACE_POINTS, 3), 3.0)
    sample = _make_known_sample(face=degenerate_face, face_presence=True)

    out = normalize_landmarks_sample(sample)

    assert np.isnan(out.pose[0, :, :3]).all()
    assert np.isnan(out.left_hand[0]).all()
    assert np.isnan(out.right_hand[0]).all()
    assert np.isnan(out.face[0]).all()


def test_valid_frame_never_produces_nan():
    sample = _make_known_sample()

    out = normalize_landmarks_sample(sample)

    assert not np.isnan(out.pose[0]).any()  # includes visibility channel
    assert not np.isnan(out.left_hand[0]).any()
    assert not np.isnan(out.right_hand[0]).any()
    assert not np.isnan(out.face[0]).any()


# ---------------------------------------------------------------------
# normalize_landmarks_sample does not mutate the input
# ---------------------------------------------------------------------


def test_does_not_mutate_input_sample():
    sample = _make_known_sample()
    original_pose = sample.pose.copy()
    original_face = sample.face.copy()

    normalize_landmarks_sample(sample)

    np.testing.assert_array_equal(sample.pose, original_pose)
    np.testing.assert_array_equal(sample.face, original_face)


def test_presence_unchanged_in_output():
    sample = _make_known_sample()

    out = normalize_landmarks_sample(sample)

    for group in ("pose", "left_hand", "right_hand", "face"):
        np.testing.assert_array_equal(
            getattr(out.presence, group), getattr(sample.presence, group)
        )


# ---------------------------------------------------------------------
# Broader checks against synthetic samples (no real data yet)
# ---------------------------------------------------------------------


def test_normalize_synthetic_sample_no_infinities():
    sample = make_synthetic_sample(num_frames=40, seed=0)

    out = normalize_landmarks_sample(sample)

    for group in ("pose", "left_hand", "right_hand", "face"):
        arr = getattr(out, group)
        # Every value must be either NaN (undefined-frame marker) or
        # finite -- never +/-inf, which would indicate an unguarded
        # division by a near-zero head size.
        assert np.all(np.isnan(arr) | np.isfinite(arr))


def test_normalize_synthetic_sample_preserves_shapes():
    sample = make_synthetic_sample(num_frames=40, seed=1)

    out = normalize_landmarks_sample(sample)

    assert out.pose.shape == sample.pose.shape
    assert out.left_hand.shape == sample.left_hand.shape
    assert out.right_hand.shape == sample.right_hand.shape
    assert out.face.shape == sample.face.shape