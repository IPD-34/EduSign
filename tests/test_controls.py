"""Control checks on made-up arrays and ids (no dataset content)."""
import copy
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from eval.controls import shuffle_mapping, zeroed_pose  # noqa: E402

IDS = [f"clip_{i:03d}" for i in range(20)]


def _pose(n_frames=7, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "keypoints": [rng.random((1, 133, 2)).astype(np.float32) + 1 for _ in range(n_frames)],
        "scores": [rng.random((1, 133)).astype(np.float32) + 1 for _ in range(n_frames)],
    }


def test_zeroed_is_zero_same_shape_and_input_untouched():
    pose = _pose()
    before = copy.deepcopy(pose)
    out = zeroed_pose(pose)
    assert len(out["keypoints"]) == len(pose["keypoints"])
    assert len(out["scores"]) == len(pose["scores"])
    for a, b in zip(out["keypoints"], pose["keypoints"]):
        assert a.shape == b.shape and a.dtype == b.dtype and not a.any()
    for a, b in zip(out["scores"], pose["scores"]):
        assert a.shape == b.shape and a.dtype == b.dtype and not a.any()
    for k in before:
        for a, b in zip(before[k], pose[k]):
            assert np.array_equal(a, b)


def test_shuffle_is_deterministic_for_fixed_seed():
    assert shuffle_mapping(IDS, 42) == shuffle_mapping(IDS, 42)
    assert shuffle_mapping(IDS, 42) != shuffle_mapping(IDS, 43)


def test_shuffle_is_a_permutation_with_no_fixed_points():
    # The original loop only stops when no clip keeps its own pose (len(ids) >= 2).
    m = shuffle_mapping(IDS, 42)
    assert sorted(m) == sorted(IDS)
    assert sorted(m.values()) == sorted(IDS)
    assert all(k != v for k, v in m.items())
