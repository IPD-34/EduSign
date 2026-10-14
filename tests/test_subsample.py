"""Subsampling checks on made-up arrays (no dataset content)."""
import copy
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from eval.subsample import subsample_indices, subsample_pose  # noqa: E402


def _pose(n):
    # frame i is filled with the value i so the kept frames can be identified
    return {
        "keypoints": [np.full((1, 133, 2), i, dtype=np.float32) for i in range(n)],
        "scores": [np.full((1, 133), i, dtype=np.float32) for i in range(n)],
    }


def test_at_or_under_cap_unchanged():
    for n in (0, 1, 100, 255, 256):
        assert subsample_indices(n, 256) == list(range(n))
        out = subsample_pose(_pose(n), 256)
        assert len(out["scores"]) == n
        for i, s in enumerate(out["scores"]):
            assert s[0, 0] == i


def test_longer_clip_exact_count_increasing_in_range():
    for n, cap in ((257, 256), (700, 256), (1056, 256), (30, 7), (10, 2)):
        idx = subsample_indices(n, cap)
        assert len(idx) == cap
        assert all(a < b for a, b in zip(idx, idx[1:]))
        assert idx[0] == 0 and idx[-1] == n - 1
        assert all(0 <= i < n for i in idx)


def test_pose_frames_follow_indices():
    out = subsample_pose(_pose(700), 256)
    idx = subsample_indices(700, 256)
    assert len(out["keypoints"]) == len(out["scores"]) == 256
    for k, s, i in zip(out["keypoints"], out["scores"], idx):
        assert k[0, 0, 0] == i and s[0, 0] == i


def test_deterministic():
    assert subsample_indices(700, 256) == subsample_indices(700, 256)
    a = subsample_pose(_pose(700), 256)
    b = subsample_pose(_pose(700), 256)
    assert all(np.array_equal(x, y) for k in a for x, y in zip(a[k], b[k]))


def test_input_not_modified():
    pose = _pose(700)
    before = copy.deepcopy(pose)
    out = subsample_pose(pose, 256)
    assert len(pose["keypoints"]) == len(pose["scores"]) == 700
    for k in before:
        assert all(np.array_equal(a, b) for a, b in zip(before[k], pose[k]))
    assert out is not pose and out["keypoints"] is not pose["keypoints"]
