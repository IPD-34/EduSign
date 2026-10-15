"""256-frame cap of data/extract_rtmw.py. Run: pytest tests/data/test_frame_cap.py"""
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "data"))
import extract_rtmw as ex  # noqa: E402


@pytest.mark.parametrize("n", [1, 100, 255, 256])
def test_up_to_cap_keeps_all_frames(n):
    assert ex.keep_indices(n) == list(range(n))


@pytest.mark.parametrize("n", [257, 500, 877, 5000])
def test_over_cap_gives_256_spread_indices(n):
    idx = ex.keep_indices(n)
    assert idx == [(i * (n - 1)) // 255 for i in range(256)]
    assert len(idx) == 256
    assert idx[0] == 0 and idx[-1] == n - 1
    assert all(b > a for a, b in zip(idx, idx[1:])), "indices must be strictly increasing"
    assert all(isinstance(i, int) for i in idx)


def _clip(path, n, w=64, h=48):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (w, h))
    assert vw.isOpened()
    for i in range(n):   # flat frame whose brightness ramps with the frame index
        vw.write(np.full((h, w, 3), (i * 255) // max(n - 1, 1), np.uint8))
    vw.release()
    return path


class _CountingDetector:
    def __init__(self):
        self.seen = []

    def __call__(self, frame):
        self.seen.append(float(frame.mean()))
        return np.zeros((0, 4))   # nobody detected -> zeros


def _wb():
    det = _CountingDetector()
    return SimpleNamespace(det_model=det, pose_model=None), det


def test_count_frames_uses_grab(tmp_path):
    assert ex.count_frames(_clip(tmp_path / "c.mp4", 37)) == 37


def test_short_clip_keeps_everything(tmp_path):
    wb, det = _wb()
    data = ex.extract_clip(wb, _clip(tmp_path / "s.mp4", 40))
    assert data["n_frames"] == 40 and len(data["keypoints"]) == len(data["scores"]) == len(det.seen) == 40


def test_long_clip_runs_pose_only_on_kept_frames(tmp_path):
    n = 300
    wb, det = _wb()
    data = ex.extract_clip(wb, _clip(tmp_path / "l.mp4", n))
    assert data["n_frames"] == n
    assert len(data["keypoints"]) == len(data["scores"]) == 256
    assert len(det.seen) == 256, "detector/pose must run only on the kept frames"
    assert det.seen[0] < 8 and det.seen[-1] > 247, "first and last frame must be kept"
    assert all(b >= a - 3 for a, b in zip(det.seen, det.seen[1:])), "frames must be processed in order"
    assert data["keypoints"][0].shape == (1, 133, 2) and data["keypoints"][0].dtype == np.float32
    assert data["scores"][0].shape == (1, 133) and data["scores"][0].dtype == np.float32
