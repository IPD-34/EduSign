"""--stats of data/extract_rtmw.py. Run: pytest tests/data/test_stats.py"""
import csv
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "data"))
import extract_rtmw as ex  # noqa: E402
from test_extract import clip, extractor  # noqa: E402,F401  (fixtures: tiny synthetic clip, real RTMW extractor)

EXPECTED_COLUMNS = ["clip_id", "n_frames", "fps", "width", "height", "frames_no_person_frac", "max_persons_detected",
                    "left_hand_mean_score", "right_hand_mean_score", "hand_low_conf_frac", "left_path_len",
                    "right_path_len"]
FORBIDDEN_WORDS = ("sentence", "text", "caption", "transcript", "translation", "reference")


def test_keypoint_indices_match_rtmlib_layout():
    from rtmlib.visualization.skeleton.coco133 import coco133
    ki = coco133["keypoint_info"]
    assert ki[ex.LEFT_WRIST]["name"] == "left_wrist" and ki[ex.RIGHT_WRIST]["name"] == "right_wrist"
    lh, rh = range(133)[ex.LEFT_HAND], range(133)[ex.RIGHT_HAND]
    assert len(lh) == len(rh) == 21
    assert ki[lh[0]]["name"] == "left_hand_root" and ki[lh[-1]]["name"] == "left_pinky_finger4"
    assert ki[rh[0]]["name"] == "right_hand_root" and ki[rh[-1]]["name"] == "right_pinky_finger4"


def _frame(lw=(0.0, 0.0), rw=(0.0, 0.0), lh=0.0, rh=0.0):
    k = np.zeros((1, 133, 2), np.float32)
    s = np.zeros((1, 133), np.float32)
    k[0, ex.LEFT_WRIST], k[0, ex.RIGHT_WRIST] = lw, rw
    s[0, ex.LEFT_HAND], s[0, ex.RIGHT_HAND] = lh, rh
    return k, s


def test_compute_clip_stats_known_values():
    frames = [_frame(lw=(0.1, 0.1), rw=(0.5, 0.5), lh=0.9, rh=0.5),   # person
              _frame(lw=(0.4, 0.5), rw=(0.5, 0.9), lh=0.7, rh=0.1),   # person: left moves 0.5, right moves 0.4
              _frame(),                                               # no person: zeros, pairs with it are skipped
              _frame(lw=(0.4, 0.5), rw=(0.5, 0.9), lh=0.2, rh=0.2)]   # person
    kp, sc = [f[0] for f in frames], [f[1] for f in frames]
    st = ex.compute_clip_stats(kp, sc, n_boxes=[1, 2, 0, 1], n_frames=10, fps=25.0, width=640, height=480)
    assert st["n_frames"] == 10 and st["fps"] == 25.0 and st["width"] == 640 and st["height"] == 480
    assert st["frames_no_person_frac"] == pytest.approx(0.25)
    assert st["max_persons_detected"] == 2
    assert st["left_hand_mean_score"] == pytest.approx((0.9 + 0.7 + 0.0 + 0.2) / 4)
    assert st["right_hand_mean_score"] == pytest.approx((0.5 + 0.1 + 0.0 + 0.2) / 4)
    # per-frame mean of both hands: 0.7, 0.4, 0.0, 0.2 -> below 0.3 on 2 of 4 frames
    assert st["hand_low_conf_frac"] == pytest.approx(0.5)
    assert st["left_path_len"] == pytest.approx(0.5)    # only the 0->1 pair; pairs touching the empty frame are skipped
    assert st["right_path_len"] == pytest.approx(0.4)


def test_compute_clip_stats_all_frames_empty():
    f = _frame()
    st = ex.compute_clip_stats([f[0]] * 3, [f[1]] * 3, n_boxes=[0, 0, 0], n_frames=3, fps=0.0, width=32, height=24)
    assert st["frames_no_person_frac"] == 1.0 and st["max_persons_detected"] == 0
    assert st["left_path_len"] == st["right_path_len"] == 0.0 and st["hand_low_conf_frac"] == 1.0


# ---- a fake extractor (no models) driving the real frame loop ----------------------------------------------------
def _clip(path, n=300, w=64, h=48):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (w, h))
    assert vw.isOpened()
    for i in range(n):
        vw.write(np.full((h, w, 3), (i * 255) // (n - 1), np.uint8))
    vw.release()
    return path


class _Det:
    def __call__(self, frame):
        m = int(frame.mean())
        if m % 5 == 0:
            return np.zeros((0, 4))                                   # nobody
        return np.array([[0, 0, 10 + m % 7, 10], [0, 0, 40, 40], [0, 0, 5, 5]][: 1 + m % 3], float)


class _Pose:
    def __call__(self, frame, bboxes):
        m = float(frame.mean())
        k = np.full((1, 133, 2), m, np.float32) + np.arange(133, dtype=np.float32)[None, :, None] * 0.01
        s = np.full((1, 133), (m % 10) / 10, np.float32)
        return k, s


def _fake_wb():
    return SimpleNamespace(det_model=_Det(), pose_model=_Pose())


def _same(a, b):
    assert a["n_frames"] == b["n_frames"]
    assert len(a["keypoints"]) == len(b["keypoints"]) == len(a["scores"]) == len(b["scores"])
    for x, y in zip(a["keypoints"] + a["scores"], b["keypoints"] + b["scores"]):
        assert x.dtype == y.dtype and x.shape == y.shape and np.array_equal(x, y)


def test_fake_extractor_outputs_identical_with_and_without_stats(tmp_path):
    path = _clip(tmp_path / "c.mp4")
    plain = ex.extract_clip(_fake_wb(), path)
    st = {}
    with_stats = ex.extract_clip(_fake_wb(), path, stats_out=st)
    _same(plain, with_stats)
    assert set(plain) == set(with_stats) == {"keypoints", "scores", "n_frames"}
    assert set(st) == set(EXPECTED_COLUMNS) - {"clip_id"}
    assert st["n_frames"] == 300 and (st["width"], st["height"]) == (64, 48)
    assert 0 < st["frames_no_person_frac"] < 1 and st["max_persons_detected"] == 3


def _manifest(tmp_path, ids):
    p = tmp_path / "m.csv"
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip_id", "article_id", "signer_id", "split"])
        for c in ids:
            w.writerow([c, "1", "P0", "test"])
    return p


def test_cli_stats_file_and_unchanged_default(tmp_path, monkeypatch):
    ids = ["a", "b"]
    clips = tmp_path / "clips"
    clips.mkdir()
    _clip(clips / "a.mp4", 40)
    _clip(clips / "b.mp4", 300)
    monkeypatch.setattr(ex, "build_extractor", lambda *a, **k: _fake_wb())
    monkeypatch.setattr(ex, "assert_on_cuda", lambda wb: None)
    m = str(_manifest(tmp_path, ids))
    plain, withs = tmp_path / "plain", tmp_path / "stats"
    assert ex.main(["--manifest", m, "--clips", str(clips), "--out", str(plain)]) == 0
    assert ex.main(["--manifest", m, "--clips", str(clips), "--out", str(withs), "--stats"]) == 0

    assert not (plain / "clip_stats.csv").exists(), "default behaviour must not write clip_stats.csv"
    assert sorted(p.name for p in plain.iterdir()) == ["a.pkl", "b.pkl", "frame_counts.csv"]
    assert sorted(p.name for p in withs.iterdir()) == ["a.pkl", "b.pkl", "clip_stats.csv", "frame_counts.csv"]
    assert (plain / "frame_counts.csv").read_text() == (withs / "frame_counts.csv").read_text()
    for c in ids:
        with open(plain / f"{c}.pkl", "rb") as f1, open(withs / f"{c}.pkl", "rb") as f2:
            _same(pickle.load(f1), pickle.load(f2))

    with open(withs / "clip_stats.csv", newline="") as f:
        rd = csv.DictReader(f)
        assert rd.fieldnames == EXPECTED_COLUMNS
        rows = list(rd)
    assert [r["clip_id"] for r in rows] == ids
    for r in rows:   # numbers only besides clip_id: no sentence text can be in here
        for col in EXPECTED_COLUMNS[1:]:
            float(r[col])
    assert not [c for c in EXPECTED_COLUMNS if any(w in c.lower() for w in FORBIDDEN_WORDS)]
    assert int(rows[0]["n_frames"]) == 40 and int(rows[1]["n_frames"]) == 300


def test_cli_stats_header_written_once_across_reruns(tmp_path, monkeypatch):
    clips = tmp_path / "clips"
    clips.mkdir()
    for c in ("a", "b"):
        _clip(clips / f"{c}.mp4", 20)
    monkeypatch.setattr(ex, "build_extractor", lambda *a, **k: _fake_wb())
    monkeypatch.setattr(ex, "assert_on_cuda", lambda wb: None)
    out = tmp_path / "out"
    base = ["--clips", str(clips), "--out", str(out), "--stats"]
    assert ex.main(["--manifest", str(_manifest(tmp_path, ["a"]))] + base) == 0
    assert ex.main(["--manifest", str(_manifest(tmp_path, ["a", "b"]))] + base) == 0   # "a" skipped, "b" appended
    lines = (out / "clip_stats.csv").read_text().splitlines()
    assert lines[0] == ",".join(EXPECTED_COLUMNS) and len(lines) == 3
    assert [l.split(",")[0] for l in lines[1:]] == ["a", "b"]


# ---- the real models ----------------------------------------------------------------------------------------------
def test_real_models_identical_with_and_without_stats(extractor, clip):
    plain = ex.extract_clip(extractor, clip)
    st = {}
    with_stats = ex.extract_clip(extractor, clip, stats_out=st)
    _same(plain, with_stats)
    assert set(st) == set(EXPECTED_COLUMNS) - {"clip_id"}
