"""data/overlay_check.py on a synthetic clip. Run: pytest tests/data/test_overlay.py"""
import pickle
import shutil
import sys
import uuid
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "data"))
import extract_rtmw as ex  # noqa: E402
import overlay_check as oc  # noqa: E402

W, H = 320, 240


def _clip(path, n):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (W, H))
    assert vw.isOpened()
    for i in range(n):   # flat gray frame whose brightness ramps with the frame index
        vw.write(np.full((H, W, 3), (i * 255) // max(n - 1, 1), np.uint8))
    vw.release()
    return path


def _pkl(path, n, kept_count=None):
    kept = ex.keep_indices(n)
    k = np.zeros((1, 133, 2), np.float32)
    k[0, :, 0] = 0.2 + 0.4 * np.arange(133) / 132      # x in [0.2, 0.6]: stays left of the pixel patch checked below
    k[0, :, 1] = 0.3
    d = {"keypoints": [k.copy() for _ in kept], "scores": [np.ones((1, 133), np.float32) for _ in kept], "n_frames": n}
    with open(path, "wb") as f:
        pickle.dump(d, f)
    return path, kept


@pytest.fixture
def out_dir():
    d = ROOT / "local" / f"_pytest_overlays_{uuid.uuid4().hex[:8]}"   # inside the git-ignored local/ folder
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_three_pngs_written_for_a_capped_clip(tmp_path, out_dir):
    n = 300
    clip = _clip(tmp_path / "clipA.mp4", n)
    pkl, kept = _pkl(tmp_path / "clipA.pkl", n)
    assert oc.main(["--clip", str(clip), "--pkl", str(pkl), "--out", str(out_dir)]) == 0

    want = [kept[0], kept[(len(kept) - 1) // 2], kept[-1]]
    assert want == [0, kept[127], n - 1]
    files = sorted(p.name for p in out_dir.iterdir())
    assert files == sorted(f"clipA_f{o:05d}.png" for o in want)
    for o in want:
        img = cv2.imread(str(out_dir / f"clipA_f{o:05d}.png"))
        assert img is not None and img.shape == (H, W, 3)
        # right frame decoded: bottom-right patch (no keypoints, no caption) has the ramp brightness
        patch = img[200:235, 290:315].astype(int)
        assert abs(patch.mean() - (o * 255) // (n - 1)) < 8, f"frame {o}"
        # keypoint 0 (body, green) is drawn at (0.2*W, 0.3*H)
        b, g, r = (int(v) for v in img[int(0.3 * H), int(0.2 * W)])
        assert (b, g, r) != (b, b, b) and g > b and g > r, "keypoint not drawn where x*width, y*height says"


def test_short_clip_draws_fewer_frames(tmp_path, out_dir):
    clip = _clip(tmp_path / "s.mp4", 2)
    pkl, _ = _pkl(tmp_path / "s.pkl", 2)
    assert oc.main(["--clip", str(clip), "--pkl", str(pkl), "--out", str(out_dir)]) == 0
    assert sorted(p.name for p in out_dir.iterdir()) == ["s_f00000.png", "s_f00001.png"]


def test_no_person_frame_draws_no_points(tmp_path, out_dir):
    clip = _clip(tmp_path / "z.mp4", 5)
    pkl, _ = _pkl(tmp_path / "z.pkl", 5)
    d = pickle.load(open(pkl, "rb"))
    d["keypoints"] = [np.zeros((1, 133, 2), np.float32) for _ in d["keypoints"]]
    pickle.dump(d, open(pkl, "wb"))
    assert oc.main(["--clip", str(clip), "--pkl", str(pkl), "--out", str(out_dir)]) == 0
    img = cv2.imread(str(out_dir / "z_f00000.png"))
    assert img[0:2, 0:2].mean() < 5   # top-left still the original frame 0 (black); nothing drawn at (0, 0)


def test_refuses_output_path_that_is_not_gitignored(tmp_path):
    clip = _clip(tmp_path / "c.mp4", 4)
    pkl, _ = _pkl(tmp_path / "c.pkl", 4)
    for bad in (tmp_path / "overlays", ROOT / "manifests" / "overlays_should_not_exist"):
        with pytest.raises(SystemExit, match="refusing to write"):
            oc.main(["--clip", str(clip), "--pkl", str(pkl), "--out", str(bad)])
        assert not bad.exists()


def test_wrong_pairing_is_rejected_and_writes_nothing(tmp_path, out_dir):
    clip = _clip(tmp_path / "c.mp4", 30)
    pkl, _ = _pkl(tmp_path / "c.pkl", 40)          # pkl says 40 frames, clip has 30
    with pytest.raises(SystemExit, match="n_frames=40"):
        oc.main(["--clip", str(clip), "--pkl", str(pkl), "--out", str(out_dir)])
    assert not out_dir.exists() or not list(out_dir.iterdir())


def test_old_pkl_without_n_frames_is_rejected(tmp_path, out_dir):
    clip = _clip(tmp_path / "c.mp4", 10)
    pkl, _ = _pkl(tmp_path / "c.pkl", 10)
    d = pickle.load(open(pkl, "rb"))
    del d["n_frames"]
    pickle.dump(d, open(pkl, "wb"))
    with pytest.raises(SystemExit, match="n_frames"):
        oc.main(["--clip", str(clip), "--pkl", str(pkl), "--out", str(out_dir)])
