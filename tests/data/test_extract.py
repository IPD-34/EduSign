"""Shape/format checks for data/extract_rtmw.py on a small clip.
Run: pytest tests/data/test_extract.py -s

The clip is synthetic (written with OpenCV) unless EDUSIGN_TEST_CLIP points to a real .mp4, e.g.
    EDUSIGN_TEST_CLIP=/path/to/clip.mp4 pytest tests/data/test_extract.py -s
The model tests need the rtmlib lightweight ONNX models (downloaded on first use) and a CUDA onnxruntime;
set EDUSIGN_TEST_DEVICE=cpu to run them without a GPU. They skip, with the reason, if the models cannot be loaded.
"""
import csv
import os
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "data"))
import extract_rtmw as ex  # noqa: E402

N_FRAMES, W, H = 6, 320, 240


def _make_clip(path, n=N_FRAMES):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (W, H))
    assert vw.isOpened(), "cannot write a test clip with OpenCV"
    rng = np.random.default_rng(0)
    for _ in range(n):
        vw.write(rng.integers(0, 255, (H, W, 3), dtype=np.uint8))
    vw.release()
    return path


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    real = os.environ.get("EDUSIGN_TEST_CLIP")
    if real:
        assert Path(real).exists(), f"EDUSIGN_TEST_CLIP not found: {real}"
        return Path(real)
    return _make_clip(tmp_path_factory.mktemp("clips") / "tiny.mp4")


@pytest.fixture(scope="module")
def extractor():
    device = os.environ.get("EDUSIGN_TEST_DEVICE", "cuda")
    try:
        return ex.build_extractor(device)
    except RuntimeError:
        raise   # CUDA requested but not active: a real failure, not a skip
    except Exception as e:   # no internet / model download failed
        pytest.skip(f"RTMW models unavailable: {type(e).__name__}: {e}")


def _check_shapes(data, n_frames=None):
    assert set(data) == {"keypoints", "scores"}
    kp, sc = data["keypoints"], data["scores"]
    assert isinstance(kp, list) and isinstance(sc, list)
    assert len(kp) == len(sc) and len(kp) > 0
    if n_frames is not None:
        assert len(kp) == n_frames
    for k, s in zip(kp, sc):
        assert k.shape == (1, 133, 2), k.shape
        assert s.shape == (1, 133), s.shape
        assert np.isfinite(k).all() and np.isfinite(s).all()


def test_no_person_gives_zeros():
    class NoPerson:
        det_model = staticmethod(lambda frame: np.zeros((0, 4)))
        pose_model = None   # must not be called when nothing is detected

    k, s = ex.extract_frame(NoPerson, np.zeros((H, W, 3), np.uint8))
    assert k.shape == (1, 133, 2) and s.shape == (1, 133)
    assert not k.any() and not s.any()


def test_keeps_largest_person():
    seen = {}

    class TwoPeople:
        det_model = staticmethod(lambda frame: np.array([[0, 0, 10, 10], [5, 5, 200, 200], [0, 0, 50, 50]], float))

        @staticmethod
        def pose_model(frame, bboxes):
            seen["bboxes"] = [list(b) for b in bboxes]
            return np.ones((len(bboxes), 133, 2)), np.ones((len(bboxes), 133))

    k, s = ex.extract_frame(TwoPeople, np.zeros((H, W, 3), np.uint8))
    assert seen["bboxes"] == [[5, 5, 200, 200]]
    assert k.shape == (1, 133, 2) and s.shape == (1, 133)


def test_extract_clip_shapes_and_normalisation(extractor, clip):
    cap = cv2.VideoCapture(str(clip))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    data = ex.extract_clip(extractor, clip)
    _check_shapes(data, n_frames=n if n > 0 else None)
    for k, s in zip(data["keypoints"], data["scores"]):
        # normalised by frame size: detected points stay roughly inside the frame (small overshoot allowed)
        assert k.min() > -0.5 and k.max() < 1.5
        assert s.min() >= 0


def test_cli_writes_one_pkl_per_clip(extractor, clip, tmp_path):
    clips = tmp_path / "clips"
    clips.mkdir()
    (clips / "a.mp4").write_bytes(Path(clip).read_bytes())
    manifest = tmp_path / "m.csv"
    with open(manifest, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip_id", "article_id", "signer_id", "split"])
        w.writerow(["a", "1", "P0", "test"])
    out = tmp_path / "out"
    rc = ex.main(["--manifest", str(manifest), "--clips", str(clips), "--out", str(out),
                  "--device", os.environ.get("EDUSIGN_TEST_DEVICE", "cuda")])
    assert rc == 0
    assert sorted(p.name for p in out.iterdir()) == ["a.pkl"]
    with open(out / "a.pkl", "rb") as f:
        _check_shapes(pickle.load(f))
