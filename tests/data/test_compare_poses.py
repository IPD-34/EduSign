"""data/compare_poses.py on small fake .pkl files. Run: pytest tests/data/test_compare_poses.py"""
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "data"))
import compare_poses as cmp  # noqa: E402


def _pose(seed, n=6, dtype=np.float32, n_frames=None):
    rng = np.random.default_rng(seed)
    return {"keypoints": [rng.random((1, 133, 2)).astype(dtype) for _ in range(n)],
            "scores": [rng.random((1, 133)).astype(dtype) for _ in range(n)],
            "n_frames": n if n_frames is None else n_frames}


def _write_folder(folder, clips):
    folder.mkdir(parents=True, exist_ok=True)
    for cid, d in clips.items():
        with open(folder / f"{cid}.pkl", "wb") as f:
            pickle.dump(d, f)


@pytest.fixture
def folders(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    clips = {"c1": _pose(1), "c2": _pose(2), "c3": _pose(3)}
    _write_folder(a, clips)
    _write_folder(b, {k: pickle.loads(pickle.dumps(v)) for k, v in clips.items()})   # deep copy
    return a, b


def _run(a, b, *extra):
    return cmp.main(["--a", str(a), "--b", str(b), *extra])


def test_identical_folders_give_zero(folders, capsys):
    a, b = folders
    assert _run(a, b) == 0
    out = capsys.readouterr().out
    assert out.count("max|d keypoints|=0.000e+00  max|d scores|=0.000e+00  OK") == 3
    assert "compared=3 exceeding_tol=0 only_in_a=0 only_in_b=0" in out


def test_perturbed_file_is_reported(folders, capsys):
    a, b = folders
    d = pickle.load(open(b / "c2.pkl", "rb"))
    d["keypoints"][4][0, 7, 1] += 0.05          # one point, one frame
    pickle.dump(d, open(b / "c2.pkl", "wb"))
    assert _run(a, b) == 1
    out = capsys.readouterr().out
    line = next(l for l in out.splitlines() if l.startswith("c2 "))
    assert "EXCEEDS tol" in line and "max|d keypoints|=5.000e-02" in line and "max|d scores|=0.000e+00" in line
    assert sum("OK" in l for l in out.splitlines() if l.startswith(("c1 ", "c3 "))) == 2
    assert "exceeding_tol=1" in out


def test_perturbed_scores_are_reported(folders, capsys):
    a, b = folders
    d = pickle.load(open(b / "c1.pkl", "rb"))
    d["scores"][0][0, 0] += 0.5
    pickle.dump(d, open(b / "c1.pkl", "wb"))
    assert _run(a, b) == 1
    assert "max|d scores|=5.000e-01  EXCEEDS tol" in capsys.readouterr().out


def test_tolerance_is_respected(folders):
    a, b = folders
    d = pickle.load(open(b / "c3.pkl", "rb"))
    d["keypoints"][0] = d["keypoints"][0] + np.float32(1e-4)
    pickle.dump(d, open(b / "c3.pkl", "wb"))
    assert _run(a, b) == 0                       # 1e-4 < default 1e-3
    assert _run(a, b, "--tol", "1e-5") == 1      # but above a tighter tolerance


def test_clips_in_only_one_folder_are_listed_and_do_not_fail(folders, capsys):
    a, b = folders
    _write_folder(a, {"only_a1": _pose(8), "only_a2": _pose(9)})
    _write_folder(b, {"only_b": _pose(10)})
    assert _run(a, b) == 0
    out = capsys.readouterr().out
    assert "only in a (2):\n  only_a1\n  only_a2" in out
    assert "only in b (1):\n  only_b" in out
    assert "compared=3 exceeding_tol=0 only_in_a=2 only_in_b=1" in out


def test_uncomparable_files_count_as_exceeding(folders, capsys):
    a, b = folders
    _write_folder(b, {"c1": _pose(1, n=5)})                      # different frame count
    (b / "c2.pkl").write_bytes(b"not a pickle")                   # unreadable
    assert _run(a, b) == 1
    out = capsys.readouterr().out
    assert "c1  CANNOT COMPARE" in out and "6 frames in a vs 5 in b" in out
    assert "c2  CANNOT COMPARE" in out
    assert "exceeding_tol=2" in out


def test_nan_difference_fails(folders):
    a, b = folders
    d = pickle.load(open(b / "c1.pkl", "rb"))
    d["keypoints"][0][0, 0, 0] = np.nan
    pickle.dump(d, open(b / "c1.pkl", "wb"))
    assert _run(a, b) == 1


def test_dtype_and_n_frames_differences_are_notes_only(tmp_path, capsys):
    a, b = tmp_path / "a", tmp_path / "b"
    _write_folder(a, {"c": _pose(1, dtype=np.float32, n_frames=6)})
    _write_folder(b, {"c": _pose(1, dtype=np.float64, n_frames=7)})   # same values (float32 vs float64 rounding aside)
    assert _run(a, b) == 0
    out = capsys.readouterr().out
    assert "keypoints dtype float32 vs float64" in out and "n_frames 6 vs 7" in out


def test_no_common_clip_is_an_error(tmp_path, capsys):
    a, b = tmp_path / "a", tmp_path / "b"
    _write_folder(a, {"x": _pose(1)})
    _write_folder(b, {"y": _pose(2)})
    assert _run(a, b) == 2
    assert "nothing compared" in capsys.readouterr().out


def test_missing_folder_is_rejected(tmp_path):
    with pytest.raises(SystemExit, match="not a folder"):
        _run(tmp_path / "nope", tmp_path)
