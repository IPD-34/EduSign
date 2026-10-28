"""data/check_poses.py on small fake .pkl files. Run: pytest tests/data/test_check_poses.py"""
import csv
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "data"))
import check_poses as cp  # noqa: E402


def _good(n=10, n_frames=None):
    return {"keypoints": [np.zeros((1, 133, 2), np.float32) for _ in range(n)],
            "scores": [np.zeros((1, 133), np.float32) for _ in range(n)],
            "n_frames": n if n_frames is None else n_frames}


def _dump(path, obj):
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    return path


def _mutated(**kw):
    d = _good()
    d.update(kw)
    return d


def _without(key):
    d = _good()
    del d[key]
    return d


BAD_CASES = {
    "unequal_lengths": (_mutated(scores=_good(9)["scores"]), "length mismatch"),
    "257_frames": (_good(257, n_frames=300), "more than 256 frames"),
    "bad_keypoint_shape": (_mutated(keypoints=[np.zeros((1, 133, 3), np.float32)] * 10), "keypoints: 10 frame(s)"),
    "keypoints_missing_person_axis": (_mutated(keypoints=[np.zeros((133, 2), np.float32)] * 10), "keypoints: 10 frame(s)"),
    "bad_score_shape": (_mutated(scores=[np.zeros((1, 17), np.float32)] * 10), "scores: 10 frame(s)"),
    "no_n_frames": (_without("n_frames"), "missing key(s): ['n_frames']"),
    "no_scores": (_without("scores"), "missing key(s): ['scores']"),
    "n_frames_not_int": (_mutated(n_frames="12"), "n_frames must be a positive int"),
    "n_frames_zero": (_mutated(n_frames=0), "n_frames must be a positive int"),
    "empty": (_good(0, n_frames=5), "no frames"),
    "not_a_dict": ([1, 2, 3], "not a dict"),
}


def test_good_files_pass(tmp_path):
    assert cp.check_pose_file(_dump(tmp_path / "a.pkl", _good(10))) == []
    assert cp.check_pose_file(_dump(tmp_path / "b.pkl", _good(256, n_frames=877))) == []   # a capped long clip


@pytest.mark.parametrize("name", sorted(BAD_CASES))
def test_bad_files_are_flagged(tmp_path, name):
    obj, expected = BAD_CASES[name]
    problems = cp.check_pose_file(_dump(tmp_path / "x.pkl", obj))
    assert any(expected in p for p in problems), problems


def test_unloadable_file_is_flagged(tmp_path):
    p = tmp_path / "junk.pkl"
    p.write_bytes(b"this is not a pickle")
    assert any("cannot load" in x for x in cp.check_pose_file(p))


def _manifest(tmp_path, ids):
    p = tmp_path / "m.csv"
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip_id", "article_id", "signer_id", "split"])
        for c in ids:
            w.writerow([c, "1", "P0", "test"])
    return p


def test_main_counts_and_exit_code(tmp_path, capsys):
    poses = tmp_path / "poses"
    poses.mkdir()
    _dump(poses / "ok1.pkl", _good(5))
    _dump(poses / "ok2.pkl", _good(256, n_frames=400))
    _dump(poses / "broken.pkl", _without("n_frames"))        # in manifest but bad
    _dump(poses / "stranger.pkl", _good(3))                  # not in manifest
    (poses / "half.pkl.tmp").write_bytes(b"x")               # leftover temp file is ignored
    manifest = _manifest(tmp_path, ["ok1", "ok2", "broken", "gone"])   # "gone" has no .pkl
    assert cp.main(["--manifest", str(manifest), "--poses", str(poses)]) == 1
    out = capsys.readouterr().out
    assert "expected=4 found=3 missing=1 bad=1 extra=1" in out
    assert "MISSING" in out and "gone" in out
    assert "EXTRA" in out and "stranger" in out
    assert "BAD" in out and "broken: missing key(s): ['n_frames']" in out


def test_main_all_good_exits_zero_even_with_extras(tmp_path, capsys):
    poses = tmp_path / "poses"
    poses.mkdir()
    _dump(poses / "a.pkl", _good(5))
    _dump(poses / "b.pkl", _good(7))
    _dump(poses / "extra.pkl", _good(2))
    assert cp.main(["--manifest", str(_manifest(tmp_path, ["a", "b"])), "--poses", str(poses)]) == 0
    assert "expected=2 found=2 missing=0 bad=0 extra=1" in capsys.readouterr().out


def test_main_bad_extra_file_still_fails(tmp_path, capsys):
    poses = tmp_path / "poses"
    poses.mkdir()
    _dump(poses / "a.pkl", _good(5))
    _dump(poses / "extra.pkl", _good(257, n_frames=300))
    assert cp.main(["--manifest", str(_manifest(tmp_path, ["a"])), "--poses", str(poses)]) == 1
    assert "bad=1" in capsys.readouterr().out
