"""data/make_pose_flags.py on synthetic clip_stats, plus checks on the real manifests/edge_eval300_pose_flags.csv.
Run: pytest tests/data/test_pose_flags.py
"""
import csv
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "data"))
import make_pose_flags as mp  # noqa: E402

MAN = ROOT / "manifests"
EXPECTED_COLUMNS = ["clip_id", "article_id", "signer_id", "split", "long_clip", "short", "no_person_frames",
                    "multi_person", "hand_dropout", "left_dominant", "nonstandard_fps", "nonstandard_resolution"]
FLAGS = EXPECTED_COLUMNS[4:]
TEXT_LIKE = ("text", "sentence", "caption", "transcript", "translation", "reference", "words")

NORMAL = dict(n_frames=300, fps=30.0, width=640, height=480, frames_no_person_frac=0.0, max_persons_detected=1,
              left_hand_mean_score=0.7, right_hand_mean_score=0.7, hand_low_conf_frac=0.1,
              left_path_len=3.0, right_path_len=3.0)

# one flag fires per case: (flag, stat overrides that should turn it on)
FIRES = [
    ("long_clip", dict(n_frames=843)),
    ("short", dict(n_frames=149)),
    ("no_person_frames", dict(frames_no_person_frac=0.001)),   # any frame without a person
    ("no_person_frames", dict(frames_no_person_frac=0.3)),
    ("multi_person", dict(max_persons_detected=2)),
    ("hand_dropout", dict(hand_low_conf_frac=0.21)),
    ("left_dominant", dict(left_path_len=4.6, right_path_len=3.0)),      # 4.6 > 1.5 * 3.0
    ("left_dominant", dict(left_path_len=0.5, right_path_len=0.0)),      # any left motion vs a still right hand
    ("nonstandard_fps", dict(fps=25.0)),
    ("nonstandard_resolution", dict(width=1280, height=720)),
    ("nonstandard_resolution", dict(width=640, height=360)),             # either side differing is enough
]
# exactly at the threshold must NOT fire (the rules are strict inequalities)
BOUNDARY = [
    ("long_clip", dict(n_frames=842)), ("short", dict(n_frames=150)), ("no_person_frames", dict(frames_no_person_frac=0.0)),
    ("multi_person", dict(max_persons_detected=1)), ("hand_dropout", dict(hand_low_conf_frac=0.2)),
    ("left_dominant", dict(left_path_len=4.5, right_path_len=3.0)), ("left_dominant", dict(left_path_len=0.0, right_path_len=0.0)),
    ("nonstandard_fps", dict(fps=30.5)), ("nonstandard_fps", dict(fps=29.9994)),
]


def _write_stats(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip_id"] + mp.STAT_COLUMNS)
        for cid, s in rows:
            w.writerow([cid] + [s[c] for c in mp.STAT_COLUMNS])


def _write_manifest(path, ids):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip_id", "article_id", "signer_id", "split"])
        for i, c in enumerate(ids):
            w.writerow([c, str(10 + i), f"P{i % 4}", "test"])


def _run(tmp_path, cases, extra_rows=(), name="out.csv"):
    """cases: [(clip_id, stat overrides)]; extra rows (not in the manifest) only join the stats file."""
    rows = [(cid, {**NORMAL, **o}) for cid, o in cases] + list(extra_rows)
    stats, manifest, out = tmp_path / "stats.csv", tmp_path / "m.csv", tmp_path / name
    _write_stats(stats, rows)
    _write_manifest(manifest, [c for c, _ in cases])
    assert mp.main([str(stats), "--manifest", str(manifest), "--out", str(out)]) == 0
    with open(out, newline="") as f:
        rd = csv.DictReader(f)
        return rd.fieldnames, {r["clip_id"]: r for r in rd}, out


def _true_flags(row):
    return [f for f in FLAGS if row[f] == "True"]


def test_normal_clip_has_no_flags(tmp_path):
    _, rows, _ = _run(tmp_path, [("n", {})])
    assert _true_flags(rows["n"]) == []


@pytest.mark.parametrize("flag,override", FIRES)
def test_each_flag_fires_alone(tmp_path, flag, override):
    # the "normal" companions keep the file's most common fps/resolution at the defaults
    cases = [("x", override)] + [(f"n{i}", {}) for i in range(5)]
    _, rows, _ = _run(tmp_path, cases)
    assert _true_flags(rows["x"]) == [flag]
    assert all(_true_flags(rows[f"n{i}"]) == [] for i in range(5))


@pytest.mark.parametrize("flag,override", BOUNDARY)
def test_value_at_the_threshold_does_not_fire(tmp_path, flag, override):
    cases = [("x", override)] + [(f"n{i}", {}) for i in range(5)]
    _, rows, _ = _run(tmp_path, cases)
    assert flag not in _true_flags(rows["x"])


def test_modes_use_the_whole_stats_file_but_only_manifest_clips_are_written(tmp_path):
    extra = [(f"train{i}", dict(NORMAL, fps=24.0)) for i in range(6)]    # 6 train clips at 24 fps beat 3 eval clips at 30
    cols, rows, _ = _run(tmp_path, [(f"e{i}", {}) for i in range(3)], extra_rows=extra)
    assert set(rows) == {"e0", "e1", "e2"}, "clips outside the manifest must not appear"
    assert all("nonstandard_fps" in _true_flags(r) for r in rows.values()), "mode fps is 24 over the whole file"
    assert mp.mode([1.0, 1.0, 2.0, 2.0]) == 1.0, "ties go to the smaller value"


def test_columns_row_order_booleans_and_no_text(tmp_path):
    cases = [("b", {}), ("a", dict(n_frames=900)), ("c", dict(max_persons_detected=3))]
    cols, rows, out = _run(tmp_path, cases)
    assert cols == EXPECTED_COLUMNS
    assert not [c for c in cols if any(w in c.lower() for w in TEXT_LIKE)]
    assert list(rows) == ["b", "a", "c"], "rows follow the manifest order"
    assert all(r[f] in ("True", "False") for r in rows.values() for f in FLAGS)
    assert rows["a"]["article_id"] == "11" and rows["a"]["split"] == "test" and rows["c"]["signer_id"] == "P2"


def test_deterministic(tmp_path):
    cases = [(f"c{i}", dict(n_frames=40 + 17 * i, fps=29.0 + 0.3 * i)) for i in range(12)]
    out1 = _run(tmp_path, cases, name="a.csv")[2]
    out2 = _run(tmp_path, cases, name="b.csv")[2]
    assert out1.read_bytes() == out2.read_bytes()


def test_missing_clip_and_bad_header_stop_without_output(tmp_path):
    _write_stats(tmp_path / "s.csv", [("a", dict(NORMAL))])
    _write_manifest(tmp_path / "m.csv", ["a", "missing"])
    with pytest.raises(SystemExit, match="no row"):
        mp.main([str(tmp_path / "s.csv"), "--manifest", str(tmp_path / "m.csv"), "--out", str(tmp_path / "o.csv")])
    (tmp_path / "bad.csv").write_text("clip_id,n_frames\na,10\n")
    with pytest.raises(SystemExit, match="missing column"):
        mp.main([str(tmp_path / "bad.csv"), "--manifest", str(tmp_path / "m.csv"), "--out", str(tmp_path / "o.csv")])
    assert not (tmp_path / "o.csv").exists()


def test_distribution_printout(tmp_path, capsys):
    _run(tmp_path, [(f"c{i}", dict(n_frames=100 + i)) for i in range(10)])
    out = capsys.readouterr().out
    assert "median" in out and "p90" in out and "n_frames" in out and "right_path_len" in out
    assert "long_clip" in out and "nonstandard_resolution" in out


def test_thresholds_and_flag_names():
    assert (mp.LONG_CLIP_FRAMES, mp.SHORT_FRAMES, mp.NO_PERSON_FRAC) == (842, 150, 0.0)
    assert mp.FPS_TOLERANCE == 0.5
    assert "capped" not in mp.FLAGS and "long_clip" in mp.FLAGS and not hasattr(mp, "CAPPED_FRAMES")


def test_long_clip_is_about_original_length_not_the_256_cap(tmp_path):
    # a 300-frame clip is longer than the extractor's 256-frame cap but is NOT a long clip any more
    _, rows, _ = _run(tmp_path, [("x", dict(n_frames=300))] + [(f"n{i}", {}) for i in range(3)])
    assert _true_flags(rows["x"]) == []


# ---------------------------------------------------------------- the real output
def _eval300():
    with open(MAN / "eval300.csv", newline="") as f:
        return [(r["clip_id"], r["article_id"], r["signer_id"], r["split"]) for r in csv.DictReader(f)]


def _real():
    p = MAN / "edge_eval300_pose_flags.csv"
    assert p.exists(), f"{p} missing - run python data/make_pose_flags.py local/clip_stats.csv"
    with open(p, newline="") as f:
        rd = csv.DictReader(f)
        return rd.fieldnames, list(rd)


def test_real_output_matches_eval300_and_has_boolean_flags():
    cols, rows = _real()
    assert cols == EXPECTED_COLUMNS
    assert [(r["clip_id"], r["article_id"], r["signer_id"], r["split"]) for r in rows] == _eval300()
    assert len(rows) == 300
    assert all(r[f] in ("True", "False") for r in rows for f in FLAGS)
    assert not [c for c in cols if any(w in c.lower() for w in TEXT_LIKE)]


def test_real_output_regenerates_identically(tmp_path):
    stats = ROOT / "local" / "clip_stats.csv"
    if not stats.exists():
        pytest.skip("local/clip_stats.csv not present (git-ignored)")
    out = tmp_path / "again.csv"
    assert mp.main([str(stats), "--out", str(out)]) == 0
    assert out.read_bytes() == (MAN / "edge_eval300_pose_flags.csv").read_bytes()
