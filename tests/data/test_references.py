"""data/fetch_references.py on a tiny fake videos.csv with made-up text (no network). Run: pytest tests/data/test_references.py"""
import csv
import io
import shutil
import sys
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "data"))
import fetch_references as fr  # noqa: E402

MANIFEST_IDS = ["111-10_0_0", "222-20_1_2", "333-30_4_5"]
FAKE_ROWS = [   # includes clips that are NOT in the manifest, in a different order
    ("P1", "9", "0", "0", "999-90_0_0.mp4", "5.0", "Zebra quartz not in the manifest."),
    ("P2", "20", "1", "2", "222-20_1_2.mp4", "4.1", "Made-up sentence, with a comma."),
    ("P1", "10", "0", "0", "111-10_0_0.mp4", "3.2", 'Made-up "quoted" sentence.'),
    ("P3", "8", "2", "2", "888-80_2_2.mp4", "2.0", "Another unrelated invented line."),
    ("P3", "30", "4", "5", "333-30_4_5.mp4", "6.6", "Third invented sentence."),
]
HEADER = ["participant", "article", "section index", "sentence index", "video filename", "video length", "sentence text"]


def _fake_stream(rows=FAKE_ROWS):
    buf = io.StringIO(newline="")
    w = csv.writer(buf)
    w.writerow(HEADER)
    w.writerows(rows)
    return io.BytesIO(("﻿" + buf.getvalue()).encode("utf-8"))   # videos.csv starts with a BOM


class _FakeCtx:
    def __init__(self, rows=FAKE_ROWS):
        self.rows = rows

    def __enter__(self):
        return _fake_stream(self.rows)

    def __exit__(self, *exc):
        pass


def _manifest(tmp_path, ids=MANIFEST_IDS):
    p = tmp_path / "m.csv"
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip_id", "article_id", "signer_id", "split"])
        for c in ids:
            w.writerow([c, "1", "P0", "test"])
    return p


@pytest.fixture
def ignored_out():
    """An output path inside the repo's git-ignored local/ folder, removed afterwards."""
    d = ROOT / "local" / f"_pytest_refs_{uuid.uuid4().hex[:8]}"
    yield d / "references.csv"
    shutil.rmtree(d, ignore_errors=True)


def test_output_has_exactly_the_manifest_clip_ids(tmp_path, ignored_out, monkeypatch, capsys):
    monkeypatch.setattr(fr, "open_videos_csv", lambda url: _FakeCtx())
    assert fr.main(["--manifest", str(_manifest(tmp_path)), "--out", str(ignored_out)]) == 0
    with open(ignored_out, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        assert rd.fieldnames == ["clip_id", "reference"]
        rows = list(rd)
    assert [r["clip_id"] for r in rows] == MANIFEST_IDS
    assert set(r["clip_id"] for r in rows) == set(MANIFEST_IDS)
    texts = {r["clip_id"]: r["reference"] for r in rows}
    assert texts["111-10_0_0"] == 'Made-up "quoted" sentence.'      # quoting survives the round trip
    assert texts["222-20_1_2"] == "Made-up sentence, with a comma."
    out = capsys.readouterr()
    assert "3 rows" in out.out
    for _, _, _, _, _, _, text in FAKE_ROWS:
        assert text not in out.out + out.err, "sentence text must never be printed"


def test_warns_about_clip_without_reference_and_prints_no_text(tmp_path, ignored_out, monkeypatch, capsys):
    ids = MANIFEST_IDS + ["404-40_0_0"]
    monkeypatch.setattr(fr, "open_videos_csv", lambda url: _FakeCtx())
    assert fr.main(["--manifest", str(_manifest(tmp_path, ids)), "--out", str(ignored_out)]) == 0
    with open(ignored_out, newline="", encoding="utf-8") as f:
        got = [r["clip_id"] for r in csv.DictReader(f)]
    assert got == MANIFEST_IDS and "404-40_0_0" not in got
    out = capsys.readouterr().out
    assert "WARNING" in out and "404-40_0_0" in out and "wrote 3 rows" in out
    assert not any(r[-1] in out for r in FAKE_ROWS)


def test_refuses_output_path_that_is_not_gitignored(tmp_path, monkeypatch):
    monkeypatch.setattr(fr, "open_videos_csv", lambda url: pytest.fail("must refuse before fetching anything"))
    for bad in (ROOT / "manifests" / "references_should_not_exist.csv",   # inside the repo, not ignored
                tmp_path / "references.csv"):                              # outside the repo
        with pytest.raises(SystemExit, match="refusing to write"):
            fr.main(["--manifest", str(_manifest(tmp_path)), "--out", str(bad)])
        assert not bad.exists()


def test_is_gitignored():
    assert fr.is_gitignored(ROOT / "local" / "references.csv")
    assert not fr.is_gitignored(ROOT / "manifests" / "train.csv")
