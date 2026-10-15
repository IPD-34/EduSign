"""Checks on manifests/{train,val,test}.csv. Run: pytest tests/data/test_manifest.py -s   (or: python tests/data/test_manifest.py)
Generate them first with: python data/make_manifests.py
"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAN = ROOT / "manifests"
SPLITS = ("train", "val", "test")
EXPECTED_COLUMNS = ["clip_id", "article_id", "signer_id", "split"]
FORBIDDEN_WORDS = ("sentence", "text", "caption", "transcript", "translation", "reference")


def load(split):
    p = MAN / f"{split}.csv"
    assert p.exists(), f"{p} missing - run python data/make_manifests.py"
    with open(p, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        return rd.fieldnames, list(rd)


def test_schema():
    for s in SPLITS:
        cols, rows = load(s)
        assert cols == EXPECTED_COLUMNS, f"{s}: columns {cols}"
        assert rows, f"{s}: empty"
        assert all(r["split"] == s for r in rows), f"{s}: wrong split label"
        assert all(r["clip_id"] and r["article_id"].isdigit() and r["signer_id"] for r in rows), f"{s}: empty/invalid field"
        assert not any(r["clip_id"].endswith(".mp4") for r in rows), f"{s}: clip_id must not include .mp4"


def test_clip_ids_unique():
    seen = set()
    for s in SPLITS:
        for r in load(s)[1]:
            assert r["clip_id"] not in seen, f"duplicate clip_id {r['clip_id']}"
            seen.add(r["clip_id"])


def test_no_sentence_column():
    for s in SPLITS:
        cols, rows = load(s)
        assert not [c for c in cols if any(w in c.lower() for w in FORBIDDEN_WORDS)], f"{s}: text-like column in {cols}"
        # values must be short identifiers, never prose
        assert all(len(r["clip_id"]) < 80 and " " not in r["clip_id"] for r in rows), f"{s}: clip_id looks like text"
        assert all(" " not in r["signer_id"] and len(r["signer_id"]) < 10 for r in rows), f"{s}: signer_id looks like text"


def test_splits_share_no_article():
    arts = {s: {r["article_id"] for r in load(s)[1]} for s in SPLITS}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        assert not (arts[a] & arts[b]), f"{a}/{b} share articles: {sorted(arts[a] & arts[b])[:5]}"


def test_report_counts():
    total = 0
    print()
    for s in SPLITS:
        rows = load(s)[1]
        total += len(rows)
        print(f"{s:5s} rows={len(rows):6d} articles={len({r['article_id'] for r in rows}):4d} signers={len({r['signer_id'] for r in rows}):3d}")
    print(f"total rows={total}")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
