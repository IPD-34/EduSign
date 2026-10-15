"""Leakage checks across manifests/{train,val,test}.csv and manifests/eval300.csv.
Run: pytest tests/data/test_leakage.py -s   (or: python tests/data/test_leakage.py)
Generate the manifests first with: python data/make_manifests.py
"""
import csv
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAN = ROOT / "manifests"
SPLITS = ("train", "val", "test")


def load(name):
    p = MAN / f"{name}.csv"
    assert p.exists(), f"{p} missing - run python data/make_manifests.py"
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def values(split, column):
    return {r[column] for r in load(split)}


def test_no_clip_id_in_more_than_one_split():
    for a, b in combinations(SPLITS, 2):
        shared = values(a, "clip_id") & values(b, "clip_id")
        assert not shared, f"{a}/{b} share {len(shared)} clip_ids, e.g. {sorted(shared)[:3]}"


def test_no_article_id_in_more_than_one_split():
    for a, b in combinations(SPLITS, 2):
        shared = values(a, "article_id") & values(b, "article_id")
        assert not shared, f"{a}/{b} share {len(shared)} article_ids, e.g. {sorted(shared)[:3]}"


def test_signer_overlap_report():
    """Informational only: signers may appear in several splits, so this never fails."""
    signers = {s: values(s, "signer_id") for s in SPLITS}
    print()
    for a, b in combinations(SPLITS, 2):
        print(f"signer overlap {a}/{b}: {len(signers[a] & signers[b])} "
              f"(of {len(signers[a])} and {len(signers[b])})")
    print(f"signers in all three splits: {len(signers['train'] & signers['val'] & signers['test'])}")


def test_eval300_is_subset_of_test():
    test_ids = values("test", "clip_id")
    eval_ids = {r["clip_id"] for r in load("eval300")}
    extra = eval_ids - test_ids
    assert not extra, f"eval300 has {len(extra)} clips not in test, e.g. {sorted(extra)[:3]}"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
