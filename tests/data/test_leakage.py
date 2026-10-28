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


# ---- derived manifests: every clip must exist in the split manifests, with the same split ------------------------
def split_of_clip():
    """{clip_id: split} over manifests/{train,val,test}.csv."""
    return {r["clip_id"]: s for s in SPLITS for r in load(s)}


def load_derived(name, script):
    p = MAN / f"{name}.csv"
    assert p.exists(), f"{p} missing - run python data/{script}"
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_text_flags_clips_exist_in_manifests_with_same_split():
    want = split_of_clip()
    rows = load_derived("edge_text_flags", "make_edge_flags.py")
    assert rows, "edge_text_flags.csv is empty"
    unknown = [r["clip_id"] for r in rows if r["clip_id"] not in want]
    assert not unknown, f"{len(unknown)} clips are not in any manifest, e.g. {unknown[:3]}"
    wrong = [r["clip_id"] for r in rows if r["split"] != want[r["clip_id"]]]
    assert not wrong, f"{len(wrong)} clips have a different split than in the manifests, e.g. {wrong[:3]}"


def test_pose_flags_clips_exist_in_manifests_with_same_split():
    want = split_of_clip()
    rows = load_derived("edge_eval300_pose_flags", "make_pose_flags.py local/clip_stats.csv")
    assert rows, "edge_eval300_pose_flags.csv is empty"
    unknown = [r["clip_id"] for r in rows if r["clip_id"] not in want]
    assert not unknown, f"{len(unknown)} clips are not in any manifest, e.g. {unknown[:3]}"
    wrong = [r["clip_id"] for r in rows if r["split"] != want[r["clip_id"]]]
    assert not wrong, f"{len(wrong)} clips have a different split than in the manifests, e.g. {wrong[:3]}"


def test_edge_candidates_are_inside_test_and_disjoint_from_train_and_val():
    want = split_of_clip()
    rows = load_derived("edge_candidates_test", "make_edge_flags.py")
    assert rows, "edge_candidates_test.csv is empty"
    ids = {r["clip_id"] for r in rows}
    unknown = sorted(i for i in ids if i not in want)
    assert not unknown, f"{len(unknown)} candidates are not in any manifest, e.g. {unknown[:3]}"
    assert all(r["split"] == "test" for r in rows), "candidate rows must carry split == test"
    not_test = sorted(i for i in ids if want[i] != "test")
    assert not not_test, f"{len(not_test)} candidates are outside the test split, e.g. {not_test[:3]}"
    for other in ("train", "val"):
        shared = ids & values(other, "clip_id")
        assert not shared, f"{len(shared)} candidates also appear in {other}, e.g. {sorted(shared)[:3]}"
    assert ids <= values("test", "clip_id")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
