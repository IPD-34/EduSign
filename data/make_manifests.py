"""Build the train/val/test clip manifests for ASL STEM Wiki.

Reads ONLY the 'video filename' and 'participant' columns of local/videos.csv (the dataset's clip index).
The sentence-text column is never read or written; outputs contain no text.

Output (manifests/{train,val,test}.csv): clip_id, article_id, signer_id, split
  clip_id    = filename without .mp4 (the pose file for a clip is <clip_id>.pkl)
  article_id = numeric Wikipedia page id inside the filename: <hash>-<article_id>_<section>_<sentence>.mp4
  signer_id  = participant column (P0..P36)
Split: by ARTICLE, seed 42, 80/10/10 (articles, not clips), so no article appears in two splits.

Also writes manifests/eval300.csv: 300 clips sampled uniformly at random (seed 42) from the test split, same columns.
And nested train subsets manifests/train_{500,1k,2k,3k}.csv: train clip_ids sorted, shuffled once with seed 42, the
first 500/1000/2000/3000 taken (each file is a prefix of the next, rows kept in shuffled order), same columns.

Usage (from the repo root):  python data/make_manifests.py
"""
import csv
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "local" / "videos.csv"
OUT = ROOT / "manifests"
SEED = 42
FRACTIONS = (0.8, 0.1, 0.1)   # train, val, test (of articles)
NAME_RE = re.compile(r"^\d+-(\d+)(?:&[^_]*)?_\d+_\d+\.mp4$")   # <hash>-<article_id>[&junk]_<sec>_<sent>.mp4
FIELDS = ["clip_id", "article_id", "signer_id", "split"]
EVAL_N = 300   # manifests/eval300.csv: uniform random sample of test clips, same seed
SUBSETS = (("train_500", 500), ("train_1k", 1000), ("train_2k", 2000), ("train_3k", 3000))   # nested train subsets


def read_index(path: Path):
    """Return [(clip_id, article_id, signer_id)] using only the two needed columns."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        try:
            i_name, i_signer = header.index("video filename"), header.index("participant")
        except ValueError as e:
            sys.exit(f"{path}: missing required column ({e}). Columns found: {header}")
        rows, bad = [], []
        for rec in reader:
            name, signer = rec[i_name].strip(), rec[i_signer].strip()   # sentence column is never touched
            m = NAME_RE.match(name)
            if not m or not signer:
                bad.append(name or "<empty>")
                continue
            rows.append((name[:-4], m.group(1), signer))
    if bad:
        sys.exit(f"{len(bad)} rows have an unparseable filename or empty participant, e.g. {bad[:3]} - not guessing, stopping.")
    return rows


def split_articles(article_ids, seed=SEED):
    arts = sorted(set(article_ids), key=int)
    random.Random(seed).shuffle(arts)
    n = len(arts)
    n_train, n_val = round(FRACTIONS[0] * n), round(FRACTIONS[1] * n)
    return {"train": set(arts[:n_train]), "val": set(arts[n_train:n_train + n_val]), "test": set(arts[n_train + n_val:])}


def main():
    if not SRC.exists():
        sys.exit(f"{SRC} not found. Copy the dataset's videos.csv to local/videos.csv (local/ is git-ignored).")
    rows = read_index(SRC)
    if len({r[0] for r in rows}) != len(rows):
        sys.exit("clip ids are not unique - not guessing, stopping.")
    parts = split_articles(r[1] for r in rows)
    OUT.mkdir(exist_ok=True)
    for split in ("train", "val", "test"):
        sel = sorted((r for r in rows if r[1] in parts[split]), key=lambda r: r[0])
        with open(OUT / f"{split}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(FIELDS)
            w.writerows((cid, aid, sid, split) for cid, aid, sid in sel)
        print(f"{split:5s} rows={len(sel):6d} articles={len(parts[split]):4d} signers={len({r[2] for r in sel}):3d}")
        if split == "test":
            test_rows = sel
        elif split == "train":
            train_rows = sel
    if len(test_rows) < EVAL_N:
        sys.exit(f"test split has only {len(test_rows)} clips, need {EVAL_N} for eval300 - not guessing, stopping.")
    sample = sorted(random.Random(SEED).sample(test_rows, EVAL_N), key=lambda r: r[0])   # uniform over clips, no replacement
    with open(OUT / "eval300.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(FIELDS)
        w.writerows((cid, aid, sid, "test") for cid, aid, sid in sample)
    print(f"eval300 rows={len(sample):4d} articles={len({r[1] for r in sample}):4d} signers={len({r[2] for r in sample}):3d}")

    if len(train_rows) < SUBSETS[-1][1]:
        sys.exit(f"train split has only {len(train_rows)} clips, need {SUBSETS[-1][1]} - not guessing, stopping.")
    shuffled = sorted(train_rows, key=lambda r: r[0])   # fixed starting order, then one shuffle
    random.Random(SEED).shuffle(shuffled)
    for name, k in SUBSETS:
        sub = shuffled[:k]
        with open(OUT / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(FIELDS)
            w.writerows((cid, aid, sid, "train") for cid, aid, sid in sub)
        print(f"{name:9s} rows={len(sub):5d} articles={len({r[1] for r in sub}):4d} signers={len({r[2] for r in sub}):3d}")


if __name__ == "__main__":
    main()
