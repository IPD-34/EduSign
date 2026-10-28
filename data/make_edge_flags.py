"""Per-clip edge-case flags about the sentence text, for slicing eval results. Outputs contain NO text.

Reads ONLY the 'video filename' and 'sentence text' columns of local/videos.csv (git-ignored dataset index) plus
manifests/{train,val,test}.csv, and writes manifests/edge_text_flags.csv, one row per clip of the three manifests:
    clip_id, article_id, signer_id, split      (copied from the manifests)
    has_digit       the sentence contains a digit (regex \\d: ASCII 0-9 and other Unicode decimal digits)
    has_acronym     the sentence contains a run of 2+ capital ASCII letters not touching other letters
                    (regex (?<![A-Za-z])[A-Z]{2,}(?![A-Za-z]); DNA, "DNA's" and CO2 match, mRNA and iPhone do not)
    long_sentence   word count (whitespace split) strictly above the 90th percentile of the TRAIN split's word counts
    short_sentence  word count strictly below the 10th percentile of the TRAIN split's word counts
Percentiles use numpy's default linear interpolation and are computed on train clips only. Flags are written as
True/False; rows are sorted by clip_id, so the output is deterministic. Sentences are read one row at a time and only
their word count and the two regex flags are kept; no sentence text is printed or written.

    python data/make_edge_flags.py
Options: --videos <csv> (default local/videos.csv), --manifests <dir> (default manifests), --out <csv>.
"""
import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SPLITS = ("train", "val", "test")
FIELDS = ["clip_id", "article_id", "signer_id", "split", "has_digit", "has_acronym", "long_sentence", "short_sentence"]
DIGIT = re.compile(r"\d")
ACRONYM = re.compile(r"(?<![A-Za-z])[A-Z]{2,}(?![A-Za-z])")
LONG_PCT, SHORT_PCT = 90, 10


def read_manifests(manifest_dir):
    rows = []
    for split in SPLITS:
        p = Path(manifest_dir) / f"{split}.csv"
        if not p.exists():
            sys.exit(f"{p} missing - run python data/make_manifests.py")
        with open(p, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.append((r["clip_id"], r["article_id"], r["signer_id"], r["split"]))
    ids = [r[0] for r in rows]
    if len(set(ids)) != len(ids):
        sys.exit("clip ids repeat across the manifests - not guessing, stopping.")
    return rows


def sentence_features(videos_csv, wanted):
    """{clip_id: (n_words, has_digit, has_acronym)} for the wanted clips; the text itself is never stored."""
    feats = {}
    with open(videos_csv, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
        try:
            i_name, i_text = header.index("video filename"), header.index("sentence text")
        except ValueError as e:
            sys.exit(f"{videos_csv}: missing required column ({e}). Columns found: {header}")
        for rec in reader:
            name = rec[i_name].strip()
            if not name.endswith(".mp4"):
                continue
            cid = name[:-4]
            if cid not in wanted:
                continue
            if cid in feats:
                sys.exit(f"{videos_csv}: clip {cid} appears more than once - not guessing, stopping.")
            text = rec[i_text]
            feats[cid] = (len(text.split()), bool(DIGIT.search(text)), bool(ACRONYM.search(text)))
    return feats


def build_flags(manifest_rows, feats):
    train_len = [feats[cid][0] for cid, _, _, split in manifest_rows if split == "train"]
    if not train_len:
        sys.exit("no train clips - cannot compute percentiles")
    hi = float(np.percentile(train_len, LONG_PCT))
    lo = float(np.percentile(train_len, SHORT_PCT))
    out = []
    for cid, aid, sid, split in sorted(manifest_rows, key=lambda r: r[0]):
        n, digit, acro = feats[cid]
        out.append((cid, aid, sid, split, digit, acro, n > hi, n < lo))
    return out, hi, lo


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", type=Path, default=ROOT / "local" / "videos.csv")
    ap.add_argument("--manifests", type=Path, default=ROOT / "manifests")
    ap.add_argument("--out", type=Path, default=None, help="default <manifests>/edge_text_flags.csv")
    args = ap.parse_args(argv)
    out_path = args.out or args.manifests / "edge_text_flags.csv"
    if not args.videos.exists():
        sys.exit(f"{args.videos} not found. Copy the dataset's videos.csv to local/videos.csv (local/ is git-ignored).")

    manifest_rows = read_manifests(args.manifests)
    feats = sentence_features(args.videos, {r[0] for r in manifest_rows})
    missing = [r[0] for r in manifest_rows if r[0] not in feats]
    if missing:
        sys.exit(f"{len(missing)} manifest clip(s) have no row in {args.videos.name}, e.g. {missing[:3]} - not guessing, stopping.")

    rows, hi, lo = build_flags(manifest_rows, feats)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(FIELDS)
        w.writerows(rows)
    tmp.replace(out_path)

    print(f"train word-count percentiles: p{LONG_PCT}={hi:g} (long_sentence: > this), p{SHORT_PCT}={lo:g} (short_sentence: < this)")
    print(f"wrote {len(rows)} rows to {out_path}")
    for name, col in (("has_digit", 4), ("has_acronym", 5), ("long_sentence", 6), ("short_sentence", 7)):
        print(f"  {name:15s} {sum(r[col] for r in rows):6d} ({100 * sum(r[col] for r in rows) / len(rows):.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
