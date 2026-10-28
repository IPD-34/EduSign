"""Per-clip pose/video edge-case flags for the eval300 clips, from the --stats output of data/extract_rtmw.py.

    python data/make_pose_flags.py local/clip_stats.csv [--manifest manifests/eval300.csv] [--out manifests/edge_eval300_pose_flags.csv]

Reads clip_stats.csv (it may hold more clips than eval300, e.g. eval300 + train_1k; only the manifest's clips are kept,
in manifest order) and writes clip_id, article_id, signer_id, split (from the manifest) plus these boolean flags
(True/False), each defined by the named constants below:
    long_clip           n_frames > LONG_CLIP_FRAMES
    short               n_frames < SHORT_FRAMES
    no_person_frames    frames_no_person_frac > NO_PERSON_FRAC
    multi_person        max_persons_detected > MULTI_PERSON
    hand_dropout        hand_low_conf_frac > HAND_DROPOUT
    left_dominant       left_path_len > LEFT_DOMINANT_RATIO * right_path_len
    nonstandard_fps     |fps - most common fps in the whole stats file| > FPS_TOLERANCE
    nonstandard_resolution  (width, height) differs from the most common pair in the whole stats file
                        (never fires on this data: all clips are 640x480)
Prints min / median / 90th percentile / max of every numeric stat over the kept clips, the modes used, and each flag's
rate. No text of any kind is read or written.
"""
import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np

# ---- Thresholds: UNTUNED HEURISTICS. They were guesses made before looking at real clip_stats; check the printed
# ---- distributions and flag rates before relying on any of them.
LONG_CLIP_FRAMES = 842       # about the 90th percentile of eval300 n_frames (original clip length, before the 256-frame cap)
SHORT_FRAMES = 150
NO_PERSON_FRAC = 0.0         # any frame without a detected person
MULTI_PERSON = 1
HAND_DROPOUT = 0.2
LEFT_DOMINANT_RATIO = 1.5
FPS_TOLERANCE = 0.5          # fps in clip_stats.csv jitters around 30 (29.9994, 30.0006, ...), so exact equality would flag most clips

ROOT = Path(__file__).resolve().parent.parent
STAT_COLUMNS = ["n_frames", "fps", "width", "height", "frames_no_person_frac", "max_persons_detected",
                "left_hand_mean_score", "right_hand_mean_score", "hand_low_conf_frac", "left_path_len", "right_path_len"]
FLAGS = ["long_clip", "short", "no_person_frames", "multi_person", "hand_dropout", "left_dominant",
         "nonstandard_fps", "nonstandard_resolution"]
FIELDS = ["clip_id", "article_id", "signer_id", "split"] + FLAGS


def read_stats(path):
    """{clip_id: {stat: float}} for every row of the stats file."""
    with open(path, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        missing = [c for c in ["clip_id"] + STAT_COLUMNS if c not in (rd.fieldnames or [])]
        if missing:
            sys.exit(f"{path}: missing column(s) {missing}; columns found: {rd.fieldnames}")
        stats = {}
        for r in rd:
            if r["clip_id"] in stats:
                sys.exit(f"{path}: clip {r['clip_id']} appears more than once - not guessing, stopping.")
            stats[r["clip_id"]] = {c: float(r[c]) for c in STAT_COLUMNS}
    return stats


def read_manifest(path):
    with open(path, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        need = ["clip_id", "article_id", "signer_id", "split"]
        if any(c not in (rd.fieldnames or []) for c in need):
            sys.exit(f"{path}: needs columns {need}; found {rd.fieldnames}")
        return [(r["clip_id"], r["article_id"], r["signer_id"], r["split"]) for r in rd]


def mode(values):
    """Most common value; ties go to the smaller value, so the result is deterministic."""
    counts = Counter(values)
    return min((v for v in counts if counts[v] == max(counts.values())))


def compute_flags(s, mode_fps, mode_res):
    return {
        "long_clip": s["n_frames"] > LONG_CLIP_FRAMES,
        "short": s["n_frames"] < SHORT_FRAMES,
        "no_person_frames": s["frames_no_person_frac"] > NO_PERSON_FRAC,
        "multi_person": s["max_persons_detected"] > MULTI_PERSON,
        "hand_dropout": s["hand_low_conf_frac"] > HAND_DROPOUT,
        "left_dominant": s["left_path_len"] > LEFT_DOMINANT_RATIO * s["right_path_len"],
        "nonstandard_fps": abs(s["fps"] - mode_fps) > FPS_TOLERANCE,
        # never fires on this data: all clips are 640x480
        "nonstandard_resolution": (s["width"], s["height"]) != mode_res,
    }


def build_rows(stats, manifest):
    """Flag rows for the manifest's clips; the modes use every row of the stats file."""
    missing = [r[0] for r in manifest if r[0] not in stats]
    if missing:
        sys.exit(f"{len(missing)} manifest clip(s) have no row in the stats file, e.g. {missing[:3]} - not guessing, stopping.")
    mode_fps = mode(s["fps"] for s in stats.values())
    mode_res = mode((s["width"], s["height"]) for s in stats.values())
    rows = []
    for cid, aid, sid, split in manifest:
        fl = compute_flags(stats[cid], mode_fps, mode_res)
        rows.append([cid, aid, sid, split] + [fl[f] for f in FLAGS])
    return rows, mode_fps, mode_res


def print_distributions(stats, manifest, mode_fps, mode_res):
    print(f"{'stat':24s} {'min':>10s} {'median':>10s} {'p90':>10s} {'max':>10s}   (over the {len(manifest)} manifest clips)")
    for c in STAT_COLUMNS:
        v = np.array([stats[r[0]][c] for r in manifest], dtype=float)
        print(f"{c:24s} {v.min():10.4g} {np.median(v):10.4g} {np.percentile(v, 90):10.4g} {v.max():10.4g}")
    print(f"most common fps in the file ({len(stats)} rows): {mode_fps:g} (tolerance {FPS_TOLERANCE:g}); "
          f"most common resolution: {int(mode_res[0])}x{int(mode_res[1])}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stats", type=Path, help="clip_stats.csv from extract_rtmw.py --stats")
    ap.add_argument("--manifest", type=Path, default=ROOT / "manifests" / "eval300.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "manifests" / "edge_eval300_pose_flags.csv")
    args = ap.parse_args(argv)
    if not args.stats.exists():
        sys.exit(f"{args.stats} not found")

    stats, manifest = read_stats(args.stats), read_manifest(args.manifest)
    rows, mode_fps, mode_res = build_rows(stats, manifest)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_name(args.out.name + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(FIELDS)
        w.writerows(rows)
    tmp.replace(args.out)

    print_distributions(stats, manifest, mode_fps, mode_res)
    print(f"wrote {len(rows)} rows to {args.out}")
    for i, name in enumerate(FLAGS):
        n = sum(r[4 + i] for r in rows)
        print(f"  {name:24s} {n:4d} ({100 * n / len(rows):5.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
