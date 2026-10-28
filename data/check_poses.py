"""Check a folder of extracted pose .pkl files against a manifest (output of data/extract_rtmw.py).

    python data/check_poses.py --manifest manifests/eval300.csv --poses local/poses

Reports manifest clip_ids with no <clip_id>.pkl (missing) and .pkl files whose clip_id is not in the manifest (extra),
then checks EVERY .pkl in the folder: a dict with keypoints, scores and n_frames; keypoints and scores are lists of
equal length, at most 256 frames (and not empty); every keypoints frame has shape (1, 133, 2), every scores frame has
shape (1, 133); n_frames is a positive int. Prints expected / found / missing / bad (plus extra) counts and exits 1 if
any clip is missing or any .pkl is bad. Extra files alone do not fail the check.
Note: .pkl files are unpickled, so only point this at folders you created yourself.
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_rtmw import MAX_FRAMES, NUM_KPTS, read_clip_ids  # noqa: E402

KP_SHAPE = (1, NUM_KPTS, 2)
SC_SHAPE = (1, NUM_KPTS)
SHOW = 20   # how many ids / problems to list per section


def check_pose_file(path):
    """Return a list of problems for one .pkl (empty list = OK)."""
    try:
        with open(path, "rb") as f:
            d = pickle.load(f)
    except Exception as e:
        return [f"cannot load: {type(e).__name__}: {e}"]
    if not isinstance(d, dict):
        return [f"not a dict (got {type(d).__name__})"]
    problems = []
    missing = [k for k in ("keypoints", "scores", "n_frames") if k not in d]
    if missing:
        problems.append(f"missing key(s): {missing}")
    kp, sc = d.get("keypoints"), d.get("scores")
    if not isinstance(kp, (list, tuple)) or not isinstance(sc, (list, tuple)):
        problems.append("keypoints and scores must both be lists of per-frame arrays")
    else:
        if len(kp) != len(sc):
            problems.append(f"length mismatch: {len(kp)} keypoints vs {len(sc)} scores")
        if len(kp) == 0:
            problems.append("no frames")
        if len(kp) > MAX_FRAMES or len(sc) > MAX_FRAMES:
            problems.append(f"more than {MAX_FRAMES} frames ({len(kp)} keypoints, {len(sc)} scores)")
        for name, frames, want in (("keypoints", kp, KP_SHAPE), ("scores", sc, SC_SHAPE)):
            bad = [i for i, a in enumerate(frames) if getattr(a, "shape", None) != want]
            if bad:
                problems.append(f"{name}: {len(bad)} frame(s) with shape != {want}, first at frame {bad[0]} "
                                f"(shape {getattr(frames[bad[0]], 'shape', type(frames[bad[0]]).__name__)})")
    if "n_frames" in d:
        n = d["n_frames"]
        if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n <= 0:
            problems.append(f"n_frames must be a positive int, got {n!r}")
    return problems


def _show(title, items):
    print(f"{title} ({len(items)}):")
    for it in items[:SHOW]:
        print(f"  {it}")
    if len(items) > SHOW:
        print(f"  ... and {len(items) - SHOW} more")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, type=Path, help="manifest CSV with a clip_id column")
    ap.add_argument("--poses", required=True, type=Path, help="folder holding <clip_id>.pkl")
    args = ap.parse_args(argv)
    if not args.poses.is_dir():
        sys.exit(f"{args.poses}: not a folder")

    expected = list(dict.fromkeys(read_clip_ids(args.manifest)))
    on_disk = {p.stem: p for p in sorted(args.poses.glob("*.pkl"))}
    want = set(expected)
    missing = [c for c in expected if c not in on_disk]
    extra = sorted(c for c in on_disk if c not in want)
    bad = {}
    for cid, p in on_disk.items():
        problems = check_pose_file(p)
        if problems:
            bad[cid] = problems

    if missing:
        _show("MISSING (in manifest, no .pkl)", missing)
    if extra:
        _show("EXTRA (.pkl not in manifest)", extra)
    if bad:
        _show("BAD", [f"{cid}: {'; '.join(pr)}" for cid, pr in bad.items()])
    found = len(want & set(on_disk))
    print(f"expected={len(expected)} found={found} missing={len(missing)} bad={len(bad)} extra={len(extra)}")
    return 1 if (missing or bad) else 0


if __name__ == "__main__":
    sys.exit(main())
