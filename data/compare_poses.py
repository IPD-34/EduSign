"""Compare two folders of pose .pkl files (outputs of data/extract_rtmw.py), clip by clip.

    python data/compare_poses.py --a local/poses_gpu --b local/poses_cpu [--tol 1e-3]

For every clip whose <clip_id>.pkl is in both folders it prints max |a - b| over all frames and points for keypoints and
for scores, and flags clips above --tol (default 1e-3). Clips present in only one folder are listed. Notes (never
failures): differing n_frames or array dtype. A clip that cannot be compared (unloadable file, different number of frames,
different shapes, NaN differences) counts as exceeding the tolerance.
Exit code: 1 if any compared clip exceeds --tol, 2 if the folders have no clip in common (nothing was compared), else 0.
Clips present in only one folder do not change the exit code. Note: .pkl files are unpickled; use folders you made yourself.
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np


def load_pose(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _stack(frames):
    return np.stack([np.asarray(f) for f in frames]).astype(np.float64)


def compare_pose_files(pa, pb):
    """Return (max |d keypoints|, max |d scores|, notes). Raises ValueError if the two files cannot be compared."""
    a, b = load_pose(pa), load_pose(pb)
    for name, d in (("a", a), ("b", b)):
        if not isinstance(d, dict) or "keypoints" not in d or "scores" not in d:
            raise ValueError(f"{name}: not a pose dict with keypoints and scores")
    notes = []
    for key in ("keypoints", "scores"):
        if len(a[key]) != len(b[key]):
            raise ValueError(f"{key}: {len(a[key])} frames in a vs {len(b[key])} in b")
    if a.get("n_frames") != b.get("n_frames"):
        notes.append(f"n_frames {a.get('n_frames')} vs {b.get('n_frames')}")
    out = []
    for key in ("keypoints", "scores"):
        if not len(a[key]):
            raise ValueError(f"{key}: no frames")
        da, db = np.asarray(a[key][0]).dtype, np.asarray(b[key][0]).dtype
        if da != db:
            notes.append(f"{key} dtype {da} vs {db}")
        xa, xb = _stack(a[key]), _stack(b[key])
        if xa.shape != xb.shape:
            raise ValueError(f"{key}: shape {xa.shape} in a vs {xb.shape} in b")
        out.append(float(np.max(np.abs(xa - xb))))   # NaN propagates, and then fails the <= tol test
    return out[0], out[1], notes


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, type=Path, help="first folder of <clip_id>.pkl")
    ap.add_argument("--b", required=True, type=Path, help="second folder of <clip_id>.pkl")
    ap.add_argument("--tol", type=float, default=1e-3, help="max allowed absolute difference (default 1e-3)")
    args = ap.parse_args(argv)
    for p in (args.a, args.b):
        if not p.is_dir():
            sys.exit(f"{p}: not a folder")

    fa = {p.stem: p for p in sorted(args.a.glob("*.pkl"))}
    fb = {p.stem: p for p in sorted(args.b.glob("*.pkl"))}
    common = sorted(set(fa) & set(fb))
    only_a, only_b = sorted(set(fa) - set(fb)), sorted(set(fb) - set(fa))

    exceeding, worst = [], 0.0
    for cid in common:
        try:
            dk, ds, notes = compare_pose_files(fa[cid], fb[cid])
        except Exception as e:
            print(f"{cid}  CANNOT COMPARE: {type(e).__name__}: {e}")
            exceeding.append(cid)
            continue
        ok = dk <= args.tol and ds <= args.tol
        worst = max(worst, dk, ds) if not (np.isnan(dk) or np.isnan(ds)) else float("nan")
        print(f"{cid}  max|d keypoints|={dk:.3e}  max|d scores|={ds:.3e}  {'OK' if ok else 'EXCEEDS tol'}"
              + (f"  [{'; '.join(notes)}]" if notes else ""))
        if not ok:
            exceeding.append(cid)

    for title, ids in (("only in a", only_a), ("only in b", only_b)):
        if ids:
            print(f"{title} ({len(ids)}):")
            for cid in ids:
                print(f"  {cid}")
    print(f"compared={len(common)} exceeding_tol={len(exceeding)} only_in_a={len(only_a)} only_in_b={len(only_b)} "
          f"tol={args.tol:g}")
    if not common:
        print("nothing compared: the folders have no clip in common")
        return 2
    return 1 if exceeding else 0


if __name__ == "__main__":
    sys.exit(main())
