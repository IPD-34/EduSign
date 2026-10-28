"""Draw extracted keypoints back onto 3 evenly spaced kept frames of a clip, to eyeball that poses line up with the video.

    python data/overlay_check.py --clip local/clips/<clip_id>.mp4 --pkl local/poses/<clip_id>.pkl [--out local/overlays]

The .pkl (from data/extract_rtmw.py) holds the KEPT frames only; the original frame behind kept frame j is
keep_indices(n_frames)[j]. Three kept frames are used: the first, the middle and the last (fewer for 1-2 kept frames).
Keypoints are x*width, y*height (the pkl stores them divided by frame width/height). Points with score < 0.3 are drawn
hollow; an all-zero frame (no person) draws nothing. Colours (BGR): body green, feet yellow, face grey, left hand blue,
right hand red. The caption says only the kept/original frame numbers - no sentence text is read or drawn.
Output: <out>/<clip_id>_f<original frame>.png. The frames are licensed video data, so --out must be git-ignored
(default local/overlays); any other path is refused before anything is read.
"""
import argparse
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_rtmw import keep_indices  # noqa: E402
from fetch_references import is_gitignored  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LOW_CONF = 0.3
GROUPS = (  # (first index, last index + 1, BGR colour) in the COCO-WholeBody 133 layout
    (0, 17, (0, 200, 0)), (17, 23, (0, 220, 220)), (23, 91, (170, 170, 170)),
    (91, 112, (255, 80, 0)), (112, 133, (0, 0, 255)))


def pick_kept(n_kept):
    """Kept-frame positions to draw: first, middle, last (unique)."""
    return sorted({0, (n_kept - 1) // 2, n_kept - 1})


def draw_pose(img, kp, sc, caption):
    """Draw one frame's (133,2) normalised keypoints and (133,) scores on a copy of the BGR image."""
    out = img.copy()
    h, w = out.shape[:2]
    if kp.any():
        for lo, hi, col in GROUPS:
            for i in range(lo, hi):
                c = (int(round(kp[i, 0] * w)), int(round(kp[i, 1] * h)))
                if sc[i] >= LOW_CONF:
                    cv2.circle(out, c, 3, col, -1, cv2.LINE_AA)
                else:
                    cv2.circle(out, c, 3, col, 1, cv2.LINE_AA)
    cv2.putText(out, caption, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, caption, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def read_frames(clip, wanted):
    """Decode through the clip once; return ({original index: BGR frame} for `wanted`, total frame count)."""
    cap = cv2.VideoCapture(str(clip))
    if not cap.isOpened():
        sys.exit(f"cannot open {clip}")
    wanted, got, i = set(wanted), {}, 0
    try:
        while cap.grab():
            if i in wanted:
                ok, frame = cap.retrieve()
                if not ok:
                    sys.exit(f"cannot decode frame {i} of {clip}")
                got[i] = frame
            i += 1
    finally:
        cap.release()
    return got, i


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", required=True, type=Path, help="the clip .mp4")
    ap.add_argument("--pkl", required=True, type=Path, help="its pose .pkl from extract_rtmw.py")
    ap.add_argument("--out", type=Path, default=ROOT / "local" / "overlays",
                    help="folder for the PNGs; must be git-ignored (default local/overlays)")
    args = ap.parse_args(argv)

    if not is_gitignored(args.out / "probe.png"):
        sys.exit(f"refusing to write to {args.out}: git does not report it as ignored, and the overlays contain "
                 f"licensed video frames. Use a path under local/ (see .gitignore).")
    with open(args.pkl, "rb") as f:
        d = pickle.load(f)
    for key in ("keypoints", "scores", "n_frames"):
        if key not in d:
            sys.exit(f"{args.pkl}: missing key {key!r} (pose files made before the 256-frame cap lack n_frames; re-extract)")
    kp, sc, n = d["keypoints"], d["scores"], int(d["n_frames"])
    kept = keep_indices(n)
    if not (len(kp) == len(sc) == len(kept)):
        sys.exit(f"{args.pkl}: {len(kp)} keypoint frames / {len(sc)} score frames, but n_frames={n} keeps {len(kept)}")

    picks = pick_kept(len(kp))
    frames, total = read_frames(args.clip, [kept[j] for j in picks])
    if total != n:
        sys.exit(f"{args.clip} has {total} frames but {args.pkl} says n_frames={n}: wrong clip/pose pairing?")

    args.out.mkdir(parents=True, exist_ok=True)
    clip_id = args.clip.stem
    for j in picks:
        orig = kept[j]
        img = draw_pose(frames[orig], np.asarray(kp[j])[0], np.asarray(sc[j])[0], f"kept {j + 1}/{len(kp)}  frame {orig}")
        ok, buf = cv2.imencode(".png", img)
        if not ok:
            sys.exit(f"cannot encode PNG for frame {orig}")
        dest = args.out / f"{clip_id}_f{orig:05d}.png"
        dest.write_bytes(buf.tobytes())
        print(f"kept frame {j} -> original frame {orig} -> {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
