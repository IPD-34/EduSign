"""
recompute_asllrp_cap.py

Day 22 follow-up: recomputes the ASLLRP max-sequence-length cap-hit rate
from the EXISTING manifest.csv (no re-extraction needed).

Why this is safe to do without rerunning extraction: sampled_num_frames
was always computed correctly in every .npz file and every manifest
row. The bug was only in the comparison used to set hit_max_cap --
dataset_processor.py was checking sampled_num_frames against the
shared/WLASL MAX_SEQUENCE_LENGTH (150) for every source, including
ASLLRP, which has its own, separately-measured cap
(ASLLRP_MAX_SEQUENCE_LENGTH = 300). This script re-derives the correct
rate directly from sampled_num_frames, which needs no correction.

Run this BEFORE trusting the "8.5% at 300" figure currently sitting
in config.py's comment -- this confirms whether that number is real
or needs updating.

Usage:
    uv run python recompute_asllrp_cap.py
"""

import csv
from pathlib import Path

from src.preprocessing.config import ASLLRP_MAX_SEQUENCE_LENGTH, ASLLRP_PROCESSED_DIR

MANIFEST_PATH = ASLLRP_PROCESSED_DIR / "manifest.csv"


def main():
    if not MANIFEST_PATH.exists():
        print(f"No manifest found at {MANIFEST_PATH}")
        return

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    successful_rows = [r for r in rows if r.get("status") == "success"]
    total = len(successful_rows)

    if total == 0:
        print("No successful rows found in manifest -- nothing to compute.")
        return

    frame_counts = []
    capped = 0

    for r in successful_rows:
        try:
            n = int(r["sampled_num_frames"])
        except (TypeError, ValueError, KeyError):
            continue
        frame_counts.append(n)
        if n >= ASLLRP_MAX_SEQUENCE_LENGTH:
            capped += 1

    n_counted = len(frame_counts)

    print("==========================")
    print("ASLLRP Cap-Hit Rate (Recomputed, Correct Cap)")
    print("==========================")
    print(f"Manifest       : {MANIFEST_PATH}")
    print(f"Total success  : {total}")
    print(f"Usable rows    : {n_counted}  (rows with a valid sampled_num_frames)")
    print(f"Correct cap    : {ASLLRP_MAX_SEQUENCE_LENGTH}")
    print()

    if n_counted:
        print(f"Cap-hit rate (correct, cap={ASLLRP_MAX_SEQUENCE_LENGTH}): "
              f"{capped}/{n_counted} ({100*capped/n_counted:.1f}%)")

        # Also show what the (buggy) old shared cap of 150 would have reported,
        # for direct comparison against the original ~60% smoke-test figure.
        old_cap = 150
        old_capped = sum(1 for n in frame_counts if n >= old_cap)
        print(f"Cap-hit rate (old/buggy, cap={old_cap}):     "
              f"{old_capped}/{n_counted} ({100*old_capped/n_counted:.1f}%)")

        print()
        print(f"Frame count stats: min={min(frame_counts)}, max={max(frame_counts)}, "
              f"mean={sum(frame_counts)/n_counted:.1f}")
    else:
        print("No rows had a usable sampled_num_frames value.")


if __name__ == "__main__":
    main()