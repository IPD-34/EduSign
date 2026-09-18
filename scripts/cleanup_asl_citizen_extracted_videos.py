"""
scripts/cleanup_asl_citizen_extracted_videos.py

Deletes raw source .mp4 videos ONLY once their landmarks are confirmed
successfully extracted (a matching .npz exists on disk for that
video_id/split -- checked directly, not inferred from the manifest
alone, since the manifest could in principle be stale). Real disk math
that makes this necessary, not optional: full ASL Citizen corpus is
~46GB raw video + ~38GB landmarks = ~84GB combined, against 65GB free
on this disk -- keeping both simultaneously does not fit.

SAFE BY CONSTRUCTION: only ever deletes a video after verifying its
.npz twin exists (open() + a cheap read of the metadata member, not
just a file-exists check, so a truncated/corrupt .npz -- e.g. from a
process kill mid-write -- does NOT count as "safe to delete"). Never
touches a video with no matching .npz, regardless of what the manifest
says. Run this periodically while extraction is ongoing, or once at
the end -- resume-safe either way, since it only ever looks at what's
ACTUALLY on disk right now.
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

VIDEOS_DIR = Path(
    r"C:\Users\KRUPAS~1\AppData\Local\Temp\claude\D--College-Projects---Resume-bidirectional-sign-language"
    r"\1f4ac0b6-58be-4111-abc9-012b28f39d49\scratchpad\asl_citizen_scaling\videos"
)
PROCESSED_DIR = Path("data/processed/asl_citizen_scaling")


def npz_is_valid(path):
    """Cheap check: metadata member reads back cleanly. Doesn't touch
    the large landmark arrays -- fast, same pattern as peek_gloss_split
    elsewhere in this project."""
    try:
        with np.load(path, allow_pickle=False) as data:
            json.loads(str(data["metadata"]))
        return True
    except Exception:
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="report what would be deleted, delete nothing")
    args = parser.parse_args()

    freed_bytes = 0
    deleted = 0
    skipped_no_landmark = 0
    skipped_corrupt_landmark = 0
    skipped_locked = 0

    for split in ("train", "val", "test"):
        video_split_dir = VIDEOS_DIR / split
        if not video_split_dir.exists():
            continue
        for video_path in video_split_dir.glob("*.mp4"):
            video_id = f"aslciti_{video_path.stem}"
            npz_path = PROCESSED_DIR / split / f"{video_id}.npz"
            if not npz_path.exists():
                skipped_no_landmark += 1
                continue
            if not npz_is_valid(npz_path):
                skipped_corrupt_landmark += 1
                continue
            size = video_path.stat().st_size
            if not args.dry_run:
                try:
                    video_path.unlink()
                except PermissionError:
                    skipped_locked += 1
                    continue
            freed_bytes += size
            deleted += 1

    freed_gb = freed_bytes / (1024 ** 3)
    print(f"{'Would delete' if args.dry_run else 'Deleted'}: {deleted} videos, {freed_gb:.2f} GB "
          f"{'would be' if args.dry_run else ''} freed")
    print(f"Skipped (no landmark yet): {skipped_no_landmark}")
    print(f"Skipped (landmark exists but unreadable/corrupt -- NOT deleted): {skipped_corrupt_landmark}")
    print(f"Skipped (locked by another process right now -- will retry next run): {skipped_locked}")


if __name__ == "__main__":
    main()
