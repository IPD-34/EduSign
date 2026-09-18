"""
scripts/emergency_cleanup_non_priority_videos.py

EMERGENCY disk-space recovery: C: drive hit 100% full. Extraction alone
can't reclaim space fast enough (backlog: 54,708 videos with no landmark
yet, extraction only processes ~500-900/check-in). This deletes the
SAFE subset instead: downloaded videos that (a) have NO landmark yet
(so cleanup_asl_citizen_extracted_videos.py wouldn't touch them) AND
(b) are NOT for one of the 1,158 WLASL-overlap words currently being
prioritized. These were downloaded for the broader full-corpus goal,
which is not urgently blocking anything right now -- deleting them
loses nothing permanent, they're re-downloadable from the same
Microsoft source whenever there's real disk headroom again (e.g. after
redirecting future downloads to D:, which has real free space).

Does NOT touch: anything with a landmark already (that's what the
normal cleanup script handles, unaffected by this), or anything in the
1,158-word priority set (still needed for the pending WLASL check).
"""
import sys
import csv
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

LOCAL_DIR = Path(
    r"C:\Users\KRUPAS~1\AppData\Local\Temp\claude\D--College-Projects---Resume-bidirectional-sign-language"
    r"\1f4ac0b6-58be-4111-abc9-012b28f39d49\scratchpad\asl_citizen_scaling"
)
SPLITS_DIR = Path(
    r"C:\Users\KRUPAS~1\AppData\Local\Temp\claude\D--College-Projects---Resume-bidirectional-sign-language"
    r"\1f4ac0b6-58be-4111-abc9-012b28f39d49\scratchpad\asl_citizen\ASL_Citizen\splits"
)
PROCESSED_DIR = Path("data/processed/asl_citizen_scaling")


def load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    with open("logs/_tmp_citizen_floor_wlasl_overlap.json") as f:
        priority_words = set(json.load(f))

    freed_bytes = 0
    deleted = 0
    kept_priority = 0
    kept_has_landmark = 0
    skipped_locked = 0

    for split in ("train", "val", "test"):
        for r in load_csv(SPLITS_DIR / f"{split}.csv"):
            gloss = r["Gloss"].strip().lower()
            fname = r["Video file"]
            video_path = LOCAL_DIR / "videos" / split / fname
            if not video_path.exists():
                continue
            video_id = f"aslciti_{Path(fname).stem}"
            npz_path = PROCESSED_DIR / split / f"{video_id}.npz"
            if npz_path.exists():
                kept_has_landmark += 1
                continue
            if gloss in priority_words:
                kept_priority += 1
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
          f"{'would be ' if args.dry_run else ''}freed")
    print(f"Kept (already has landmark): {kept_has_landmark}")
    print(f"Kept (needed for 1158-word priority set): {kept_priority}")
    print(f"Skipped (locked right now): {skipped_locked}")


if __name__ == "__main__":
    main()
