"""
channel_check_extract.py

Day 16: extract landmarks from real, self-recorded channel-check footage
(data/raw/channel_check/*.mp4) using the existing, already-validated
SequenceBuilder pipeline -- same extraction path as WLASL/ASLLRP, no
parallel extraction logic written for this one-off diagnostic dataset.

Produces ONE full-channel (pose+face+hands) npz per clip -- Config A
(hands+face only) is derived from this at scoring time by zeroing the
pose channel, rather than extracting twice. This keeps extraction
simple and avoids a second source of bugs right when a clean,
trustworthy comparison is needed (per the Week 3 plan's own logic).

Expects a manifest CSV at data/raw/channel_check/manifest.csv with
columns: filename,gloss
e.g.
    C_also_1.mp4,also
    C_also_2.mp4,also
    A_able_1.mp4,able

Usage:
    uv run python scripts/channel_check_extract.py
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from src.preprocessing.sequence_builder import SequenceBuilder
from src.preprocessing.sample_writer import write_sample

CHANNEL_CHECK_RAW_DIR = RAW_DATA_DIR / "channel_check"
CHANNEL_CHECK_PROCESSED_DIR = PROCESSED_DATA_DIR / "channel_check"
MANIFEST_PATH = CHANNEL_CHECK_RAW_DIR / "manifest.csv"


def main():
    if not MANIFEST_PATH.exists():
        print(f"No manifest found at {MANIFEST_PATH}")
        print("Expected columns: filename,gloss -- create this before running extraction.")
        return

    CHANNEL_CHECK_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"Manifest entries: {len(rows)}")

    builder = SequenceBuilder()
    results = []
    successful = 0
    failed = 0

    for row in rows:
        filename = row["filename"]
        gloss = row["gloss"]
        video_path = CHANNEL_CHECK_RAW_DIR / filename
        video_id = Path(filename).stem  # e.g. "C_also_1"

        if not video_path.exists():
            print(f"[MISSING] {filename} listed in manifest but not found on disk -- skipping")
            failed += 1
            continue

        try:
            # Full holistic extraction -- same as WLASL/ASLLRP. Config A is
            # derived at scoring time by masking the pose channel, not by
            # extracting a second time.
            processed = builder.build(
                video_path=str(video_path),
                video_id=video_id,
                gloss=gloss,
                split="channel_check",
                source="channel_check",
                frame_start=None,
                frame_end=None,
                translation=None,
            )

            output_path = CHANNEL_CHECK_PROCESSED_DIR / f"{video_id}.npz"
            write_sample(processed, output_path)

            results.append({
                "video_id": video_id,
                "gloss": gloss,
                "num_frames": processed.metadata.num_frames,
                "status": "success",
            })
            successful += 1
            print(f"[OK] {video_id} ({gloss}) -- {processed.metadata.num_frames} frames")

        except Exception as e:
            results.append({
                "video_id": video_id,
                "gloss": gloss,
                "num_frames": None,
                "status": f"failed: {e}",
            })
            failed += 1
            print(f"[FAILED] {video_id}: {e}")

    # Write a processed-side manifest too, so the scoring script has a
    # single source of truth for video_id -> ground-truth gloss.
    out_manifest_path = CHANNEL_CHECK_PROCESSED_DIR / "manifest.csv"
    with open(out_manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["video_id", "gloss", "num_frames", "status"])
        writer.writeheader()
        writer.writerows(results)

    print("\n==========================")
    print("Channel-Check Extraction Complete")
    print("==========================")
    print(f"Successful: {successful}, Failed: {failed}")
    print(f"Processed manifest written to {out_manifest_path}")


if __name__ == "__main__":
    main()