#!/usr/bin/env python3
"""
Build the master manifest for the channel-check recording batch.

Scans a folder of clips named <person>_<gloss>_<take#>.mp4, extracts video
metadata, cross-checks against the expected 48-gloss / 3-take assignment,
and writes a single manifest CSV.

Run:
    uv run python build_manifest.py
    uv run python build_manifest.py --dir data/raw/channel_check

Requires opencv for metadata (duration/fps/resolution). If not installed:
    uv add opencv-python-headless
"""

import os
import csv
import argparse
from datetime import datetime

try:
    import cv2
    HAVE_CV2 = True
except ImportError:
    HAVE_CV2 = False

# ---- expected assignment (person -> glosses), for completeness checking ----
# Real names mapped from the original A/B/C/D split based on gloss overlap.
EXPECTED_GLOSSES = {
    "atharv": ["able", "about", "above", "accept", "act", "action", "active",
               "activity", "actor", "add", "address", "adult"],
    "sarthak": ["afraid", "after", "afternoon", "again", "against", "age", "ago",
                "agree", "ahead", "alarm", "alone", "already"],
    "krupa": ["also", "always", "amazing", "angry", "animal", "another",
              "answer", "any", "apple", "appointment", "area", "arm"],
    "manasi": ["army", "around", "arrive", "art", "artist", "ask", "attention",
               "aunt", "awake", "away", "baby", "back"],
}
EXPECTED_TAKES = 3
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi"}


def get_video_metadata(path):
    """Return (duration_sec, width, height, fps) or (None, None, None, None) if unreadable."""
    if not HAVE_CV2:
        return None, None, None, None
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None, None, None, None
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = frame_count / fps if fps > 0 else None
    cap.release()
    return duration, width, height, fps


def parse_filename(filename):
    """
    Parse <person>_<gloss>_<take>.ext into (person, gloss, take).
    Assumes gloss has no underscores (true for this vocab set).
    Returns None if it doesn't match the expected pattern.
    """
    stem, ext = os.path.splitext(filename)
    if ext.lower() not in VIDEO_EXTS:
        return None
    parts = stem.split("_")
    if len(parts) < 3:
        return None
    person = parts[0].lower()
    take_str = parts[-1]
    gloss = "_".join(parts[1:-1]).lower()
    if not take_str.isdigit():
        return None
    return person, gloss, int(take_str)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/raw/channel_check",
                     help="Folder containing the recorded clips")
    ap.add_argument("--out", default=None,
                     help="Output CSV path (default: <dir>/manifest.csv)")
    args = ap.parse_args()

    scan_dir = args.dir
    out_path = args.out or os.path.join(scan_dir, "manifest.csv")

    if not HAVE_CV2:
        print("WARNING: opencv not installed — duration/fps/resolution will be blank.")
        print("Run: uv add opencv-python-headless\n")

    rows = []
    unparsed = []
    seen = {}  # (person, gloss) -> set of takes found

    for filename in sorted(os.listdir(scan_dir)):
        full_path = os.path.join(scan_dir, filename)
        if not os.path.isfile(full_path):
            continue
        parsed = parse_filename(filename)
        if parsed is None:
            ext = os.path.splitext(filename)[1].lower()
            if ext in VIDEO_EXTS:
                unparsed.append(filename)
            continue

        person, gloss, take = parsed
        duration, width, height, fps = get_video_metadata(full_path)
        size_bytes = os.path.getsize(full_path)
        mtime = datetime.fromtimestamp(os.path.getmtime(full_path)).isoformat(timespec="seconds")

        rows.append({
            "filename": filename,
            "person": person,
            "gloss": gloss,
            "take": take,
            "filepath": full_path,
            "duration_sec": round(duration, 2) if duration else "",
            "width": width or "",
            "height": height or "",
            "fps": round(fps, 2) if fps else "",
            "size_bytes": size_bytes,
            "recorded_at": mtime,
        })
        seen.setdefault((person, gloss), set()).add(take)

    # write manifest
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "filename", "person", "gloss", "take", "filepath",
            "duration_sec", "width", "height", "fps", "size_bytes", "recorded_at"
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Manifest written: {out_path}")
    print(f"Total clips parsed: {len(rows)}")
    if unparsed:
        print(f"\n{len(unparsed)} video files did NOT match naming convention (skipped):")
        for u in unparsed:
            print(f"  - {u}")

    # completeness check
    print("\n--- Completeness check ---")
    total_missing = 0
    unexpected = []
    for person, glosses in EXPECTED_GLOSSES.items():
        for gloss in glosses:
            takes = seen.get((person, gloss), set())
            missing_takes = set(range(1, EXPECTED_TAKES + 1)) - takes
            if missing_takes:
                total_missing += len(missing_takes)
                print(f"  MISSING: {person}_{gloss} — take(s) {sorted(missing_takes)}")
        # flag anything recorded under this person not in their assigned list
        person_glosses_found = {g for (p, g) in seen if p == person}
        extra = person_glosses_found - set(glosses)
        if extra:
            unexpected.extend((person, g) for g in extra)

    if total_missing == 0:
        print("  All expected clips present (48 glosses x 3 takes x 4 people).")

    if unexpected:
        print(f"\n{len(unexpected)} clip(s) recorded under a gloss not in that person's assignment:")
        for person, gloss in unexpected:
            print(f"  - {person}_{gloss} (check spelling / assignment)")


if __name__ == "__main__":
    main()