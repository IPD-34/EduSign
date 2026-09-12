#!/usr/bin/env python3
"""
Match WLASL dataset videos to the 48 team glosses (Persons A-D) and copy
matched clips into per-person output folders + write a master mapping CSV.

Run from your project root:
    python match_wlasl_glosses.py

Adjust the paths below if your layout differs.
"""

import json
import os
import csv
import glob
import shutil

# ---- paths (adjust if needed) ----
WLASL_ROOT = "data/raw/wlasl"
METADATA_DIR = os.path.join(WLASL_ROOT, "metadata")
VIDEOS_DIR = os.path.join(WLASL_ROOT, "videos")
OUTPUT_DIR = os.path.join(WLASL_ROOT, "matched_by_person")
MASTER_CSV = os.path.join(WLASL_ROOT, "wlasl_gloss_video_mapping.csv")

# ---- the 48 team glosses, split by person ----
TEAM_GLOSSES = {
    "A": ["able", "about", "above", "accept", "act", "action", "active",
          "activity", "actor", "add", "address", "adult"],
    "B": ["afraid", "after", "afternoon", "again", "against", "age", "ago",
          "agree", "ahead", "alarm", "alone", "already"],
    "C": ["also", "always", "amazing", "angry", "animal", "another",
          "answer", "any", "apple", "appointment", "area", "arm"],
    "D": ["army", "around", "arrive", "art", "artist", "ask", "attention",
          "aunt", "awake", "away", "baby", "back"],
}

VIDEO_EXTS = [".mp4", ".mkv", ".swf", ".webm", ".avi"]


def find_wlasl_json(metadata_dir):
    """Locate the WLASL instance metadata json (usually WLASL_v0.3.json)."""
    candidates = glob.glob(os.path.join(metadata_dir, "*.json"))
    if not candidates:
        raise FileNotFoundError(
            f"No .json files found in {metadata_dir}. "
            "Point METADATA_DIR / the filename manually."
        )
    # Prefer a file that actually looks like the WLASL gloss->instances schema
    for path in candidates:
        try:
            with open(path, "r") as f:
                data = json.load(f)
            if isinstance(data, list) and data and "gloss" in data[0] and "instances" in data[0]:
                print(f"Using metadata file: {path}")
                return path
        except Exception:
            continue
    # fall back to first json found
    print(f"Couldn't confirm schema, defaulting to: {candidates[0]}")
    return candidates[0]


def find_video_file(video_id):
    """Search VIDEOS_DIR (recursively) for a file matching video_id."""
    for ext in VIDEO_EXTS:
        # direct match
        direct = os.path.join(VIDEOS_DIR, f"{video_id}{ext}")
        if os.path.exists(direct):
            return direct
    # recursive search in case videos are nested in subfolders
    for ext in VIDEO_EXTS:
        matches = glob.glob(os.path.join(VIDEOS_DIR, "**", f"{video_id}{ext}"), recursive=True)
        if matches:
            return matches[0]
    return None


def main():
    json_path = find_wlasl_json(METADATA_DIR)
    with open(json_path, "r") as f:
        wlasl_data = json.load(f)

    # reverse lookup: gloss -> person
    gloss_to_person = {}
    for person, glosses in TEAM_GLOSSES.items():
        for g in glosses:
            gloss_to_person[g.lower()] = person

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for person in TEAM_GLOSSES:
        os.makedirs(os.path.join(OUTPUT_DIR, person), exist_ok=True)

    rows = []
    found_glosses = set()

    for entry in wlasl_data:
        gloss = entry.get("gloss", "").lower()
        if gloss not in gloss_to_person:
            continue
        person = gloss_to_person[gloss]
        for inst in entry.get("instances", []):
            video_id = inst.get("video_id")
            if not video_id:
                continue
            src = find_video_file(video_id)
            if src is None:
                continue
            found_glosses.add(gloss)
            ext = os.path.splitext(src)[1]
            dst_name = f"{person}_{gloss}_{video_id}{ext}"
            dst = os.path.join(OUTPUT_DIR, person, dst_name)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
            rows.append({
                "person": person,
                "gloss": gloss,
                "video_id": video_id,
                "source_path": src,
                "dest_path": dst,
                "split": inst.get("split", ""),
                "signer_id": inst.get("signer_id", ""),
            })

    with open(MASTER_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "person", "gloss", "video_id", "source_path", "dest_path", "split", "signer_id"
        ])
        writer.writeheader()
        writer.writerows(rows)

    all_target_glosses = {g.lower() for glosses in TEAM_GLOSSES.values() for g in glosses}
    missing = sorted(all_target_glosses - found_glosses)

    print(f"\nMatched {len(rows)} video instances across {len(found_glosses)}/48 glosses.")
    print(f"Mapping CSV written to: {MASTER_CSV}")
    print(f"Videos copied under: {OUTPUT_DIR}/<person>/")
    if missing:
        print(f"\nNo WLASL video found for {len(missing)} glosses (may not exist in WLASL, or filenames didn't match):")
        for g in missing:
            print(f"  - {g}")


if __name__ == "__main__":
    main()