from pathlib import Path
import csv
import json
import subprocess

from tqdm import tqdm

# -----------------------------
# Paths
# -----------------------------

ROOT = Path(__file__).resolve().parents[1]

JSON_PATH = ROOT / "data" / "raw" / "wlasl" / "metadata" / "WLASL_v0.3.json"

VIDEOS_DIR = ROOT / "data" / "raw" / "wlasl" / "videos"
MANIFEST_DIR = ROOT / "data" / "raw" / "wlasl" / "manifests"

VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

MANIFEST_PATH = MANIFEST_DIR / "download_manifest.csv"


# -----------------------------
# Load metadata
# -----------------------------

with open(JSON_PATH, "r", encoding="utf-8") as f:
    glosses = json.load(f)

instances = []

for gloss in glosses:
    gloss_name = gloss["gloss"]

    for inst in gloss["instances"]:

        inst["gloss"] = gloss_name
        instances.append(inst)

print(f"Total instances : {len(instances)}")


# -----------------------------
# Manifest
# -----------------------------

with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as csvfile:

    writer = csv.writer(csvfile)

    writer.writerow(
        [
            "video_id",
            "gloss",
            "split",
            "source",
            "url",
            "status",
            "error",
            "filename",
        ]
    )

    downloaded = 0
    skipped = 0
    failed = 0

    for inst in tqdm(instances):

        video_id = inst["video_id"]
        gloss = inst["gloss"]
        split = inst["split"]
        source = inst["source"]
        url = inst["url"]

        output_file = VIDEOS_DIR / f"{video_id}.mp4"

        if output_file.exists():

            skipped += 1

            writer.writerow(
                [
                    video_id,
                    gloss,
                    split,
                    source,
                    url,
                    "already_exists",
                    "",
                    output_file.name,
                ]
            )

            continue

        try:

            subprocess.run(
                [
                    "yt-dlp",
                    "-o",
                    str(output_file),
                    url,
                ],
                check=True,
            )

            downloaded += 1

            writer.writerow(
                [
                    video_id,
                    gloss,
                    split,
                    source,
                    url,
                    "success",
                    "",
                    output_file.name,
                ]
            )

        except Exception as e:

            failed += 1

            writer.writerow(
                [
                    video_id,
                    gloss,
                    split,
                    source,
                    url,
                    "failed",
                    str(e),
                    "",
                ]
            )

print()
print("=" * 40)
print(f"Downloaded : {downloaded}")
print(f"Skipped    : {skipped}")
print(f"Failed     : {failed}")
print("=" * 40)