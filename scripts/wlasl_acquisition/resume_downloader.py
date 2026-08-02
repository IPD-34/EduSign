from pathlib import Path
import csv
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]

CANDIDATES_CSV = ROOT / "data" / "raw" / "wlasl" / "manifests" / "resume_candidates.csv"
VIDEOS_DIR = ROOT / "data" / "raw" / "wlasl" / "videos"
RESUME_MANIFEST = ROOT / "data" / "raw" / "wlasl" / "manifests" / "download_manifest_resume.csv"

VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
RESUME_MANIFEST.parent.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------
# Read resume manifest (already processed in this run)
# ----------------------------------------------------
processed_ids = set()

if RESUME_MANIFEST.exists():
    with open(RESUME_MANIFEST, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            processed_ids.add(str(row["video_id"]))

# ----------------------------------------------------
# Read candidates
# ----------------------------------------------------
with open(CANDIDATES_CSV, newline="", encoding="utf-8") as f:
    candidates = list(csv.DictReader(f))

total = len(candidates)

# ----------------------------------------------------
# Create manifest if needed
# ----------------------------------------------------
if not RESUME_MANIFEST.exists():
    with open(RESUME_MANIFEST, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "video_id",
                "gloss",
                "split",
                "source",
                "url",
                "status",
                "error",
                "filename",
            ],
        )
        writer.writeheader()

# ----------------------------------------------------
# Stats
# ----------------------------------------------------
success = 0
failed = 0
skipped = 0

print("=" * 60)
print(f"Total resume candidates : {total}")
print(f"Already processed       : {len(processed_ids)}")
print("=" * 60)

# ----------------------------------------------------
# Download loop
# ----------------------------------------------------
for index, row in enumerate(candidates, start=1):

    video_id = row["video_id"]

    if video_id in processed_ids:
        skipped += 1
        continue

    outfile = VIDEOS_DIR / f"{video_id}.mp4"

    # Extra safety
    if outfile.exists():
        status = "already_exists"
        error = ""
        filename = outfile.name

        print(f"[{index}/{total}] {video_id:<8} ALREADY EXISTS")

    else:

        print(f"[{index}/{total}] {video_id:<8} Downloading...")

        try:

            subprocess.run(
                [
                    "yt-dlp",
                    "--no-playlist",
                    "--no-warnings",
                    "-o",
                    str(outfile),
                    row["url"],
                ],
                check=True,
                timeout=120,
            )

            status = "success"
            error = ""
            filename = outfile.name
            success += 1

            print("    ✓ Success")

        except Exception as e:

            status = "failed"
            error = str(e)[:500]
            filename = ""

            failed += 1

            print("    ✗ Failed")

    # ------------------------------------------------
    # Immediately append to manifest
    # ------------------------------------------------
    with open(RESUME_MANIFEST, "a", newline="", encoding="utf-8") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "video_id",
                "gloss",
                "split",
                "source",
                "url",
                "status",
                "error",
                "filename",
            ],
        )

        writer.writerow(
            {
                "video_id": video_id,
                "gloss": row["gloss"],
                "split": row["split"],
                "source": row["source"],
                "url": row["url"],
                "status": status,
                "error": error,
                "filename": filename,
            }
        )

    time.sleep(0.5)

print("\n")
print("=" * 60)
print("Resume Download Complete")
print("=" * 60)
print(f"Success           : {success}")
print(f"Failed            : {failed}")
print(f"Skipped           : {skipped}")
print(f"Manifest          : {RESUME_MANIFEST}")
print("=" * 60)