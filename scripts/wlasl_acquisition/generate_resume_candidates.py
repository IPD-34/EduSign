from pathlib import Path
import json
import csv

ROOT = Path(__file__).resolve().parents[1]

METADATA_PATH = ROOT / "data" / "raw" / "wlasl" / "metadata" / "WLASL_v0.3.json"
MANIFEST_PATH = ROOT / "data" / "raw" / "wlasl" / "manifests" / "download_manifest.csv"
VIDEOS_DIR = ROOT / "data" / "raw" / "wlasl" / "videos"
OUTPUT_PATH = ROOT / "data" / "raw" / "wlasl" / "manifests" / "resume_candidates.csv"

# -----------------------------
# Load metadata
# -----------------------------
with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

metadata_by_id = {}
all_ids = set()

for gloss_entry in metadata:
    gloss = gloss_entry["gloss"]
    split = gloss_entry.get("split", "")

    for inst in gloss_entry["instances"]:
        video_id = str(inst["video_id"])

        metadata_by_id[video_id] = {
            "video_id": video_id,
            "gloss": gloss,
            "split": split,
            "url": inst["url"],
            "source": inst["source"],
        }

        all_ids.add(video_id)

# -----------------------------
# Attempted IDs
# -----------------------------
attempted_ids = set()

with open(MANIFEST_PATH, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        attempted_ids.add(str(row["video_id"]))

# -----------------------------
# Existing videos
# -----------------------------
recovered_ids = {
    p.stem
    for p in VIDEOS_DIR.glob("*.mp4")
}

# -----------------------------
# Compute candidate list
# -----------------------------
never_attempted = all_ids - attempted_ids
resume_candidates = sorted(never_attempted - recovered_ids)

# -----------------------------
# Save CSV
# -----------------------------
with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["video_id", "gloss", "split", "source", "url"],
    )

    writer.writeheader()

    for video_id in resume_candidates:
        writer.writerow(metadata_by_id[video_id])

print()
print("=" * 40)
print(f"Official metadata      : {len(all_ids)}")
print(f"Attempted previously   : {len(attempted_ids)}")
print(f"Recovered videos       : {len(recovered_ids)}")
print(f"Never attempted        : {len(never_attempted)}")
print(f"Resume candidates      : {len(resume_candidates)}")
print("=" * 40)
print()
print(f"Saved -> {OUTPUT_PATH}")