from pathlib import Path
import json
import csv
from urllib.parse import urlparse

# ==========================================================
# Paths
# ==========================================================

ROOT = Path(__file__).resolve().parents[1]

METADATA_PATH = ROOT / "data" / "raw" / "wlasl" / "metadata" / "WLASL_v0.3.json"
VIDEOS_DIR = ROOT / "data" / "raw" / "wlasl" / "videos"
MANIFEST_DIR = ROOT / "data" / "raw" / "wlasl" / "manifests"

MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

MISSING_TXT = MANIFEST_DIR / "missing.txt"
DETAILS_CSV = MANIFEST_DIR / "missing_details.csv"

# ==========================================================
# Load metadata
# ==========================================================

with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

# ==========================================================
# Existing video IDs
# ==========================================================

existing = {
    p.stem
    for p in VIDEOS_DIR.glob("*.mp4")
}

print(f"Existing videos: {len(existing)}")

missing_ids = []
rows = []

total = 0

for gloss_entry in metadata:

    gloss = gloss_entry["gloss"]

    for inst in gloss_entry["instances"]:

        total += 1

        video_id = str(inst["video_id"])

        if video_id in existing:
            continue

        url = inst.get("url", "")
        domain = urlparse(url).netloc

        missing_ids.append(video_id)

        rows.append({
            "video_id": video_id,
            "gloss": gloss,
            "url": url,
            "domain": domain,
        })

# ==========================================================
# Write missing.txt
# ==========================================================

with open(MISSING_TXT, "w", encoding="utf-8") as f:

    for vid in sorted(missing_ids, key=int):
        f.write(f"{vid}\n")

# ==========================================================
# Write detailed CSV
# ==========================================================

with open(DETAILS_CSV, "w", newline="", encoding="utf-8") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "video_id",
            "gloss",
            "url",
            "domain",
        ],
    )

    writer.writeheader()
    writer.writerows(rows)

print("\n==============================")
print(f"Total metadata entries : {total}")
print(f"Recovered videos       : {len(existing)}")
print(f"Missing videos         : {len(missing_ids)}")
print("==============================")

print(f"\nmissing.txt      -> {MISSING_TXT}")
print(f"missing_details.csv -> {DETAILS_CSV}")