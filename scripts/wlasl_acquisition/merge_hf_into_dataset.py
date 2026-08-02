from pathlib import Path
import shutil
import csv

# ==========================================================
# Paths
# ==========================================================

ROOT = Path(__file__).resolve().parents[1]

VIDEOS_DIR = ROOT / "data" / "raw" / "wlasl" / "videos"
HF_DIR = ROOT / "data" / "raw" / "wlasl" / "hf_mirror"
MANIFEST_DIR = ROOT / "data" / "raw" / "wlasl" / "manifests"

MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

LOG_PATH = MANIFEST_DIR / "hf_merge_manifest.csv"

# ==========================================================
# Validation
# ==========================================================

if not VIDEOS_DIR.exists():
    raise FileNotFoundError(f"Dataset folder not found:\n{VIDEOS_DIR}")

if not HF_DIR.exists():
    raise FileNotFoundError(f"HF mirror folder not found:\n{HF_DIR}")

# ==========================================================
# Scan directories
# ==========================================================

existing_files = {f.name: f for f in VIDEOS_DIR.glob("*.mp4")}
hf_files = {f.name: f for f in HF_DIR.glob("*.mp4")}

print(f"Existing dataset videos : {len(existing_files)}")
print(f"HF mirror videos        : {len(hf_files)}")

rows = []

already_present = 0
copied = 0

# ==========================================================
# Merge
# ==========================================================

for filename, src in sorted(hf_files.items()):

    dst = VIDEOS_DIR / filename
    video_id = Path(filename).stem

    if dst.exists():

        already_present += 1

        rows.append({
            "video_id": video_id,
            "filename": filename,
            "status": "already_present"
        })

        continue

    shutil.copy2(src, dst)

    copied += 1

    rows.append({
        "video_id": video_id,
        "filename": filename,
        "status": "copied_from_hf"
    })

# ==========================================================
# Save manifest
# ==========================================================

with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "video_id",
            "filename",
            "status",
        ],
    )

    writer.writeheader()
    writer.writerows(rows)

# ==========================================================
# Final stats
# ==========================================================

final_total = len(list(VIDEOS_DIR.glob("*.mp4")))

print("\n" + "=" * 60)
print("MERGE SUMMARY")
print("=" * 60)

print(f"Original dataset         : {len(existing_files)}")
print(f"HF mirror               : {len(hf_files)}")
print(f"Already present         : {already_present}")
print(f"Copied from HF          : {copied}")
print("-" * 60)
print(f"Final unique videos     : {final_total}")

print(f"\nMerge manifest saved to:")
print(LOG_PATH)

print("=" * 60)