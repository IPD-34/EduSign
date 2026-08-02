from pathlib import Path
import csv
import shutil

from tqdm import tqdm

import fiftyone as fo
import fiftyone.utils.huggingface as fouh


# ==========================================================
# Paths
# ==========================================================

ROOT = Path(__file__).resolve().parents[1]

OUT_DIR = ROOT / "data" / "raw" / "wlasl" / "hf_mirror"

MANIFEST_DIR = ROOT / "data" / "raw" / "wlasl" / "manifests"

LOG_PATH = MANIFEST_DIR / "hf_mirror_recovery.csv"

OUT_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================
# Load FiftyOne dataset
# ==========================================================

try:
    print("Loading existing FiftyOne dataset...")

    dataset = fo.load_dataset("Voxel51/WLASL")

except Exception:

    print("Dataset not found locally.")
    print("Loading from Hugging Face...")

    dataset = fouh.load_from_hub("Voxel51/WLASL")


print(f"\nDataset name : {dataset.name}")
print(f"Samples      : {len(dataset)}")


# ==========================================================
# Recover videos
# ==========================================================

rows = []

copied = 0
already_copied = 0
missing = 0
copy_failed = 0

print("\nRecovering videos...\n")

for sample in tqdm(dataset.iter_samples(progress=False), total=len(dataset)):

    src = Path(sample.filepath)

    filename = src.name
    video_id = src.stem

    dst = OUT_DIR / filename

    # ---------------------------------------------
    # Already recovered
    # ---------------------------------------------

    if dst.exists():

        already_copied += 1

        rows.append(
            {
                "video_id": video_id,
                "filename": filename,
                "status": "already_copied",
                "src": str(src),
                "error": "",
            }
        )

        continue

    # ---------------------------------------------
    # Missing in HF cache
    # ---------------------------------------------

    if not src.exists():

        missing += 1

        rows.append(
            {
                "video_id": video_id,
                "filename": filename,
                "status": "missing_locally",
                "src": str(src),
                "error": "",
            }
        )

        continue

    # ---------------------------------------------
    # Copy file
    # ---------------------------------------------

    try:

        shutil.copy2(src, dst)

        copied += 1

        rows.append(
            {
                "video_id": video_id,
                "filename": filename,
                "status": "copied",
                "src": str(src),
                "error": "",
            }
        )

    except Exception as e:

        copy_failed += 1

        rows.append(
            {
                "video_id": video_id,
                "filename": filename,
                "status": "copy_failed",
                "src": str(src),
                "error": str(e),
            }
        )


# ==========================================================
# Save recovery manifest
# ==========================================================

with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "video_id",
            "filename",
            "status",
            "src",
            "error",
        ],
    )

    writer.writeheader()
    writer.writerows(rows)


# ==========================================================
# Summary
# ==========================================================

print("\n" + "=" * 50)
print("HF MIRROR RECOVERY SUMMARY")
print("=" * 50)

print(f"Dataset samples      : {len(dataset)}")
print(f"Copied               : {copied}")
print(f"Already copied       : {already_copied}")
print(f"Missing locally      : {missing}")
print(f"Copy failed          : {copy_failed}")

print(f"\nRecovery log written to:\n{LOG_PATH}")

print("=" * 50)