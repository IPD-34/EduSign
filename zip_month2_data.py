"""
zip_month2_data.py

Packages the Month 2 ASLLRP data + supporting files for Kaggle upload:
corrected ASLLRP .npz corpus, merged vocab, ASLLRP token mapping, and
the Day 12 WLASL checkpoint (needed for first-launch encoder transfer).

Uses zipfile directly, not Compress-Archive -- per Month 1's own
finding, Compress-Archive threw OutOfMemoryException on a large file
count; this writes one file at a time instead of loading the whole
archive into memory.
"""

import zipfile
from pathlib import Path

OUTPUT_ZIP = "edusign_month2_asllrp_data.zip"

files_to_zip = []

# Full corrected ASLLRP corpus -- confirmed 1,234 .npz files across
# train/val/test, plus manifest.csv
asllrp_dir = Path("data/processed/asllrp")
if asllrp_dir.exists():
    files_to_zip.extend(f for f in asllrp_dir.rglob("*") if f.is_file())

# Supporting files needed by train_month2_asllrp.py
for f in [
    Path("data/vocab_merged.json"),
    Path("data/asllrp_token_to_id.json"),
    Path("checkpoints/day12_latest.pt"),
]:
    if f.exists():
        files_to_zip.append(f)
    else:
        print(f"[WARNING] Expected file not found, skipping: {f}")

print(f"Zipping {len(files_to_zip)} files...")

with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
    for i, f in enumerate(files_to_zip, 1):
        zf.write(f, arcname=str(f).replace("\\", "/"))
        if i % 200 == 0:
            print(f"  {i}/{len(files_to_zip)} written...")

print(f"Done! Created {OUTPUT_ZIP}")