import sys, shutil, csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.config import RAW_DATA_DIR
from src.preprocessing.metadata_readers.wlasl_reader import WLASLReader

reader = WLASLReader()  # adjust constructor args if it needs explicit paths

# pick 4 samples with DISTINCT glosses, not just the first 4 in the list
samples = list(reader.samples())
picked = []
seen_glosses = set()
for s in samples:
    if s.gloss not in seen_glosses and Path(s.video_path).exists():
        picked.append(s)
        seen_glosses.add(s.gloss)
    if len(picked) == 4:
        break

dest_dir = RAW_DATA_DIR / "channel_check"
dest_dir.mkdir(parents=True, exist_ok=True)

rows = []
for s in picked:
    src_path = Path(s.video_path)
    dest_name = f"{s.gloss}_1.mp4"
    shutil.copy(src_path, dest_dir / dest_name)
    rows.append({"filename": dest_name, "gloss": s.gloss})
    print(f"Copied {src_path.name} -> {dest_name} (gloss: {s.gloss})")

with open(dest_dir / "manifest.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["filename", "gloss"])
    writer.writeheader()
    writer.writerows(rows)

print(f"\n{len(rows)} stand-in clips + manifest.csv ready in {dest_dir}")