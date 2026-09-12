"""
check_asllrp_failures.py

Reads data/processed/asllrp/manifest.csv (this run's results) and prints
just the failed rows with their error messages -- a quick way to see
what's actually going wrong with the 33 failures from the corrected-cap
(300) ASLLRP re-run, without scrolling through 512 successful rows.

Usage:
    uv run python scripts/check_asllrp_failures.py
"""

import csv
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.config import ASLLRP_PROCESSED_DIR

MANIFEST_PATH = ASLLRP_PROCESSED_DIR / "manifest.csv"


def main():
    if not MANIFEST_PATH.exists():
        print(f"No manifest found at {MANIFEST_PATH}")
        return

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    failed = [r for r in rows if r["status"] != "success"]
    print(f"Total rows in this manifest: {len(rows)}")
    print(f"Failed: {len(failed)}\n")

    if not failed:
        print("No failures in this manifest.")
        return

    # Group by error type/message to spot patterns rather than reading
    # 33 individual tracebacks one at a time.
    error_counts = Counter()
    for r in failed:
        # status looks like "failed: <exception message>"
        msg = r["status"].replace("failed: ", "").strip()
        # Bucket by exception type (first word/class name) if present
        error_counts[msg[:80]] += 1

    print("=" * 60)
    print("Error message frequency (grouped, truncated to 80 chars)")
    print("=" * 60)
    for msg, count in error_counts.most_common():
        print(f"{count:>3}x  {msg}")

    print("\n" + "=" * 60)
    print("Full failed rows")
    print("=" * 60)
    for r in failed:
        print(f"{r['video_id']:>12} | {r['gloss']:>20} | {r['split']:>6} | {r['status']}")


if __name__ == "__main__":
    main()