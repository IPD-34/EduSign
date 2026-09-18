"""
scripts/process_asl_citizen_full_parallel.py

Extends the already-validated 200-word extraction (process_asl_citizen_
scaling_parallel.py) to ASL Citizen's FULL 2,731-word vocabulary, now
that the data-sufficiency audit confirmed 97.9% of the full vocabulary
clears the 10+/split floor at the metadata level -- this extracts real
landmarks for it, not just counting rows.

Identical machinery to the 200-word script (same NUM_WORKERS=4, same
interleaved train/val/test shuffle, same resume-safe pre-filter, same
_init_worker/_process_one pattern) -- the ONLY change is the word
filter (full vocab instead of the top-200 FIXED list) and, as a direct
consequence of reusing the SAME PROCESSED_DIR, the already-extracted
6,744 files from the 200-word run are automatically skipped (existing
.npz at the expected path), not redone.

PREREQUISITE, not yet fully satisfied when this script was written:
only the 200-word slice's 6,744 source videos exist locally -- the
remaining ~76,655 videos for the rest of the vocabulary are being
downloaded by a separate concurrent process (download_asl_citizen_
full.py in the scratchpad). This script only processes videos that
ALREADY exist on disk at run time (build_all_samples filters on
video_path.exists()) -- it will naturally pick up more of the vocabulary
each time it's rerun as the download progresses, same resume-safe
philosophy as the rest of this pipeline. Rerun this script periodically
rather than expecting one run to catch everything.
"""
import json
from pathlib import Path

from process_asl_citizen_scaling_parallel import (
    build_all_samples, _init_worker, _process_one, PROCESSED_DIR, NUM_WORKERS,
)
import multiprocessing as mp
import random
import csv


SHUFFLE_SEED = 11


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=NUM_WORKERS)
    args = parser.parse_args()

    with open("logs/asl_citizen_FULL_vocab_word_list.json") as f:
        word_filter = set(json.load(f))

    all_samples = build_all_samples(word_filter)
    total_dataset_size = len(all_samples)

    rng = random.Random(SHUFFLE_SEED)
    rng.shuffle(all_samples)

    samples = []
    already_done = 0
    for s in all_samples:
        output_path = PROCESSED_DIR / s["split"] / f"{s['video_id']}.npz"
        if output_path.exists():
            already_done += 1
        else:
            samples.append(s)

    print(f"Total in target set (full vocab, videos downloaded so far): {total_dataset_size}")
    print(f"Already done         : {already_done}")
    print(f"Remaining to run     : {len(samples)}")

    if args.limit is not None:
        samples = samples[:args.limit]
        print(f"(--limit applied: processing first {len(samples)} of the remaining samples)")

    print(f"Using {args.workers} worker processes\n")

    if not samples:
        print("Nothing left to process (either fully done, or download hasn't produced new videos yet).")
        return

    manifest_rows = []
    successful = 0
    failed = 0
    archived = 0
    deleted = 0

    with mp.Pool(processes=args.workers, initializer=_init_worker, maxtasksperchild=100) as pool:
        for i, result in enumerate(pool.imap_unordered(_process_one, samples), start=1):
            manifest_rows.append(result)
            if result["status"] == "success":
                successful += 1
                if result.get("video_disposal") == "archived":
                    archived += 1
                elif result.get("video_disposal") == "deleted":
                    deleted += 1
            else:
                failed += 1
                print(f"[FAILED] {result['video_id']}: {result['status']}")

            if i % 200 == 0 or i == len(samples):
                print(f"Progress: {i}/{len(samples)} this run (success={successful}, failed={failed}, "
                      f"archived={archived}, deleted={deleted}) | "
                      f"total done overall: {already_done + successful}/{total_dataset_size}")

    if manifest_rows:
        manifest_path = PROCESSED_DIR / "manifest_full_vocab.csv"
        with open(manifest_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
            if f.tell() == 0:
                writer.writeheader()
            writer.writerows(manifest_rows)
        print(f"\nManifest appended to {manifest_path}")

    print("\n==========================\nProcessing Complete (this run)\n==========================")
    print(f"Video disposal — archived to external drive: {archived}, deleted (drive not connected): {deleted}")
    print(f"This run — Successful: {successful}, Failed: {failed}")
    print(f"Overall  — Done: {already_done + successful}/{total_dataset_size}")


if __name__ == "__main__":
    main()
