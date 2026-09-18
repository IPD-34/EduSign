"""
scripts/process_asl_citizen_overlap_priority.py

The WLASL cross-dataset check needs all 1,158 ASL-Citizen/WLASL
overlap words at 10+/split coverage, but the full-corpus download+
extraction (83,399 videos, unordered/shuffled) only reaches 88/1,158
of them so far -- the other 1,070 are scattered randomly through the
remaining ~76,000 videos, so waiting for the FULL job to get there
"naturally" means waiting on most of a ~60+ hour job for a result that
only needs a ~23K-video slice of it.

This downloads+extracts ONLY the videos for the 1,158 overlap words --
same download/extraction machinery as the full-corpus scripts, same
PROCESSED_DIR (so nothing here conflicts with or duplicates the
full-corpus job also running; both are resume-safe against the same
target files). Runs both steps in one script since the overlap-word
slice is small enough that download won't meaningfully outpace
extraction the way it did for the full 83K-video corpus.
"""
import sys
import csv
import json
import multiprocessing as mp
import random
from pathlib import Path
from remotezip import RemoteZip

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from process_asl_citizen_scaling_parallel import (
    _init_worker, _process_one, PROCESSED_DIR, NUM_WORKERS, wait_for_disk_space, OLD_LOCAL_DIR,
)

LOCAL_DIR = Path("data/raw/asl_citizen_scaling_videos")  # REDIRECTED to D: -- same path as
                                                            # process_asl_citizen_scaling_parallel.py,
                                                            # must match so both scripts share progress
                                                            # instead of duplicating downloads
SPLITS_DIR = Path(
    r"C:\Users\KRUPAS~1\AppData\Local\Temp\claude\D--College-Projects---Resume-bidirectional-sign-language"
    r"\1f4ac0b6-58be-4111-abc9-012b28f39d49\scratchpad\asl_citizen\ASL_Citizen\splits"
)
URL = "https://download.microsoft.com/download/b/8/8/b88c0bae-e6c1-43e1-8726-98cf5af36ca4/ASL_Citizen.zip"
SHUFFLE_SEED = 11


def load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    with open("logs/_tmp_citizen_floor_wlasl_overlap.json") as f:
        word_filter = set(json.load(f))

    targets = []
    for split in ("train", "val", "test"):
        for r in load_csv(SPLITS_DIR / f"{split}.csv"):
            gloss = r["Gloss"].strip().lower()
            if gloss in word_filter:
                targets.append((split, r["Video file"], gloss))

    print(f"Total videos for the {len(word_filter)}-word overlap vocabulary: {len(targets)}")

    # Step 1: download whatever isn't already on disk (skips anything the
    # full-corpus download already fetched -- same video files, same paths --
    # AND anything still sitting on the old C: location, so a stalled bulk
    # move doesn't cause a wasted re-download of the same video)
    to_download = [(s, f) for s, f, g in targets
                   if not (LOCAL_DIR / "videos" / s / f).exists()
                   and not (OLD_LOCAL_DIR / "videos" / s / f).exists()]
    print(f"Videos to download: {len(to_download)}")
    if to_download:
        with RemoteZip(URL, timeout=30) as rz:  # BUG FIX: no timeout meant a stalled
                                                   # connection could hang forever (same
                                                   # bug class found and fixed for Groq
                                                   # clients earlier today) -- confirmed
                                                   # remotezip forwards **kwargs straight
                                                   # into requests.get(), so this is real
            for i, (split, fname) in enumerate(to_download):
                wait_for_disk_space()  # D: now holds the videos too -- same floor applies to downloads
                dest_dir = LOCAL_DIR / "videos" / split
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest_path = dest_dir / fname
                if dest_path.exists():
                    continue
                try:
                    data = rz.read("ASL_Citizen/videos/" + fname)
                    dest_path.write_bytes(data)
                except Exception as e:
                    print(f"[DOWNLOAD FAILED] {fname}: {type(e).__name__}: {e}")
                if (i + 1) % 1000 == 0:
                    print(f"  downloaded {i+1}/{len(to_download)}")
    print("Download phase done.\n")

    # Step 2: extract whatever's downloaded but not yet extracted
    samples = []
    already_done = 0
    for split, fname, gloss in targets:
        video_path = LOCAL_DIR / "videos" / split / fname
        if not video_path.exists():
            video_path = OLD_LOCAL_DIR / "videos" / split / fname  # fallback: old C: location
            if not video_path.exists():
                continue
        video_id = f"aslciti_{Path(fname).stem}"
        output_path = PROCESSED_DIR / split / f"{video_id}.npz"
        if output_path.exists():
            already_done += 1
            continue
        samples.append({"video_id": video_id, "gloss": gloss, "split": split, "video_path": str(video_path)})

    rng = random.Random(SHUFFLE_SEED)
    rng.shuffle(samples)
    print(f"Already extracted: {already_done}")
    print(f"Remaining to extract: {len(samples)}")

    if not samples:
        print("Nothing left to extract.")
        return

    successful, failed, archived, deleted = 0, 0, 0, 0
    with mp.Pool(processes=NUM_WORKERS, initializer=_init_worker, maxtasksperchild=100) as pool:
        for i, result in enumerate(pool.imap_unordered(_process_one, samples), start=1):
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
                print(f"Progress: {i}/{len(samples)} (success={successful}, failed={failed}, "
                      f"archived={archived}, deleted={deleted}) | "
                      f"total done overall: {already_done + successful}/{len(targets)}")

    print(f"\nDone. Extracted {successful} this run, {failed} failed. "
          f"Overall: {already_done + successful}/{len(targets)}")
    print(f"Video disposal — archived to external drive: {archived}, deleted (drive not connected): {deleted}")


if __name__ == "__main__":
    main()
