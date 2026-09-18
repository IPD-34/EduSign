"""
scripts/process_asl_citizen_scaling_parallel.py

Parallel, resumable replacement for process_asl_citizen_scaling.py --
the single-process version was killed after 15h11m at a REAL measured
throughput of 2.23 files/min (2033/911.2min), ~6x slower than the
~13 files/min figure that had been carried over from an earlier,
never-precisely-remeasured run -- and after 15+ hours it had only
gotten partway through train.csv (val/test still at 0), because
ASLCitizenSliceReader.samples() iterates split-by-split (all of train,
then all of val, then all of test) -- fine for the original single
40-word/1226-video slice, badly wrong at 6,744 videos where "still
working through train" for 15 hours means zero usable (train+val+test)
data for evaluation this whole time.

TWO fixes, both grounded in this project's own existing, proven
infrastructure -- not invented for this task:

1. PARALLELISM: reuses run_preprocessing_parallel.py's exact pattern
   (mp.Pool, NUM_WORKERS=4 -- that specific number is not a guess, it's
   the empirically-validated stable count for THIS machine, documented
   in that script: "11 caused memory pressure/thrashing on this machine
   (16GB RAM)"). Each worker builds its own SequenceBuilder (own
   MediaPipe Holistic instance) -- not shared across processes, same
   requirement noted in the original script.

2. INTERLEAVED ORDER: the samples list is built from ALL THREE splits
   together, then shuffled (fixed seed) before dispatch -- so partial
   completion at any point gives roughly proportional train/val/test
   coverage, instead of exhausting train before touching val or test.

RESUME-SAFE (same as run_preprocessing_parallel.py): already-done
samples (an existing .npz at the expected output path) are filtered
out BEFORE the pool starts, not re-submitted -- the 2,059 files the
killed single-process run already produced are preserved and reused,
not redone. If this process is itself interrupted (session restart or
otherwise), rerunning this exact script picks up from wherever it left
off at zero cost, rather than needing a manual resume decision.
"""
import argparse
import csv
import json
import multiprocessing as mp
import random
import shutil
import string
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.sequence_builder import SequenceBuilder
from src.preprocessing.sample_writer import write_sample
from src.preprocessing.video_reader import VideoReader

LOCAL_DIR = Path("data/raw/asl_citizen_scaling_videos")  # REDIRECTED from C: scratchpad to D:
                                                            # after today's disk-full crisis -- D: has
                                                            # real headroom, C: doesn't. data/raw/ is
                                                            # already gitignored, matches this project's
                                                            # own convention for raw external data.
OLD_LOCAL_DIR = Path(  # FALLBACK: the bulk move of already-downloaded videos from the old
    r"C:\Users\KRUPAS~1\AppData\Local\Temp\claude\D--College-Projects---Resume-bidirectional-sign-language"
    r"\1f4ac0b6-58be-4111-abc9-012b28f39d49\scratchpad\asl_citizen_scaling"
)  # C: location stalled (both mv and robocopy hung -- likely a locked-file/AV-scan
   # interaction, not diagnosed further) -- rather than block on finishing that move,
   # build_all_samples() checks both locations, so nothing is blocked either way and
   # the split-across-two-drives state resolves itself naturally as extraction (via
   # its own archive-or-delete step) works through whichever copy it finds.
SPLITS_DIR = Path(
    r"C:\Users\KRUPAS~1\AppData\Local\Temp\claude\D--College-Projects---Resume-bidirectional-sign-language"
    r"\1f4ac0b6-58be-4111-abc9-012b28f39d49\scratchpad\asl_citizen\ASL_Citizen\splits"
)
PROCESSED_DIR = Path("data/processed/asl_citizen_scaling")

NUM_WORKERS = 4  # empirically-validated stable count on this machine, per run_preprocessing_parallel.py
SHUFFLE_SEED = 11
MAX_SEQUENCE_LENGTH = 150  # WLASL/isolated-sign default -- ASL Citizen clips are short isolated signs, same regime

_builder = None
_probe_reader = None


def _init_worker():
    global _builder, _probe_reader
    _builder = SequenceBuilder()
    _probe_reader = VideoReader()


D_DRIVE_MIN_FREE_GB = 20
ARCHIVE_MARKER_FOLDER = "EduSign_Archive"  # create this folder at the root of the external
                                             # drive once connected -- auto-detected by letter,
                                             # no config needed beyond that


def wait_for_disk_space(drive_path="D:\\", min_free_gb=D_DRIVE_MIN_FREE_GB, check_interval=30):
    """PAUSES (does not kill) the caller until drive_path has at least
    min_free_gb free. Checked before EVERY video, not just once at
    startup -- a startup-only check would miss the exact failure mode
    from today's crisis (plenty of space at launch, exhausted hours
    into a run). Logs clearly on both pause and resume so this is
    visible in status checks, not a silent stall that looks like a hang."""
    free_gb = shutil.disk_usage(drive_path).free / (1024 ** 3)
    if free_gb > min_free_gb:
        return
    print(f"[DISK PAUSE] {drive_path} at {free_gb:.2f}GB free (floor: {min_free_gb}GB) -- "
          f"pausing until space recovers. Not a crash -- extraction/download will resume automatically.")
    while free_gb <= min_free_gb:
        time.sleep(check_interval)
        free_gb = shutil.disk_usage(drive_path).free / (1024 ** 3)
    print(f"[DISK RESUME] {drive_path} back to {free_gb:.2f}GB free -- continuing.")


def _find_archive_drive():
    """Looks for a folder named EduSign_Archive at the root of any drive
    letter -- auto-detects the external drive whenever it's plugged in,
    without needing to know its letter in advance. Returns the archive
    directory Path, or None if not found (caller falls back to delete)."""
    for letter in string.ascii_uppercase:
        if letter in ("C",):  # never archive to the low-space system drive
            continue
        candidate = Path(f"{letter}:\\{ARCHIVE_MARKER_FOLDER}")
        try:
            if candidate.is_dir():
                return candidate
        except OSError:
            continue
    return None


def archive_or_delete_video(video_path, split, fname):
    """Tries to MOVE the source video to the external archive drive
    (preserves it for a possible future Tier C raw-video approach)
    instead of deleting it -- deletion is only the FALLBACK when the
    archive drive isn't connected right now, so extraction is never
    blocked waiting for it. Returns 'archived' or 'deleted' so callers
    can log/count which happened."""
    archive_root = _find_archive_drive()
    if archive_root is not None:
        try:
            dest_dir = archive_root / "asl_citizen_scaling_videos" / split
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_path = dest_dir / fname
            shutil.move(str(video_path), str(dest_path))
            return "archived"
        except Exception:
            pass  # archive drive present but move failed (e.g. disconnected mid-move) -- fall through to delete
    video_path.unlink()
    return "deleted"


def _process_one(sample_dict):
    global _builder, _probe_reader
    video_id = sample_dict["video_id"]
    gloss = sample_dict["gloss"]
    split = sample_dict["split"]
    video_path = Path(sample_dict["video_path"])
    output_path = PROCESSED_DIR / split / f"{video_id}.npz"

    wait_for_disk_space()  # checked before every video, not just at startup

    try:
        try:
            original_num_frames = _probe_reader.probe(video_path)["num_frames"]
        except Exception:
            original_num_frames = None

        processed = _builder.build(
            video_path=video_path, video_id=video_id, gloss=gloss, split=split,
            source="asl_citizen", frame_start=None, frame_end=None,
            max_sequence_length=MAX_SEQUENCE_LENGTH,
        )
        write_sample(processed, output_path)
        sampled_num_frames = processed.metadata.num_frames

        # Dispose of the source video AS SOON AS its landmark is verified
        # written -- not a periodic/manual cleanup pass. Verify (don't just
        # assume) the landmark is genuinely readable before touching the
        # video at all -- same check cleanup_asl_citizen_extracted_videos.py
        # uses. Prefers archiving to the external drive (Tier C raw-video
        # fallback stays possible); only deletes if that drive isn't
        # connected right now -- never blocks extraction waiting for it.
        disposal = None
        try:
            with np.load(output_path, allow_pickle=False) as check:
                json.loads(str(check["metadata"]))
            disposal = archive_or_delete_video(video_path, split, video_path.name)
        except Exception:
            pass  # landmark unreadable or video already gone -- leave the video, next cleanup pass will sort it out

        return {"video_id": video_id, "gloss": gloss, "split": split,
                "original_num_frames": original_num_frames, "sampled_num_frames": sampled_num_frames,
                "hit_max_cap": sampled_num_frames >= MAX_SEQUENCE_LENGTH, "status": "success",
                "video_disposal": disposal}
    except Exception as e:
        return {"video_id": video_id, "gloss": gloss, "split": split,
                "original_num_frames": None, "sampled_num_frames": None, "hit_max_cap": None,
                "status": f"failed: {type(e).__name__}: {e}", "video_disposal": None}


def build_all_samples(word_filter):
    def load_csv(path):
        with open(path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    all_samples = []
    for split in ("train", "val", "test"):
        rows = load_csv(SPLITS_DIR / f"{split}.csv")
        for row in rows:
            gloss = row["Gloss"].strip().lower()
            if gloss not in word_filter:
                continue
            fname = row["Video file"]
            video_path = LOCAL_DIR / "videos" / split / fname
            if not video_path.exists():
                video_path = OLD_LOCAL_DIR / "videos" / split / fname  # fallback: old C: location
                if not video_path.exists():
                    continue
            video_id = f"aslciti_{Path(fname).stem}"
            all_samples.append({"video_id": video_id, "gloss": gloss, "split": split, "video_path": str(video_path)})
    return all_samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=NUM_WORKERS)
    args = parser.parse_args()

    # FIXED, deterministic word list (see script comment history / report):
    # top400_words.json was found to be non-deterministic across separate
    # recomputations (Python set-iteration-order-dependent tie-breaks,
    # since ASL Citizen's per-word counts cluster too tightly for most
    # comparisons to have a unique winner) -- caused a 640-video mismatch
    # between what was downloaded and what a fresh recompute considered
    # "top-200". This file is the deterministic ranking of the exact
    # 200 words actually downloaded, not recomputed from scratch.
    with open("logs/asl_citizen_scaling_word_list_FIXED.json") as f:
        word_filter = set(json.load(f))

    all_samples = build_all_samples(word_filter)
    total_dataset_size = len(all_samples)

    rng = random.Random(SHUFFLE_SEED)
    rng.shuffle(all_samples)  # interleave train/val/test so partial progress is usable

    samples = []
    already_done = 0
    for s in all_samples:
        output_path = PROCESSED_DIR / s["split"] / f"{s['video_id']}.npz"
        if output_path.exists():
            already_done += 1
        else:
            samples.append(s)

    print(f"Total in target set : {total_dataset_size}")
    print(f"Already done         : {already_done}")
    print(f"Remaining to run     : {len(samples)}")

    if args.limit is not None:
        samples = samples[:args.limit]
        print(f"(--limit applied: processing first {len(samples)} of the remaining samples)")

    print(f"Using {args.workers} worker processes\n")

    if not samples:
        print("Nothing left to process.")
        return

    manifest_rows = []
    successful = 0
    failed = 0

    with mp.Pool(processes=args.workers, initializer=_init_worker, maxtasksperchild=100) as pool:
        for i, result in enumerate(pool.imap_unordered(_process_one, samples), start=1):
            manifest_rows.append(result)
            if result["status"] == "success":
                successful += 1
            else:
                failed += 1
                print(f"[FAILED] {result['video_id']}: {result['status']}")

            if i % 100 == 0 or i == len(samples):
                print(f"Progress: {i}/{len(samples)} this run (success={successful}, failed={failed}) | "
                      f"total done overall: {already_done + successful}/{total_dataset_size}")

    if manifest_rows:
        manifest_path = PROCESSED_DIR / "manifest_parallel.csv"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
            writer.writeheader()
            writer.writerows(manifest_rows)
        print(f"\nManifest written to {manifest_path}")

    print("\n==========================\nProcessing Complete (parallel)\n==========================")
    print(f"This run — Successful: {successful}, Failed: {failed}")
    print(f"Overall  — Done: {already_done + successful}/{total_dataset_size}")


if __name__ == "__main__":
    main()
