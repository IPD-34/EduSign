"""
run_asllrp_preprocessing_parallel.py

Parallel preprocessing for ASLLRP, mirroring run_preprocessing_parallel.py's
structure (resume-before-pool-dispatch, maxtasksperchild memory recycling,
manifest logging) but using ASLLRP_MAX_SEQUENCE_LENGTH (250), not WLASL's
150 — continuous-signing utterances are structurally longer; see config.py
comment and scripts/asllrp_acquisition/check_asllrp_length_distribution.py
for the measured distribution behind this choice.

Deliberately NOT run concurrently with the WLASL run — see project notes
on memory pressure from running multiple MediaPipe-heavy jobs at once.

Usage:
    uv run python scripts/run_asllrp_preprocessing_parallel.py
    uv run python scripts/run_asllrp_preprocessing_parallel.py --limit 20
    uv run python scripts/run_asllrp_preprocessing_parallel.py --workers 4
"""

import argparse
import csv
import multiprocessing as mp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.config import ASLLRP_PROCESSED_DIR, ASLLRP_XML_DIR, ASLLRP_VIDEO_DIR, ASLLRP_MAX_SEQUENCE_LENGTH
from src.preprocessing.metadata_readers.asllrp_reader import ASLLRPReader
from src.preprocessing.sequence_builder import SequenceBuilder
from src.preprocessing.sample_writer import write_sample
from src.preprocessing.video_reader import VideoReader

NUM_WORKERS = 4  # matches the stable worker count confirmed on this machine

_builder = None
_probe_reader = None


def _init_worker():
    global _builder, _probe_reader
    _builder = SequenceBuilder()
    _probe_reader = VideoReader()


def _process_one(sample_dict):
    global _builder, _probe_reader

    video_id = sample_dict["video_id"]
    gloss = sample_dict["gloss"]
    split = sample_dict["split"]
    source = sample_dict["source"]
    video_path = sample_dict["video_path"]
    frame_start = sample_dict.get("frame_start")
    frame_end = sample_dict.get("frame_end")
    translation = sample_dict.get("translation")

    output_path = ASLLRP_PROCESSED_DIR / split / f"{video_id}.npz"

    try:
        try:
            original_num_frames = _probe_reader.probe(video_path)["num_frames"]
        except Exception:
            original_num_frames = None

        processed = _builder.build(
            video_path=video_path,
            video_id=video_id,
            gloss=gloss,
            split=split,
            source=source,
            frame_start=frame_start,
            frame_end=frame_end,
            translation=translation,
        )

        write_sample(processed, output_path)

        sampled_num_frames = processed.metadata.num_frames

        return {
            "video_id": video_id,
            "gloss": gloss,
            "split": split,
            "original_num_frames": original_num_frames,
            "sampled_num_frames": sampled_num_frames,
            "hit_max_cap": sampled_num_frames >= ASLLRP_MAX_SEQUENCE_LENGTH,
            "status": "success",
        }

    except Exception as e:
        return {
            "video_id": video_id,
            "gloss": gloss,
            "split": split,
            "original_num_frames": None,
            "sampled_num_frames": None,
            "hit_max_cap": None,
            "status": f"failed: {e}",
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Process only first N NEW samples (smoke test)")
    parser.add_argument("--workers", type=int, default=NUM_WORKERS, help="Number of parallel worker processes")
    args = parser.parse_args()

    reader = ASLLRPReader(xml_dir=ASLLRP_XML_DIR, video_dir=ASLLRP_VIDEO_DIR)

    all_samples = []
    for s in reader.samples():
        all_samples.append({
            "video_id": s.video_id,
            "gloss": s.gloss,
            "split": s.split,
            "source": s.source,
            "video_path": s.video_path,
            "frame_start": s.frame_start,
            "frame_end": s.frame_end,
            "translation": s.translation,
        })

    total_dataset_size = len(all_samples)

    samples = []
    already_done = 0
    for s in all_samples:
        output_path = ASLLRP_PROCESSED_DIR / s["split"] / f"{s['video_id']}.npz"
        if output_path.exists():
            already_done += 1
        else:
            samples.append(s)

    print(f"Total in dataset : {total_dataset_size}")
    print(f"Already done     : {already_done}")
    print(f"Remaining to run : {len(samples)}")

    if args.limit is not None:
        samples = samples[:args.limit]
        print(f"(--limit applied: processing first {len(samples)} of the remaining samples)")

    print(f"Using {args.workers} worker processes")
    print(f"MAX_SEQUENCE_LENGTH (ASLLRP-specific): {ASLLRP_MAX_SEQUENCE_LENGTH}\n")

    if not samples:
        print("Nothing left to process — all samples already done.")
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

            if i % 50 == 0 or i == len(samples):
                print(f"Progress: {i}/{len(samples)} this run "
                      f"(success={successful}, failed={failed}) | "
                      f"total done overall: {already_done + successful}/{total_dataset_size}")

    if manifest_rows:
        manifest_path = ASLLRP_PROCESSED_DIR / "manifest.csv"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
            writer.writeheader()
            writer.writerows(manifest_rows)
        print(f"\nManifest written to {manifest_path} (this run's results only)")

    print("\n==========================")
    print("Processing Complete (ASLLRP, parallel)")
    print("==========================")
    print(f"This run — Successful: {successful}, Failed: {failed}")
    print(f"Overall  — Done: {already_done + successful}/{total_dataset_size}")
    if successful:
        capped = sum(1 for r in manifest_rows if r.get("hit_max_cap"))
        print(f"Hit max-length cap (this run): {capped}/{successful} ({100*capped/successful:.1f}%)")


if __name__ == "__main__":
    main()