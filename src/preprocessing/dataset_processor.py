"""
dataset_processor.py

Processes a dataset of (video, gloss) samples through the shared
landmark-extraction pipeline. Reader- and source-agnostic — works
for WLASL, ASLLRP, HF mirror, or anything else that yields samples
with the expected attributes.

Also logs a per-sample manifest (original vs. sampled frame count,
whether MAX_SEQUENCE_LENGTH was hit) so the "% of clips hit the cap"
question has a real measured answer instead of an assumed one.

Pipeline:

    <any reader>.samples()
          ↓
    SequenceBuilder
          ↓
    SampleWriter
          ↓
    Processed Dataset (.npz) + manifest.csv
"""

import csv
from pathlib import Path

from src.preprocessing.config import MAX_SEQUENCE_LENGTH
from src.preprocessing.sample_writer import write_sample
from src.preprocessing.sequence_builder import SequenceBuilder
from src.preprocessing.video_reader import VideoReader


class DatasetProcessor:
    """
    Processes every sample yielded by the given reader.
    """

    def __init__(self, reader, processed_dir: Path):
        self.reader = reader
        self.processed_dir = processed_dir
        self.builder = SequenceBuilder()
        self.probe_reader = VideoReader()  # metadata-only probe, for manifest logging

    def process(self, limit=None):

        total = 0
        successful = 0
        failed = 0
        count = 0
        manifest_rows = []

        for sample in self.reader.samples():

            total += 1

            try:

                print(f"\nProcessing {sample.video_id} ({sample.gloss})")
                print("  Building sequence...")

                try:
                    original_num_frames = self.probe_reader.probe(sample.video_path)["num_frames"]
                except Exception:
                    original_num_frames = None

                processed = self.builder.build(
                    video_path=sample.video_path,
                    video_id=sample.video_id,
                    gloss=sample.gloss,
                    split=sample.split,
                    source=sample.source,
                    frame_start=getattr(sample, "frame_start", None),
                    frame_end=getattr(sample, "frame_end", None),
                    translation=getattr(sample, "translation", None),
                )

                print("  Sequence built.")
                print("  Writing sample...")

                output_path = (
                    self.processed_dir
                    / sample.split
                    / f"{sample.video_id}.npz"
                )

                write_sample(processed, output_path)

                sampled_num_frames = processed.metadata.num_frames

                manifest_rows.append({
                    "video_id": sample.video_id,
                    "gloss": sample.gloss,
                    "split": sample.split,
                    "original_num_frames": original_num_frames,
                    "sampled_num_frames": sampled_num_frames,
                    "hit_max_cap": sampled_num_frames >= MAX_SEQUENCE_LENGTH,
                    "status": "success",
                })

                successful += 1

                print(f"[{successful}] Processed {sample.video_id} ({sample.gloss})")

                count += 1

                if limit is not None and count >= limit:
                    break

            except Exception as e:

                failed += 1

                manifest_rows.append({
                    "video_id": getattr(sample, "video_id", "unknown"),
                    "gloss": getattr(sample, "gloss", ""),
                    "split": getattr(sample, "split", ""),
                    "original_num_frames": None,
                    "sampled_num_frames": None,
                    "hit_max_cap": None,
                    "status": f"failed: {e}",
                })

                print(f"[FAILED] {sample.video_id}: {e}")

        self._write_manifest(manifest_rows)

        print("\n==========================")
        print(f"Processing Complete — source: {self.reader.__class__.__name__}")
        print("==========================")
        print(f"Total      : {total}")
        print(f"Successful : {successful}")
        print(f"Failed     : {failed}")

        if successful:
            capped = sum(1 for r in manifest_rows if r.get("hit_max_cap"))
            print(f"Hit max-length cap: {capped}/{successful} ({100*capped/successful:.1f}%)")

    def _write_manifest(self, rows):
        if not rows:
            return
        manifest_path = self.processed_dir / "manifest.csv"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nManifest written to {manifest_path}")