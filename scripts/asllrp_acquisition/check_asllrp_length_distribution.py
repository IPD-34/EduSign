"""
check_asllrp_length_distribution.py

Before picking a MAX_SEQUENCE_LENGTH for ASLLRP, look at the real
distribution of utterance lengths (in sampled frames, at TARGET_FPS)
rather than guessing a bigger number. The Week 1 smoke test showed
60% of samples hitting the 150-frame cap shared with WLASL --
continuous-signing utterances are structurally longer than WLASL's
isolated signs, so a shared cap likely isn't appropriate.

Does NOT run MediaPipe extraction -- only probes video metadata and
runs frame_sampler's trim/resample math, which is cheap and fast.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.preprocessing.config import (
    ASLLRP_XML_DIR,
    ASLLRP_VIDEO_DIR,
    TARGET_FPS,
)
from src.preprocessing.metadata_readers.asllrp_reader import ASLLRPReader
from src.preprocessing.frame_sampler import select_frame_indices
from src.preprocessing.video_reader import VideoReader


def main():
    reader = ASLLRPReader(xml_dir=ASLLRP_XML_DIR, video_dir=ASLLRP_VIDEO_DIR)
    probe_reader = VideoReader()

    lengths = []
    errors = 0

    for i, sample in enumerate(reader.samples(), start=1):
        try:
            probe = probe_reader.probe(sample.video_path)

            # Use a very high cap here so we see the TRUE untruncated
            # length after trim+resample, not one already cut short
            indices = select_frame_indices(
                num_frames=probe["num_frames"],
                fps=probe["fps"],
                target_fps=TARGET_FPS,
                max_sequence_length=999999,
                frame_start=sample.frame_start,
                frame_end=sample.frame_end,
            )
            lengths.append(len(indices))

        except Exception as e:
            errors += 1
            print(f"[SKIP] {sample.video_id}: {e}")

        if i % 100 == 0:
            print(f"  Probed {i} utterances...")

    if not lengths:
        print("No usable utterances found.")
        return

    lengths.sort()
    n = len(lengths)

    def pct(p):
        idx = min(n - 1, int(n * p))
        return lengths[idx]

    print("\n==========================")
    print(f"ASLLRP Utterance Length Distribution (sampled frames @ {TARGET_FPS} fps)")
    print("==========================")
    print(f"Total utterances probed : {n}")
    print(f"Errors/skipped          : {errors}")
    print(f"Min                     : {lengths[0]}")
    print(f"p25                     : {pct(0.25)}")
    print(f"Median (p50)            : {pct(0.50)}")
    print(f"p75                     : {pct(0.75)}")
    print(f"p90                     : {pct(0.90)}")
    print(f"p95                     : {pct(0.95)}")
    print(f"p99                     : {pct(0.99)}")
    print(f"Max                     : {lengths[-1]}")

    for cap in [150, 200, 250, 300, 400, 500]:
        hit = sum(1 for l in lengths if l >= cap)
        print(f"Cap-hit rate at {cap:>4}: {hit}/{n} ({100*hit/n:.1f}%)")


if __name__ == "__main__":
    main()