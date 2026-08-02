"""
frame_sampler.py

Decides which frame indices to process, before any extraction happens.
Pure index arithmetic — no video decoding, unit-testable on plain ints.

Two independent steps:
1. Trim to the annotated signing segment (frame_start/frame_end), if any.
   frame_end == -1 or None means "no trim info, use whole clip."
2. Resample the trimmed range to TARGET_FPS, then hard-cap at
   MAX_SEQUENCE_LENGTH via uniform downsampling (not head-truncation),
   so a long real sign's tail isn't just cut off.
"""

from typing import List, Optional

import numpy as np


def select_frame_indices(
    num_frames: int,
    fps: float,
    target_fps: float,
    max_sequence_length: int,
    frame_start: Optional[int] = None,
    frame_end: Optional[int] = None,
) -> List[int]:

    if num_frames <= 0:
        return []

    # 1. Trim to annotated segment
    # in select_frame_indices(), replace the trim step: 
    start = (frame_start - 1) if frame_start is not None and frame_start > 0 else 0
    end = frame_end if frame_end is not None and frame_end > 0 else num_frames
    end = min(end, num_frames)

    if start >= end:
        start, end = 0, num_frames  # degenerate annotation — fall back to whole clip

    trimmed = list(range(start, end))

    # 2. Uniform resample to target fps
    if fps <= 0:
        fps = target_fps  # bad metadata guard
    step = max(1, round(fps / target_fps))
    sampled = trimmed[::step] or trimmed[:1]

    # 3. Hard cap, uniform downsample
    if len(sampled) > max_sequence_length:
        idx = np.linspace(0, len(sampled) - 1, max_sequence_length).round().astype(int)
        sampled = [sampled[i] for i in sorted(set(idx.tolist()))]

    return sampled