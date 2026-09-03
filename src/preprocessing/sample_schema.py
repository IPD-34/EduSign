"""
sample_schema.py

Defines the processed representation of one sign video.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class SampleMetadata:
    video_id: str
    gloss: str
    split: str
    source: str
    fps: float
    num_frames: int
    translation: Optional[str] = None
    frame_indices: Optional[list[int]] = None
    sign_spans: Optional[list[list[int] | None]] = None


@dataclass
class LandmarkPresence:
    pose: np.ndarray        # (T,) bool
    left_hand: np.ndarray   # (T,) bool
    right_hand: np.ndarray  # (T,) bool
    face: np.ndarray        # (T,) bool


@dataclass
class ProcessedSample:
    metadata: SampleMetadata

    pose: np.ndarray
    left_hand: np.ndarray
    right_hand: np.ndarray
    face: np.ndarray

    presence: LandmarkPresence