"""
wlasl_reader.py

Reads WLASL metadata and exposes each video as a structured sample.

Responsibilities:
    - Load WLASL_v0.3.json
    - Iterate over all video instances
    - Return dataset information only

This module does NOT:
    - Read videos
    - Extract landmarks
    - Save .npz files
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.preprocessing.config import (
    WLASL_METADATA,
    WLASL_VIDEO_DIR,
)


@dataclass
class WLASLSample:
    """
    Represents one video in the WLASL dataset.
    """

    video_id: str
    gloss: str
    split: str
    source: str
    video_path: Path
    frame_start: Optional[int]
    frame_end: Optional[int]


class WLASLReader:
    """
    Reads WLASL metadata and yields one sample at a time.
    """

    def __init__(self):

        with open(WLASL_METADATA, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

    def samples(self):
        """
        Iterate over every video in WLASL.
        """

        for gloss_entry in self.metadata:

            gloss = gloss_entry["gloss"]

            for instance in gloss_entry["instances"]:

                video_id = str(instance["video_id"])

                split = instance["split"]

                source = instance.get("url", "")

                video_path = WLASL_VIDEO_DIR / f"{video_id}.mp4"

                if not video_path.exists():
                    continue

                yield WLASLSample(
                    video_id=video_id,
                    gloss=gloss,
                    split=split,
                    source=source,
                    video_path=video_path,
                    frame_start=instance.get("frame_start"),
                    frame_end=instance.get("frame_end"),
                )