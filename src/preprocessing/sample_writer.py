"""
sample_writer.py

Handles serialization of ProcessedSample objects.

Responsibilities:
    - Save ProcessedSample -> .npz
    - Load .npz -> ProcessedSample

This module does NOT perform preprocessing.
It only converts between in-memory objects and disk.
"""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from src.preprocessing.sample_schema import (
    LandmarkPresence,
    ProcessedSample,
    SampleMetadata,
)


def write_sample(
    sample: ProcessedSample,
    output_path: Path,
) -> None:
    """
    Save a ProcessedSample as a compressed .npz file.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata_json = json.dumps(asdict(sample.metadata))

    np.savez_compressed(
        output_path,

        # Landmarks
        pose=sample.pose,
        face=sample.face,
        left_hand=sample.left_hand,
        right_hand=sample.right_hand,

        # Presence flags
        presence_pose=sample.presence.pose,
        presence_face=sample.presence.face,
        presence_left_hand=sample.presence.left_hand,
        presence_right_hand=sample.presence.right_hand,

        # Metadata
        metadata=metadata_json,
    )


def load_sample(
    input_path: Path,
) -> ProcessedSample:
    """
    Load a ProcessedSample from a .npz file.
    """

    with np.load(input_path, allow_pickle=False) as data:

        metadata = SampleMetadata(
            **json.loads(str(data["metadata"]))
        )

        presence = LandmarkPresence(
            pose=data["presence_pose"],
            face=data["presence_face"],
            left_hand=data["presence_left_hand"],
            right_hand=data["presence_right_hand"],
        )

        sample = ProcessedSample(
            metadata=metadata,

            pose=data["pose"],
            face=data["face"],
            left_hand=data["left_hand"],
            right_hand=data["right_hand"],

            presence=presence,
        )

    return sample