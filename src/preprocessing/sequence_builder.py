"""
sequence_builder.py

Converts a video into a ProcessedSample.

Pipeline:
    Video
        ↓
    VideoReader.probe()        — cheap metadata, no decode
        ↓
    frame_sampler.select_frame_indices()  — decide what to keep, before extraction
        ↓
    VideoReader.read(frame_indices=...)   — decode only what's needed
        ↓
    HolisticExtractor (per kept frame)
        ↓
    HolisticFrame sequence
        ↓
    ProcessedSample

No detector logic here. No sampling policy here beyond calling frame_sampler.

DAY 22 FIX: build() previously always imported and used the shared/WLASL
MAX_SEQUENCE_LENGTH (150) when calling select_frame_indices(), regardless
of the `source` argument passed in. This meant ASLLRP samples were being
truncated to 150 frames at EXTRACTION time -- before any .npz was written
-- even though ASLLRP_MAX_SEQUENCE_LENGTH (300) was set specifically
because continuous-signing utterances run longer than WLASL's isolated
signs. This is a real data-loss bug, not a logging bug: frames beyond
150 were permanently dropped for ASLLRP samples and cannot be recovered
without re-extraction.

Fix: build() now takes an explicit `max_sequence_length` parameter,
defaulting to the shared constant for backward compatibility. Callers
that process multiple sources (DatasetProcessor) must pass the correct
per-source value explicitly -- see dataset_processor.py's
_max_sequence_length_for() helper.
"""

import numpy as np

from src.landmarks.extractor import HolisticExtractor
from src.preprocessing.config import MAX_SEQUENCE_LENGTH, TARGET_FPS
from src.preprocessing.frame_sampler import select_frame_indices
from src.preprocessing.sample_schema import (
    LandmarkPresence,
    ProcessedSample,
    SampleMetadata,
)
from src.preprocessing.video_reader import VideoReader


class SequenceBuilder:

    def __init__(self):
        self.reader = VideoReader()
        self.extractor = HolisticExtractor()

    def build(
        self,
        video_path,
        *,
        video_id,
        gloss,
        split,
        source,
        frame_start=None,
        frame_end=None,
        translation=None,
        sign_spans=None, 
        max_sequence_length=MAX_SEQUENCE_LENGTH,
    ) -> ProcessedSample:
        """
        max_sequence_length: hard cap passed straight to
        select_frame_indices(). Defaults to the shared/WLASL constant
        for backward compatibility, but callers handling multiple
        sources (e.g. DatasetProcessor) MUST pass the correct
        per-source value explicitly -- do not rely on this default
        for ASLLRP or any other source with its own cap.
        """

        probe = self.reader.probe(video_path)

        frame_indices = select_frame_indices(
            num_frames=probe["num_frames"],
            fps=probe["fps"],
            target_fps=TARGET_FPS,
            max_sequence_length=max_sequence_length,
            frame_start=frame_start,
            frame_end=frame_end,
        )

        video = self.reader.read(video_path, frame_indices=frame_indices)

        pose_sequence, face_sequence, left_sequence, right_sequence = [], [], [], []
        pose_present, face_present, left_present, right_present = [], [], [], []

        for i, rgb_frame in enumerate(video["frames"]):

            if i % 25 == 0:
                print(f"    Frame {i + 1}/{video['num_frames']}")

            holistic_frame, _, _, _ = self.extractor.extract(rgb_frame)

            pose_sequence.append(holistic_frame.pose)
            face_sequence.append(holistic_frame.face)
            left_sequence.append(holistic_frame.left_hand)
            right_sequence.append(holistic_frame.right_hand)

            pose_present.append(holistic_frame.pose_present)
            face_present.append(holistic_frame.face_present)
            left_present.append(holistic_frame.left_hand_present)
            right_present.append(holistic_frame.right_hand_present)

        metadata = SampleMetadata(
            video_id=video_id,
            gloss=gloss,
            split=split,
            source=source,
            fps=video["fps"],
            num_frames=video["num_frames"],  # sampled count, not the raw file's frame count
            translation=translation,
            frame_indices=frame_indices,  
            sign_spans=sign_spans,
        )

        presence = LandmarkPresence(
            pose=np.array(pose_present, dtype=bool),
            left_hand=np.array(left_present, dtype=bool),
            right_hand=np.array(right_present, dtype=bool),
            face=np.array(face_present, dtype=bool),
        )

        return ProcessedSample(
            metadata=metadata,
            pose=np.stack(pose_sequence),
            left_hand=np.stack(left_sequence),
            right_hand=np.stack(right_sequence),
            face=np.stack(face_sequence),
            presence=presence,
        )