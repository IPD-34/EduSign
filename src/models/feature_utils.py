"""
feature_utils.py

Shared helper for converting a ProcessedSample's landmark arrays into
the single flattened per-frame feature vector the encoder expects
(RAW_FEATURE_DIM = 1692, per encoder.py). Used by both the Day 9
smoke test and the real Dataset/DataLoader (Day 10) -- keeping this
in one place avoids the two silently drifting apart.
"""

import numpy as np


def flatten_frame_features(sample) -> np.ndarray:
    """
    sample: a ProcessedSample (from sample_writer.load_sample()).
    Returns: (T, RAW_FEATURE_DIM) float array -- pose+face+left_hand+right_hand,
             concatenated per timestep, in that fixed order.
    """
    T = sample.pose.shape[0]
    pose_flat = sample.pose.reshape(T, -1)
    face_flat = sample.face.reshape(T, -1)
    left_flat = sample.left_hand.reshape(T, -1)
    right_flat = sample.right_hand.reshape(T, -1)
    return np.concatenate([pose_flat, face_flat, left_flat, right_flat], axis=1)