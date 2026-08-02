from dataclasses import dataclass

import numpy as np


@dataclass
class HolisticFrame:
    pose: np.ndarray          # (33,4)
    face: np.ndarray          # (478,3)
    left_hand: np.ndarray     # (21,3)
    right_hand: np.ndarray    # (21,3)

    pose_present: bool
    face_present: bool
    left_hand_present: bool
    right_hand_present: bool

    image_width: int
    image_height: int