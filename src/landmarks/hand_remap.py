"""
hand_remap.py -- mirrors face_remap.py exactly, for hand crops.

Converts a HandLandmarker result computed on a CROPPED, UPSCALED region
back into full-frame-normalized [0,1] coordinates, and merges the
(up to two, independent) per-hand crop results into one combined
object with the same shape HandDetector.detect() would have returned
(a .hand_landmarks list and a .handedness list, one entry per detected
hand) -- so frame_assembler.assemble_holistic_frame's existing
_extract_hands() consumes it completely unchanged.
"""

from dataclasses import dataclass, field


@dataclass
class RemappedLandmark:
    x: float
    y: float
    z: float


@dataclass
class HandednessCategory:
    category_name: str
    score: float


@dataclass
class CombinedHandResult:
    hand_landmarks: list = field(default_factory=list)
    handedness: list = field(default_factory=list)


def remap_hand_landmarks(crop_result, x0, y0, crop_w, crop_h, full_w, full_h):
    """
    crop_result: a HandLandmarker result computed on a crop tightly
    anchored to ONE side (left or right) -- but the crop detector is
    configured num_hands=2 (see hand_detector.py's constructor comment
    for why num_hands=1 is NOT equivalent), so more than one candidate
    can technically come back even though the crop targets one hand.
    Picks the HIGHEST-confidence entry explicitly rather than assuming
    index 0 is it. Returns (landmarks, handedness) remapped to full-
    frame coordinates, or (None, None) if the crop had no detection --
    caller decides what to do next (e.g. fall back to full-frame
    detection for this hand).
    """
    if not crop_result.hand_landmarks:
        return None, None

    best_idx = max(range(len(crop_result.handedness)), key=lambda i: crop_result.handedness[i][0].score)

    remapped = []
    for lm in crop_result.hand_landmarks[best_idx]:
        full_x = (x0 + lm.x * crop_w) / full_w
        full_y = (y0 + lm.y * crop_h) / full_h
        remapped.append(RemappedLandmark(x=full_x, y=full_y, z=lm.z))

    handedness = crop_result.handedness[best_idx]
    return remapped, handedness


def merge_hand_results(left_landmarks, left_handedness, right_landmarks, right_handedness):
    result = CombinedHandResult()
    if left_landmarks is not None:
        result.hand_landmarks.append(left_landmarks)
        result.handedness.append(left_handedness)
    if right_landmarks is not None:
        result.hand_landmarks.append(right_landmarks)
        result.handedness.append(right_handedness)
    return result
