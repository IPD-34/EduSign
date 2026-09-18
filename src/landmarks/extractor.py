import cv2

from .face_detector import FaceDetector
from .face_remap import remap_face_landmarks
from .face_roi import compute_face_roi
from .frame_assembler import assemble_holistic_frame
from .hand_detector import HandDetector
from .hand_remap import merge_hand_results, remap_hand_landmarks
from .hand_roi import compute_hand_roi
from .pose_detector import PoseDetector

MIN_FACE_CROP_SIZE = 256  # target size for the upscaled crop, in pixels

# ASL Citizen extraction fix (see hand_roi.py's docstring for the root-
# cause finding this addresses): same "crop tightly, then upscale to a
# minimum working size" idea MIN_FACE_CROP_SIZE already used for the
# face, applied to hands for the first time. 900, not a smaller round
# number: A/B-tested directly against a real, genuinely borderline ASL
# Citizen frame (a mostly-sleeve-covered fist) -- targets as low as
# 1200px on a slightly-differently-anchored crop still failed to
# detect, while the original ad-hoc test (a ~250x200 crop upscaled 4x
# to 1000x800) did succeed (0.872 confidence). This one frame is
# fragile enough that no single number is guaranteed to rescue it --
# the real evidence for whether this helps is the aggregate hand-
# presence rate across the full 1,226-video slice (see the extraction
# report), not this one example -- but 900 is grounded in the one
# configuration actually confirmed to work, not a round-number guess.
MIN_HAND_CROP_SIZE = 900


class HolisticExtractor:

    def __init__(self):
        self.pose_detector = PoseDetector()
        self.face_detector = FaceDetector()
        self.hand_detector = HandDetector()

    def extract(self, rgb_image):
        height, width = rgb_image.shape[:2]

        pose_result = self.pose_detector.detect(rgb_image)
        face_result = self._detect_face(rgb_image, pose_result, width, height)
        hand_result = self._detect_hands(rgb_image, pose_result, width, height)

        frame = assemble_holistic_frame(
            pose_result, face_result, hand_result, rgb_image.shape,
        )

        return frame, pose_result, face_result, hand_result

    def _detect_face(self, rgb_image, pose_result, width, height):
        if not pose_result.pose_landmarks:
            return self.face_detector.detect(rgb_image)

        x0, y0, x1, y1 = compute_face_roi(pose_result.pose_landmarks[0], width, height)
        crop = rgb_image[y0:y1, x0:x1]

        if crop.size == 0:
            return self.face_detector.detect(rgb_image)

        crop_h, crop_w = crop.shape[:2]
        scale = max(1.0, MIN_FACE_CROP_SIZE / max(crop_w, crop_h))
        if scale > 1.0:
            crop = cv2.resize(
                crop,
                (int(crop_w * scale), int(crop_h * scale)),
                interpolation=cv2.INTER_LINEAR,
            )

        crop_result = self.face_detector.detect(crop)

        if not crop_result.face_landmarks:
            return self.face_detector.detect(rgb_image)  # fallback: try full frame as last resort

        return remap_face_landmarks(
            crop_result, x0, y0, x1 - x0, y1 - y0, width, height,
        )

    def _detect_hands(self, rgb_image, pose_result, width, height):
        """Pose-anchored crop-then-upscale-then-detect for EACH hand
        independently, falling back to a single full-frame detect()
        call (the pipeline's original, only behavior) whenever pose
        landmarks are unavailable or a hand's anchor points aren't
        trusted enough to crop around -- see hand_roi.py's docstring
        for why (root cause: scale, not confidence)."""
        if not pose_result.pose_landmarks:
            return self.hand_detector.detect(rgb_image)

        pose_landmarks = pose_result.pose_landmarks[0]

        left_landmarks, left_handedness = self._detect_one_hand(rgb_image, pose_landmarks, "left", width, height)
        right_landmarks, right_handedness = self._detect_one_hand(rgb_image, pose_landmarks, "right", width, height)

        if left_landmarks is None and right_landmarks is None:
            # Neither crop path produced anything (no trusted anchor,
            # or a trusted anchor whose crop still came up empty) --
            # last-resort full-frame attempt, same as the face path's
            # own final fallback.
            return self.hand_detector.detect(rgb_image)

        return merge_hand_results(left_landmarks, left_handedness, right_landmarks, right_handedness)

    def _detect_one_hand(self, rgb_image, pose_landmarks, side, width, height):
        roi = compute_hand_roi(pose_landmarks, side, width, height)
        if roi is None:
            return None, None  # pose itself doesn't trust this limb is usefully in frame

        x0, y0, x1, y1 = roi
        crop = rgb_image[y0:y1, x0:x1]
        if crop.size == 0:
            return None, None

        crop_h, crop_w = crop.shape[:2]
        scale = max(1.0, MIN_HAND_CROP_SIZE / max(crop_w, crop_h))
        if scale > 1.0:
            crop = cv2.resize(
                crop,
                (int(crop_w * scale), int(crop_h * scale)),
                interpolation=cv2.INTER_LINEAR,
            )

        crop_result = self.hand_detector.detect_in_crop(crop)
        return remap_hand_landmarks(crop_result, x0, y0, x1 - x0, y1 - y0, width, height)