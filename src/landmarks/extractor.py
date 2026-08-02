import cv2

from .face_detector import FaceDetector
from .face_remap import remap_face_landmarks
from .face_roi import compute_face_roi
from .frame_assembler import assemble_holistic_frame
from .hand_detector import HandDetector
from .pose_detector import PoseDetector

MIN_FACE_CROP_SIZE = 256  # target size for the upscaled crop, in pixels


class HolisticExtractor:

    def __init__(self):
        self.pose_detector = PoseDetector()
        self.face_detector = FaceDetector()
        self.hand_detector = HandDetector()

    def extract(self, rgb_image):
        height, width = rgb_image.shape[:2]

        pose_result = self.pose_detector.detect(rgb_image)
        face_result = self._detect_face(rgb_image, pose_result, width, height)
        hand_result = self.hand_detector.detect(rgb_image)

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