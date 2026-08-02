from pathlib import Path

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class FaceDetector:
    def __init__(self):
        model_path = Path("models/face_landmarker.task")

        options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(
                model_asset_path=str(model_path)
            ),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.3,
            min_face_presence_confidence=0.3,
            min_tracking_confidence=0.3,
        )

        self.detector = vision.FaceLandmarker.create_from_options(options)

    def detect(self, rgb_image):
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_image,
        )

        return self.detector.detect(mp_image)