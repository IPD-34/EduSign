from pathlib import Path

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class HandDetector:
    def __init__(self):
        model_path = Path("models/hand_landmarker.task")

        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(
                model_asset_path=str(model_path)
            ),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
        )

        self.detector = vision.HandLandmarker.create_from_options(options)

    def detect(self, rgb_image):
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_image,
        )

        return self.detector.detect(mp_image)