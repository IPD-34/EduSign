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

        # ASL Citizen extraction fix: a SEPARATE detector instance,
        # used for the pose-anchored per-hand crops (see extractor.py).
        # TWO real findings, both confirmed by direct A/B testing on a
        # real ASL Citizen frame before picking these settings (not
        # guessed):
        #   1. num_hands=1 is NOT just "num_hands=2 restricted to the
        #      top result" -- on the same crop, at the same confidence,
        #      num_hands=2 found a hand (score 0.872) that num_hands=1
        #      did not find at all. Whatever internal candidate
        #      selection/NMS the Tasks API does evidently differs by
        #      num_hands, so the crop detector stays num_hands=2 (it
        #      still returns at most one hand per SIDE in practice,
        #      since the crop is already anchored tightly around one
        #      wrist -- extractor.py picks the highest-confidence entry
        #      if more than one comes back, rather than assuming index 0).
        #   2. min_hand_detection_confidence=0.3 (below the library's
        #      0.5 default) is deliberately used HERE, but NOT for the
        #      full-frame fallback detector above: a crop already
        #      anchored to a real, pose-confirmed wrist position is a
        #      much lower-false-positive-risk context than scanning an
        #      entire frame at low confidence would be, so the same
        #      threshold argument that would be reckless on the full
        #      frame is reasonable on a tightly-anchored crop.
        crop_options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(
                model_asset_path=str(model_path)
            ),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.3,
        )
        self.crop_detector = vision.HandLandmarker.create_from_options(crop_options)

    def detect(self, rgb_image):
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_image,
        )

        return self.detector.detect(mp_image)

    def detect_in_crop(self, rgb_image):
        """Same as detect(), but via the crop-tuned detector (num_hands=2,
        lower confidence) -- used for pose-anchored per-hand crops, see
        extractor.py and the constructor comment above for why this is
        a separate, deliberately different configuration."""
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_image,
        )

        return self.crop_detector.detect(mp_image)