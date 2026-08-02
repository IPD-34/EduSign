from pathlib import Path

import cv2

from src.landmarks.face_detector import FaceDetector


def main():
    image_path = Path("test.png")   # <-- change this

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Could not read {image_path}")
        return

    print("Image shape:", image.shape)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    detector = FaceDetector()
    result = detector.detect(image)

    print("Faces detected:", len(result.face_landmarks))

    if result.face_landmarks:
        print("Face landmarks:", len(result.face_landmarks[0]))


if __name__ == "__main__":
    main()