from pathlib import Path

import cv2

from src.landmarks.hand_detector import HandDetector


def main():
    image = cv2.imread(str(Path("test.jpeg")))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    detector = HandDetector()
    result = detector.detect(image)

    print("Hands detected:", len(result.hand_landmarks))

    for i, handedness in enumerate(result.handedness):
        print(f"Hand {i}: {handedness[0].category_name}")


if __name__ == "__main__":
    main()