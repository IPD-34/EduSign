from pathlib import Path

import cv2

from src.landmarks.pose_detector import PoseDetector


def main():
    image = cv2.imread(str(Path("test.jpeg")))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    detector = PoseDetector()

    result = detector.detect(image)

    print("Pose detected:", len(result.pose_landmarks))


if __name__ == "__main__":
    main()