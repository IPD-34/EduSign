import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.landmarks.extractor import HolisticExtractor


def main():

    image = cv2.imread("test.png")

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )

    extractor = HolisticExtractor()

    (
        frame,
        pose_result,
        face_result,
        hand_result,
    ) = extractor.extract(image)

    print("\n===== RAW MEDIAPIPE RESULTS =====")

    print("Poses :", len(pose_result.pose_landmarks))
    print("Faces :", len(face_result.face_landmarks))
    print("Hands :", len(hand_result.hand_landmarks))

    print("\n===== HOLISTIC FRAME =====")

    print("Pose :", frame.pose.shape)
    print("Face :", frame.face.shape)
    print("Left :", frame.left_hand.shape)
    print("Right:", frame.right_hand.shape)

    print()

    print("Pose Present :", frame.pose_present)
    print("Face Present :", frame.face_present)
    print("Left Present :", frame.left_hand_present)
    print("Right Present:", frame.right_hand_present)


if __name__ == "__main__":
    main()