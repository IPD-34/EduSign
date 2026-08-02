import numpy as np

from .constants import (
    FACE_LANDMARKS,
    HAND_LANDMARKS,
    POSE_LANDMARKS,
)
from .schemas import HolisticFrame


def _empty_pose():
    return np.zeros((POSE_LANDMARKS, 4), dtype=np.float32)


def _empty_face():
    return np.zeros((FACE_LANDMARKS, 3), dtype=np.float32)


def _empty_hand():
    return np.zeros((HAND_LANDMARKS, 3), dtype=np.float32)


def _extract_pose(result):

    if not result.pose_landmarks:
        return _empty_pose(), False

    pose = np.zeros((POSE_LANDMARKS, 4), dtype=np.float32)

    for i, lm in enumerate(result.pose_landmarks[0]):
        pose[i] = [
            lm.x,
            lm.y,
            lm.z,
            lm.visibility,
        ]

    return pose, True


def _extract_face(result):

    if not result.face_landmarks:
        return _empty_face(), False

    face = np.zeros((FACE_LANDMARKS, 3), dtype=np.float32)

    for i, lm in enumerate(result.face_landmarks[0]):
        face[i] = [
            lm.x,
            lm.y,
            lm.z,
        ]

    return face, True


def _extract_hands(result):

    left = _empty_hand()
    right = _empty_hand()

    left_present = False
    right_present = False

    if not result.hand_landmarks:
        return (
            left,
            right,
            left_present,
            right_present,
        )

    for hand_landmarks, handedness in zip(
        result.hand_landmarks,
        result.handedness,
    ):

        label = handedness[0].category_name.lower()

        hand = np.zeros(
            (HAND_LANDMARKS, 3),
            dtype=np.float32,
        )

        for i, lm in enumerate(hand_landmarks):
            hand[i] = [
                lm.x,
                lm.y,
                lm.z,
            ]

        if label == "left":
            left = hand
            left_present = True

        else:
            right = hand
            right_present = True

    return (
        left,
        right,
        left_present,
        right_present,
    )


def assemble_holistic_frame(
    pose_result,
    face_result,
    hand_result,
    image_shape,
):

    height, width = image_shape[:2]

    pose, pose_present = _extract_pose(
        pose_result
    )

    face, face_present = _extract_face(
        face_result
    )

    (
        left,
        right,
        left_present,
        right_present,
    ) = _extract_hands(
        hand_result
    )

    return HolisticFrame(
        pose=pose,
        face=face,
        left_hand=left,
        right_hand=right,
        pose_present=pose_present,
        face_present=face_present,
        left_hand_present=left_present,
        right_hand_present=right_present,
        image_width=width,
        image_height=height,
    )