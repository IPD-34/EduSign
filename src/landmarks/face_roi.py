# Pose landmark indices (MediaPipe Pose topology)
NOSE = 0
LEFT_EAR = 7
RIGHT_EAR = 8
MOUTH_LEFT = 9
MOUTH_RIGHT = 10

FACE_ROI_LANDMARK_INDICES = [NOSE, LEFT_EAR, RIGHT_EAR, MOUTH_LEFT, MOUTH_RIGHT]


def compute_face_roi(pose_landmarks, image_width, image_height, padding=2.2):
    xs = [pose_landmarks[i].x * image_width for i in FACE_ROI_LANDMARK_INDICES]
    ys = [pose_landmarks[i].y * image_height for i in FACE_ROI_LANDMARK_INDICES]

    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    half_size = max(max(xs) - min(xs), max(ys) - min(ys)) * padding / 2 + 1

    x0 = max(0, int(cx - half_size))
    y0 = max(0, int(cy - half_size))
    x1 = min(image_width, int(cx + half_size))
    y1 = min(image_height, int(cy + half_size))

    return x0, y0, x1, y1