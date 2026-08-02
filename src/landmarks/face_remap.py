from dataclasses import dataclass


@dataclass
class RemappedLandmark:
    x: float
    y: float
    z: float


@dataclass
class RemappedFaceResult:
    face_landmarks: list


def remap_face_landmarks(crop_result, x0, y0, crop_w, crop_h, full_w, full_h):
    if not crop_result.face_landmarks:
        return RemappedFaceResult(face_landmarks=[])

    remapped = []
    for lm in crop_result.face_landmarks[0]:
        full_x = (x0 + lm.x * crop_w) / full_w
        full_y = (y0 + lm.y * crop_h) / full_h
        remapped.append(RemappedLandmark(x=full_x, y=full_y, z=lm.z))

    return RemappedFaceResult(face_landmarks=[remapped])