"""
hand_roi.py

ASL Citizen extraction-fix follow-up: mirrors face_roi.py's
"crop tightly around the region of interest, using pose landmarks as
the anchor" strategy -- exact same idea, never previously applied to
hands. Root cause (confirmed empirically, not assumed): on ASL
Citizen's wider webcam-style framing (whole upper body visible, hand
occupying a small fraction of a 640x480 frame), MediaPipe's hand
detector found ZERO hand candidates on the full frame even at
min_hand_detection_confidence as low as 0.05 -- not a confidence
problem, a scale problem. The SAME frame, cropped to the hand region
and upscaled, detected a hand at 0.872 confidence, comfortably above
even the DEFAULT 0.5 threshold. Lowering the confidence threshold
alone would have changed nothing, since no candidate is proposed to
score against it in the first place on the full frame.

Pose landmarks 15/17/19/21 (left wrist/pinky/index/thumb) and
16/18/20/22 (right) are BlazePose's own approximate fingertip
landmarks -- lower-fidelity than the dedicated hand model, but exactly
the multi-point spread face_roi.py already relies on (nose/ears/mouth)
for a stable bounding box, and directly confirmed on a real ASL
Citizen frame: the visible hand's four points landed with
visibility 0.93-0.98, tightly clustered around the hand's real
on-screen position; the out-of-frame hand's four points scored
visibility 0.15-0.31 -- pose's own confidence already flags "this
limb probably isn't usefully visible" before a crop is even attempted.
"""

LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_PINKY, RIGHT_PINKY = 17, 18
LEFT_INDEX, RIGHT_INDEX = 19, 20
LEFT_THUMB, RIGHT_THUMB = 21, 22

HAND_ROI_LANDMARK_INDICES = {
    "left": [LEFT_WRIST, LEFT_PINKY, LEFT_INDEX, LEFT_THUMB],
    "right": [RIGHT_WRIST, RIGHT_PINKY, RIGHT_INDEX, RIGHT_THUMB],
}

# Below this mean pose-landmark visibility for the 4 hand-anchor points,
# treat the limb as "pose itself doesn't trust this is usefully in
# frame" and skip the crop attempt (falls back to the full-frame path,
# same as when pose has no landmarks at all) -- confirmed against the
# real frame above: the genuinely-out-of-frame left hand's 4 points
# averaged ~0.22, the genuinely-visible right hand's averaged ~0.96 --
# a wide, unambiguous gap, so 0.4 is a conservative cut well clear of
# either observed cluster, not a fine-tuned knob.
MIN_ANCHOR_VISIBILITY = 0.4

# Hand-specific padding, larger than face_roi's 2.2: wrist/fingertip
# pose landmarks are noisier frame-to-frame than face landmarks, and a
# fist (all 4 anchor points nearly coincident) can collapse the raw
# bounding box to near-zero -- both need more margin than a face crop
# does.
DEFAULT_PADDING = 3.5

# Absolute floor on crop half-size, as a fraction of the larger image
# dimension -- guards specifically against the closed-fist collapse
# case (padding alone can't rescue a bounding box that started at ~0),
# calibrated against the real test crop (250x200 out of 640x480, i.e.
# ~35-40% of frame) that was empirically confirmed to work.
MIN_HALF_SIZE_FRACTION = 0.18


def compute_hand_roi(pose_landmarks, side, image_width, image_height, padding=DEFAULT_PADDING):
    """
    Returns (x0, y0, x1, y1) crop box in pixel coordinates, or None if
    the anchor points' mean visibility is below MIN_ANCHOR_VISIBILITY
    (pose itself doesn't trust this limb is usefully in frame -- caller
    should fall back to the full-frame path, not force a crop around
    noise).
    """
    indices = HAND_ROI_LANDMARK_INDICES[side]
    points = [pose_landmarks[i] for i in indices]

    mean_visibility = sum(p.visibility for p in points) / len(points)
    if mean_visibility < MIN_ANCHOR_VISIBILITY:
        return None

    xs = [p.x * image_width for p in points]
    ys = [p.y * image_height for p in points]

    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    raw_half_size = max(max(xs) - min(xs), max(ys) - min(ys)) * padding / 2
    min_half_size = MIN_HALF_SIZE_FRACTION * max(image_width, image_height)
    half_size = max(raw_half_size, min_half_size) + 1

    x0 = max(0, int(cx - half_size))
    y0 = max(0, int(cy - half_size))
    x1 = min(image_width, int(cx + half_size))
    y1 = min(image_height, int(cy + half_size))

    return x0, y0, x1, y1
