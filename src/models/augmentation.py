"""
augmentation.py

Stage B1: spatial-only landmark augmentation. Perturbs per-frame
COORDINATE VALUES only -- never touches frame count, sequence length,
or which timesteps exist. This is deliberately why Stage B1 needs no
sign_spans/boundary-target remapping (Day 31's alignment machinery is
untouched: build_boundary_targets() operates on frame_indices, which
this file never reads or changes) and no target_tokens changes (gloss
sequence is landmark-value-independent by construction). Frame-count-
changing augmentation (speed perturbation / resampling) is explicitly
OUT of scope here -- that's Stage B2, deferred, and would need the same
class of remapping work as Day 31's original boundary-label build.

--- WHICH STREAMS, AND WHY ---
All four streams (pose, face, left_hand, right_hand) are jittered, not
hands-only -- excluding pose/face entirely would forgo real augmentation
diversity on the body/context signal for no clearly stated benefit. But
hands get a SMALLER magnitude than pose/face (half), not the same one:
hands are the primary carrier of sign identity (handshape) and are also
physically smaller in real-world size than the torso, so the same
NORMALIZED-coordinate jitter represents a larger fraction of the hand's
own size than of the torso's -- standard reasoning in skeleton-based
augmentation for treating smaller/more semantically-loaded joints more
conservatively.

Pose's 4th channel (visibility/confidence, NOT a coordinate) is NEVER
perturbed -- confirmed the schema directly (src/landmarks/schemas.py:
pose is (33,4), face/hands are (33,3)/(478,3)/(21,3); only pose carries
a 4th column, and it is a confidence score, not a spatial coordinate).

--- MAGNITUDE, GROUNDED IN REAL DATA + A DOCUMENTED CAVEAT ---
Checked this session against a real processed sample
(data/processed/asllrp/train, first file alphabetically): landmark x/y
values sit on MediaPipe's normalized-image-coordinate scale (observed
range roughly -1.3 to 1.9, frame-0 values concentrated around 0.2-0.7;
shoulder-to-shoulder distance ~0.27 in x). Gaussian jitter std is set
relative to that scale:
  JITTER_STD_POSE_FACE = 0.010  (pose, face)
  JITTER_STD_HANDS     = 0.005  (left_hand, right_hand -- half)
This project has no measured, sample-level ground-truth for its OWN
MediaPipe tracking jitter (that would need repeated takes of an
identical static pose, not available) -- 0.01 is an ESTIMATE informed
by commonly-cited approximate MediaPipe static-subject tracking-noise
magnitudes (roughly 0.5-2% of a normalized frame dimension in general
MediaPipe robustness discussions), not a project-specific measurement.
Stated as an estimate, not asserted as measured.

Scale (+/-5%) and rotation (+/-5 degrees, image-plane x/y only, NOT a
full 3D rotation -- a documented simplification) are each drawn ONCE
PER SAMPLE (per utterance) and applied identically to every frame of
that utterance -- this represents "recorded from a slightly different
camera distance/angle," a whole-clip-level variation. Applying a fresh
random scale/rotation independently per FRAME would instead look like
the body flickering/warping frame to frame, which no real recording
ever does -- that would be a bug, not augmentation. Per-frame Gaussian
jitter, by contrast, IS drawn independently per frame per coordinate --
this mimics genuine per-frame MediaPipe tracking noise, which really is
frame-independent.

Rotation/scale center: the mean, across ALL frames of the utterance, of
the midpoint between MediaPipe BlazePose landmarks 11 and 12 (left/right
shoulder -- standard, externally-documented MediaPipe Pose topology,
not project-specific). Averaged across frames rather than taken from
frame 0 alone, so a person moving during the clip doesn't leave later
frames rotating/scaling around a stale, poorly-centered point.

--- PER-EPOCH, NOT FIXED-PER-SAMPLE ---
This module has no caching and no sample-index-derived seed -- every
call draws fresh randomness from the global RNG. Wired into
SignLanguageDataset.__getitem__() (src/models/dataset.py), which the
DataLoader calls fresh every time a sample is drawn, every epoch --
so the same utterance gets DIFFERENT noise/scale/rotation each epoch,
which is standard practice for this technique (if it were fixed per
sample, the model would simply memorize one new fixed variant instead
of learning to be robust to a distribution of variants).
"""

import dataclasses

import numpy as np

LEFT_SHOULDER_IDX = 11   # standard MediaPipe BlazePose topology
RIGHT_SHOULDER_IDX = 12

JITTER_STD_POSE_FACE = 0.010
JITTER_STD_HANDS = 0.005
SCALE_DELTA = 0.05          # +/- 5%
MAX_ROTATION_DEGREES = 5.0  # +/- 5 degrees, image-plane (x,y) only


def _shoulder_center(pose: np.ndarray) -> tuple[float, float]:
    """pose: (T, 33, 4). Returns (cx, cy) -- mean shoulder-midpoint across
    all frames, x/y only."""
    mid = 0.5 * (pose[:, LEFT_SHOULDER_IDX, :2] + pose[:, RIGHT_SHOULDER_IDX, :2])  # (T, 2)
    return float(mid[:, 0].mean()), float(mid[:, 1].mean())


def _rotate_scale_xy(coords_xy: np.ndarray, cx: float, cy: float, cos_t: float, sin_t: float, scale: float) -> np.ndarray:
    """coords_xy: (..., 2). Rotate+scale around (cx, cy). Returns a NEW array."""
    dx = coords_xy[..., 0] - cx
    dy = coords_xy[..., 1] - cy
    new_x = cx + scale * (dx * cos_t - dy * sin_t)
    new_y = cy + scale * (dx * sin_t + dy * cos_t)
    out = coords_xy.copy()
    out[..., 0] = new_x
    out[..., 1] = new_y
    return out


def _augment_stream(arr: np.ndarray, cx: float, cy: float, cos_t: float, sin_t: float, scale: float,
                     jitter_std: float, rng: np.random.Generator) -> np.ndarray:
    """arr: (T, N, 3), or (T, N, 4) for pose (column 3 = visibility).
    Returns a NEW array; the input is never mutated. Only columns 0:3
    (x,y,z) are ever touched -- a 4th column (pose's visibility) is
    excluded by the [...,:3]/[...,:2] slicing below automatically, no
    branching needed, so there is nothing to get wrong per-caller."""
    out = arr.copy()
    xy = out[..., :2]
    out[..., :2] = _rotate_scale_xy(xy, cx, cy, cos_t, sin_t, scale)
    out[..., 2] = out[..., 2] * scale  # depth: scaled (consistent "closer/further" story), not rotated
    # Per-frame independent Gaussian jitter, coordinates only (x,y,z) --
    # never the visibility column, per the module docstring.
    out[..., :3] += rng.normal(loc=0.0, scale=jitter_std, size=out[..., :3].shape).astype(out.dtype)
    return out


def apply_spatial_augmentation(sample, rng: np.random.Generator | None = None):
    """
    sample: a ProcessedSample (src/preprocessing/sample_schema.py).
    Returns a NEW ProcessedSample with augmented pose/face/left_hand/
    right_hand arrays (dataclasses.replace -- metadata and presence are
    passed through by reference, completely untouched: this function
    never changes frame count, sign_spans, frame_indices, gloss, or any
    other metadata field). The input sample object itself is never
    mutated.

    rng: inject a numpy Generator for a reproducible/testable call (e.g.
    a smoke test's seeded run); defaults to a fresh, unseeded Generator
    (genuinely different noise on every call -- the per-epoch behavior
    this module's docstring describes).
    """
    if rng is None:
        rng = np.random.default_rng()

    cx, cy = _shoulder_center(sample.pose)
    scale = float(rng.uniform(1.0 - SCALE_DELTA, 1.0 + SCALE_DELTA))
    theta = np.deg2rad(rng.uniform(-MAX_ROTATION_DEGREES, MAX_ROTATION_DEGREES))
    cos_t, sin_t = float(np.cos(theta)), float(np.sin(theta))

    new_pose = _augment_stream(sample.pose, cx, cy, cos_t, sin_t, scale, JITTER_STD_POSE_FACE, rng)
    new_face = _augment_stream(sample.face, cx, cy, cos_t, sin_t, scale, JITTER_STD_POSE_FACE, rng)
    new_left = _augment_stream(sample.left_hand, cx, cy, cos_t, sin_t, scale, JITTER_STD_HANDS, rng)
    new_right = _augment_stream(sample.right_hand, cx, cy, cos_t, sin_t, scale, JITTER_STD_HANDS, rng)

    return dataclasses.replace(sample, pose=new_pose, face=new_face, left_hand=new_left, right_hand=new_right)
