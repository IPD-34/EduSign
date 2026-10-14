r"""Pose-input controls (zeroed, shuffled), copied unchanged from the script that
produced the baseline predictions (pred_openasl.csv).

Source: C:\Users\l\Desktop\personb\Uni-Sign\run_batch_unisign.py

Not copied: eval_translation.py (personb, cmd_controls, lines 29-71) defines a
related but different pair of controls that work on files in a folder (perm over
sorted file names, default seed 0, zeroed keeps non-float arrays). It did not
produce pred_openasl.csv and is intentionally not reproduced here.
"""
import random

import numpy as np


def zeroed_pose(pose):
    """Zeroed control. Copied from run_batch_unisign.py lines 44-48 (make_zeroed):

        def make_zeroed(pose):
            n = len(pose["scores"])
            kp = [np.zeros_like(k) for k in pose["keypoints"]]
            sc = [np.zeros_like(s) for s in pose["scores"]]
            return {"keypoints": kp, "scores": sc}

    Every per-frame keypoint and score array becomes zeros of the same shape and
    dtype; the number of frames is kept. Only "keypoints" and "scores" are
    returned (any other keys in `pose` are dropped, as in the original). The
    input is not modified. (The unused `n = len(pose["scores"])` line is omitted;
    it had no effect other than the same KeyError if "scores" is missing.)
    """
    kp = [np.zeros_like(k) for k in pose["keypoints"]]
    sc = [np.zeros_like(s) for s in pose["scores"]]
    return {"keypoints": kp, "scores": sc}


def shuffle_mapping(ids, seed):
    """Shuffled control: which clip's pose each clip receives.

    Copied from run_batch_unisign.py lines 88-93 (inside main):

        rng = random.Random(args.seed)
        shuffled = ids[:]
        while True:  # derangement-ish: no clip keeps its own pose
            rng.shuffle(shuffled)
            if all(a != b for a, b in zip(ids, shuffled)) or len(ids) < 2:
                break

    and used at lines 99-100 as `zip(ids, shuffled)`: clip ids[i] is fed the pose
    of clip shuffled[i]. `ids` must be in the order the script used (sorted pose
    file names, optionally truncated by --max_clips). The script passes
    args.seed, whose default is 42 (utils.py line 486). Uses the standard
    library `random.Random`, not numpy, to reproduce the same permutation.

    Returns a dict {clip_id: donor_clip_id}. For len(ids) >= 2 no clip maps to
    itself; for len(ids) < 2 the single clip maps to itself (original behaviour).
    """
    ids = list(ids)
    rng = random.Random(seed)
    shuffled = ids[:]
    while True:  # derangement-ish: no clip keeps its own pose
        rng.shuffle(shuffled)
        if all(a != b for a, b in zip(ids, shuffled)) or len(ids) < 2:
            break
    return dict(zip(ids, shuffled))
