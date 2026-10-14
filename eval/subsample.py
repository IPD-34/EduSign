r"""Deterministic replacement for Uni-Sign's random frame subsampling.

Original branch (C:\Users\l\Desktop\personb\Uni-Sign\datasets.py, S2T_Dataset_online.load_pose):

    637  duration = len(pose['scores'])
    640  if duration > self.max_length:
    641      tmp = sorted(random.sample(range(duration), k=self.max_length))
    642  else:
    643      tmp = list(range(duration))
    ...
    647  skeletons = pose['keypoints']
    648  confs = pose['scores']
    651  for index in tmp: skeletons_tmp.append(skeletons[index]); confs_tmp.append(confs[index])

The same branch exists at datasets.py:473 (S2T_Dataset) and :573 (S2T_Dataset_news).
The cap comes from `self.max_length = args.max_length` (datasets.py:611) whose
default is 256 (utils.py:523, `--max_length`). Output of the random branch: exactly
max_length indices, sorted (temporal order), distinct, drawn without replacement.
Keys indexed: pose['keypoints'] and pose['scores'] (per-frame sequences).
"""

DEFAULT_MAX_FRAMES = 256  # utils.py:523 (--max_length default)


def subsample_indices(n_frames, max_frames=DEFAULT_MAX_FRAMES):
    """Frame indices to keep, deterministic, in temporal order.

    If n_frames <= max_frames: all indices, [0, 1, ..., n_frames - 1].
    Otherwise: max_frames indices, index i (i = 0..max_frames-1) being

        (i * (n_frames - 1)) // (max_frames - 1)       (integer floor division)

    i.e. evenly spaced from frame 0 to frame n_frames - 1 inclusive. Because
    n_frames > max_frames the spacing is > 1, so the indices are strictly
    increasing (distinct) and all lie in [0, n_frames - 1]. For max_frames == 1
    the single index is 0.
    """
    if n_frames < 0:
        raise ValueError("n_frames must be >= 0")
    if max_frames < 1:
        raise ValueError("max_frames must be >= 1")
    if n_frames <= max_frames:
        return list(range(n_frames))
    if max_frames == 1:
        return [0]
    return [(i * (n_frames - 1)) // (max_frames - 1) for i in range(max_frames)]


def subsample_pose(pose, max_frames=DEFAULT_MAX_FRAMES):
    """Apply subsample_indices to the per-frame entries the original branch indexes.

    Mirrors datasets.py:637-656: duration is len(pose['scores']); 'keypoints' and
    'scores' are indexed with the same indices and returned as lists. Returns a NEW
    dict with only those two keys (as the original discards everything else) and
    does not modify `pose`. Frames themselves are not copied.
    """
    idx = subsample_indices(len(pose["scores"]), max_frames)
    return {
        "keypoints": [pose["keypoints"][i] for i in idx],
        "scores": [pose["scores"][i] for i in idx],
    }
