"""
compare_drift_vs_stateless_normalization.py

Month 3 deliverable's part (b): per real ASLLRP sample, how often and by
how much does robustness.drift_tracker's EWMA head_size reference diverge
from robustness.normalisation's stateless per-frame reference?

Comparison is restricted to frames where the face is actually present
that frame (presence_face[t] == True) -- those are the only frames where
the stateless reference has a defined value to compare against at all.
On a face-absent frame, drift tracking still has a (held) value but the
stateless version has none -- that's not a "divergence to measure," it's
the coverage difference already tested directly in
test_drift_tracking_covers_at_least_as_many_frames_as_stateless_on_real_samples.

Reports, per sample: number of face-present frames compared, mean
absolute difference and max absolute difference in head_size (same units
as the landmark coordinates -- head_size is a bounding-box diagonal), plus
an overall summary and an honest read on whether these sequence lengths
(40-300ish frames) are long enough for drift tracking to show a
meaningful difference from the stateless reference.

Usage:
    uv run python -m scripts.compare_drift_vs_stateless_normalization
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from robustness.drift_tracker import DEFAULT_ALPHA, track_sequence
from robustness.normalisation import _DEFAULT_MIN_HEAD_SIZE, compute_head_size
from src.preprocessing.sample_writer import load_sample

NPZ_DIRS = [
    "data/processed/asllrp/train",
    "data/processed/asllrp/val",
]


def compare_one_sample(sample, alpha: float = DEFAULT_ALPHA) -> dict:
    raw_size = compute_head_size(
        sample.face, sample.presence.face, min_head_size=_DEFAULT_MIN_HEAD_SIZE
    )  # (T,), NaN where face absent/degenerate -- this IS the stateless reference
    _tracked_center, tracked_size = track_sequence(sample.face, sample.presence.face, alpha=alpha)

    present_mask = sample.presence.face.astype(bool) & ~np.isnan(raw_size)
    n_compared = int(present_mask.sum())

    if n_compared == 0:
        return {
            "video_id": sample.metadata.video_id,
            "split": sample.metadata.split,
            "num_frames": len(raw_size),
            "n_compared": 0,
            "mean_abs_diff": None,
            "max_abs_diff": None,
            "mean_relative_diff_pct": None,
        }

    diffs = np.abs(tracked_size[present_mask] - raw_size[present_mask])
    mean_diff = float(diffs.mean())
    max_diff = float(diffs.max())
    mean_relative_pct = float((diffs / raw_size[present_mask]).mean() * 100)

    return {
        "video_id": sample.metadata.video_id,
        "split": sample.metadata.split,
        "num_frames": len(raw_size),
        "n_compared": n_compared,
        "mean_abs_diff": mean_diff,
        "max_abs_diff": max_diff,
        "mean_relative_diff_pct": mean_relative_pct,
    }


def main() -> None:
    alpha = DEFAULT_ALPHA
    rows = []
    for npz_dir in NPZ_DIRS:
        for npz_path in sorted(Path(npz_dir).glob("*.npz")):
            sample = load_sample(npz_path)
            rows.append(compare_one_sample(sample, alpha=alpha))

    print(f"=== Drift-tracked vs. stateless head_size reference (alpha={alpha}) ===")
    print(f"{'video_id':<16} {'split':<6} {'T':>5} {'n_cmp':>6} "
          f"{'mean_abs_diff':>14} {'max_abs_diff':>13} {'mean_rel_%':>11}")
    for r in rows:
        if r["n_compared"] == 0:
            print(f"{r['video_id']:<16} {r['split']:<6} {r['num_frames']:>5} "
                  f"{r['n_compared']:>6}      (no face-present frames)")
            continue
        print(f"{r['video_id']:<16} {r['split']:<6} {r['num_frames']:>5} "
              f"{r['n_compared']:>6} {r['mean_abs_diff']:>14.6f} "
              f"{r['max_abs_diff']:>13.6f} {r['mean_relative_diff_pct']:>10.2f}%")

    usable = [r for r in rows if r["n_compared"] > 0]
    mean_abs_diffs = [r["mean_abs_diff"] for r in usable]
    max_abs_diffs = [r["max_abs_diff"] for r in usable]
    mean_rel_pcts = [r["mean_relative_diff_pct"] for r in usable]

    print(f"\nn_samples={len(rows)}  n_usable={len(usable)}")
    print(f"mean(mean_abs_diff) across samples: {np.mean(mean_abs_diffs):.6f}")
    print(f"mean(max_abs_diff) across samples:  {np.mean(max_abs_diffs):.6f}")
    print(f"mean(mean_relative_diff_pct):        {np.mean(mean_rel_pcts):.3f}%")
    print(f"max(max_abs_diff) overall:           {np.max(max_abs_diffs):.6f} "
          f"(sample: {usable[int(np.argmax(max_abs_diffs))]['video_id']})")

    print(
        "\nHONEST READ: sequences here are 40-300ish frames. With the default "
        f"alpha={alpha}, the EWMA's effective averaging window is roughly "
        f"1/alpha = {1/alpha:.0f} frames -- so on a face-present-heavy "
        "sequence, the tracked estimate converges close to the per-frame "
        "stateless value within the first ~10-20 frames and mostly just "
        "smooths high-frequency per-frame jitter afterward, not a genuine "
        "long-horizon 'drift' the stateless reference would have missed. "
        "The real, measurable benefit shown here is COVERAGE (frames the "
        "stateless version marks NaN but drift tracking still normalizes -- "
        "see test_drift_tracking_covers_at_least_as_many_frames_as_stateless), "
        "not a large correction to the reference value itself on sequences "
        "this short. A longer, multi-minute session (the actual Month 3 "
        "motivating scenario) is where sustained camera/pose drift would "
        "diverge further from a single stateless per-frame reference than "
        "anything measurable on these single-utterance clips."
    )


if __name__ == "__main__":
    main()
