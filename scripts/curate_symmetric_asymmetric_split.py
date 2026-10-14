"""
curate_symmetric_asymmetric_split.py

C3 (bimanual asymmetry-routed attention) needs an evaluation subset split
into symmetric vs. asymmetric two-hand sign configurations (Battison's
Dominance and Symmetry Conditions), so the ablation can be scored
separately on each subset.

--- Step 1 finding: no existing metadata field for this ---
Checked before writing anything: src/preprocessing/sample_schema.py's
SampleMetadata (video_id, gloss, split, source, fps, num_frames,
translation, frame_indices, sign_spans) has NO handedness/symmetry/
sign_type field, and neither do the actual .npz files on disk (confirmed
by loading a real sample and inspecting its metadata directly). The
SIGN_TYPE categories (Lexical/Fingerspelled/Loan/Gestures/Classifiers/
Number Signs) mentioned in boundary_labels.py's docstring, and the
DOMINANT_HAND/NON_DOMINANT_HAND per-sign spans that same docstring
references, come from ASLLRP's raw SignStream XML -- but:
  - src/preprocessing/metadata_readers/asllrp_reader.py only extracts
    LABEL (gloss) per SIGN element; it does not parse SIGN_TYPE,
    DOMINANT_HAND, or NON_DOMINANT_HAND attributes at all.
  - The raw XML files themselves are not present anywhere in this repo
    (only the already-processed .npz outputs are) -- so even a reader
    change couldn't backfill this for the current 41 samples without
    the original ASLLRP XML/video source, which is out of this task's
    reach ("data already in the repo").
So this is NOT a "read an existing field" task -- it's the heuristic
path (step 3 in the request).

--- Heuristic (first-pass, NOT ground truth) ---
Per-hand "motion" proxy: for each hand, sum the frame-to-frame Euclidean
displacement of that hand's centroid (mean of its 21 landmarks) across
consecutive PRESENT frames only (occlusion gaps are skipped, not treated
as zero-displacement, so a hand that's briefly occluded mid-sign doesn't
get penalized as "static"). This is total path length, not just
positional variance -- path length also captures an alternating/
oscillating symmetric motion (e.g. clapping-type signs), which raw
variance-around-the-mean can understate for a hand that moves back and
forth through the same region.

  ratio = min(motion_L, motion_R) / max(motion_L, motion_R)

  ratio >= SYMMETRIC_RATIO_THRESHOLD  -> "symmetric" (both hands move
      comparably)
  ratio <= ASYMMETRIC_RATIO_THRESHOLD -> "asymmetric" (one hand moves
      much more than the other -- consistent with a dominant hand
      articulating against a static/base non-dominant hand)
  otherwise                           -> "ambiguous" (needs human
      spot-check)

Also marked "ambiguous" (with an explicit reason, not silently folded
into a bucket) when:
  - either hand's presence fraction across the utterance is too low to
    trust the motion estimate (MIN_PRESENCE_FRACTION)
  - both hands' motion is negligible (MIN_MOTION_EPS) -- ratio is
    undefined/unstable when the denominator is near zero, so this
    isn't reported as a spurious ratio near 1.0

These thresholds are explicitly first-pass and tunable via CLI flags --
they are NOT validated against any human-labeled ground truth yet. Every
non-ambiguous label from this script should still be spot-checked before
being treated as authoritative for the ablation.

Usage:
    uv run python -m scripts.curate_symmetric_asymmetric_split
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.preprocessing.sample_writer import load_sample

DEFAULT_NPZ_DIRS = [
    "data/processed/asllrp/train",
    "data/processed/asllrp/val",
]
DEFAULT_OUTPUT_PATH = "data/eval_splits/symmetric_asymmetric_split.json"

DEFAULT_SYMMETRIC_RATIO_THRESHOLD = 0.6
DEFAULT_ASYMMETRIC_RATIO_THRESHOLD = 0.3
DEFAULT_MIN_PRESENCE_FRACTION = 0.3
DEFAULT_MIN_MOTION_EPS = 1e-4


def _hand_motion(hand: np.ndarray, presence: np.ndarray) -> float:
    """
    Total path length of the hand's centroid across consecutive PRESENT
    frames. hand: (T, 21, 3). presence: (T,) bool. Gaps in presence are
    skipped (not bridged as zero-displacement, not bridged across the
    gap either) -- each maximal run of present frames contributes its
    own internal consecutive displacements only.
    """
    centroid = hand.mean(axis=1)  # (T, 3)
    present_idx = np.flatnonzero(presence)
    if len(present_idx) < 2:
        return 0.0

    motion = 0.0
    for a, b in zip(present_idx[:-1], present_idx[1:]):
        if b == a + 1:  # consecutive present frames only
            motion += float(np.linalg.norm(centroid[b] - centroid[a]))
    return motion


def classify_symmetry(
    left_hand: np.ndarray,
    right_hand: np.ndarray,
    presence_left: np.ndarray,
    presence_right: np.ndarray,
    symmetric_ratio_threshold: float = DEFAULT_SYMMETRIC_RATIO_THRESHOLD,
    asymmetric_ratio_threshold: float = DEFAULT_ASYMMETRIC_RATIO_THRESHOLD,
    min_presence_fraction: float = DEFAULT_MIN_PRESENCE_FRACTION,
    min_motion_eps: float = DEFAULT_MIN_MOTION_EPS,
) -> dict:
    """
    Returns {"category": ..., "evidence": {...}} for one sample. category
    is "symmetric", "asymmetric", or "ambiguous" (evidence always includes
    a "reason" key when ambiguous).
    """
    presence_fraction_left = float(presence_left.mean()) if len(presence_left) else 0.0
    presence_fraction_right = float(presence_right.mean()) if len(presence_right) else 0.0

    motion_left = _hand_motion(left_hand, presence_left)
    motion_right = _hand_motion(right_hand, presence_right)

    evidence = {
        "motion_left": motion_left,
        "motion_right": motion_right,
        "presence_fraction_left": presence_fraction_left,
        "presence_fraction_right": presence_fraction_right,
    }

    if presence_fraction_left < min_presence_fraction or presence_fraction_right < min_presence_fraction:
        evidence["ratio"] = None
        evidence["reason"] = (
            "insufficient_hand_presence: presence_fraction_left="
            f"{presence_fraction_left:.2f}, presence_fraction_right="
            f"{presence_fraction_right:.2f} (min required: {min_presence_fraction})"
        )
        return {"category": "ambiguous", "evidence": evidence}

    if max(motion_left, motion_right) < min_motion_eps:
        evidence["ratio"] = None
        evidence["reason"] = (
            f"negligible_motion_both_hands: motion_left={motion_left:.6f}, "
            f"motion_right={motion_right:.6f} (min: {min_motion_eps})"
        )
        return {"category": "ambiguous", "evidence": evidence}

    ratio = min(motion_left, motion_right) / max(motion_left, motion_right)
    evidence["ratio"] = ratio

    if ratio >= symmetric_ratio_threshold:
        return {"category": "symmetric", "evidence": evidence}
    if ratio <= asymmetric_ratio_threshold:
        return {"category": "asymmetric", "evidence": evidence}

    evidence["reason"] = (
        f"ratio {ratio:.3f} falls between asymmetric threshold "
        f"({asymmetric_ratio_threshold}) and symmetric threshold "
        f"({symmetric_ratio_threshold})"
    )
    return {"category": "ambiguous", "evidence": evidence}


def build_split(
    npz_dirs: list[str],
    symmetric_ratio_threshold: float = DEFAULT_SYMMETRIC_RATIO_THRESHOLD,
    asymmetric_ratio_threshold: float = DEFAULT_ASYMMETRIC_RATIO_THRESHOLD,
    min_presence_fraction: float = DEFAULT_MIN_PRESENCE_FRACTION,
    min_motion_eps: float = DEFAULT_MIN_MOTION_EPS,
) -> dict:
    results = {}
    for npz_dir in npz_dirs:
        npz_dir_path = Path(npz_dir)
        split_name = npz_dir_path.name  # "train" or "val"
        for npz_path in sorted(npz_dir_path.glob("*.npz")):
            sample = load_sample(npz_path)
            meta = sample.metadata

            classification = classify_symmetry(
                sample.left_hand,
                sample.right_hand,
                sample.presence.left_hand,
                sample.presence.right_hand,
                symmetric_ratio_threshold=symmetric_ratio_threshold,
                asymmetric_ratio_threshold=asymmetric_ratio_threshold,
                min_presence_fraction=min_presence_fraction,
                min_motion_eps=min_motion_eps,
            )

            results[meta.video_id] = {
                "category": classification["category"],
                "source": "heuristic",
                "evidence": classification["evidence"],
                "split": split_name,
                "gloss": meta.gloss,
            }

    return results


def print_summary(results: dict) -> None:
    total = len(results)
    counts = {"symmetric": 0, "asymmetric": 0, "ambiguous": 0}
    by_split = {}
    for video_id, entry in results.items():
        counts[entry["category"]] += 1
        by_split.setdefault(entry["split"], {"symmetric": 0, "asymmetric": 0, "ambiguous": 0})
        by_split[entry["split"]][entry["category"]] += 1

    print("=== Symmetric/asymmetric curation summary ===")
    print(f"Total samples classified: {total}")
    for category in ("symmetric", "asymmetric", "ambiguous"):
        pct = (counts[category] / total * 100) if total else 0.0
        print(f"  {category}: {counts[category]} ({pct:.1f}%)")
    print("By split:")
    for split_name, split_counts in sorted(by_split.items()):
        print(f"  {split_name}: {split_counts}")

    print(
        "\nSAMPLE-SIZE CAVEAT: only 41 total samples are available "
        "(20 train + 21 val) under data/processed/asllrp/. Once split "
        "three ways (symmetric / asymmetric / ambiguous), each bucket "
        "will typically hold well under 20 samples -- too small to "
        "measure 'a directional accuracy gain on the asymmetric subset "
        "with no regression on the symmetric subset' with any "
        "statistical confidence. More raw ASLLRP data should be pulled "
        "and preprocessed into data/processed/asllrp/ before this split "
        "is used as the actual C3 ablation evaluation set -- this run "
        "should be treated as a pipeline dry run on the currently "
        "available pool, not a sufficient final eval subset."
    )
    print(
        "\nHEURISTIC CAVEAT: 'symmetric'/'asymmetric' labels here are a "
        "motion-based proxy, not linguistic ground truth. Every category "
        "(including non-ambiguous ones) should be human spot-checked "
        "before being used as the authoritative C3 ablation split."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz-dirs", nargs="+", default=DEFAULT_NPZ_DIRS)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--symmetric-ratio-threshold", type=float, default=DEFAULT_SYMMETRIC_RATIO_THRESHOLD)
    parser.add_argument("--asymmetric-ratio-threshold", type=float, default=DEFAULT_ASYMMETRIC_RATIO_THRESHOLD)
    parser.add_argument("--min-presence-fraction", type=float, default=DEFAULT_MIN_PRESENCE_FRACTION)
    parser.add_argument("--min-motion-eps", type=float, default=DEFAULT_MIN_MOTION_EPS)
    args = parser.parse_args()

    results = build_split(
        args.npz_dirs,
        symmetric_ratio_threshold=args.symmetric_ratio_threshold,
        asymmetric_ratio_threshold=args.asymmetric_ratio_threshold,
        min_presence_fraction=args.min_presence_fraction,
        min_motion_eps=args.min_motion_eps,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {len(results)} sample classifications to {output_path}\n")

    print_summary(results)


if __name__ == "__main__":
    main()
