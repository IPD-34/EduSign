"""
curate_symmetric_asymmetric_split_sign_level.py

Sibling to curate_symmetric_asymmetric_split.py -- does NOT replace or
modify it. That script computes one motion-ratio symmetry classification
per whole UTTERANCE (a multi-gloss sentence-level clip). But Battison's
Symmetry/Dominance Condition is a property of one SIGN, not a sentence,
and the utterance-level version explicitly flagged this as a real
limitation (a sentence mixing symmetric and asymmetric signs gets
smeared into one aggregate ratio). This script applies the exact same
motion-ratio heuristic (imported, not reimplemented, from
curate_symmetric_asymmetric_split.classify_symmetry) at SIGN granularity
instead, using sign_spans + frame_indices -- already present in
SampleMetadata -- to slice each utterance into per-sign windows first.

Which granularity (utterance vs. sign) is the right one for the C3
ablation is an OPEN QUESTION sent to Person C -- this script exists to
produce the sign-level number so that question can be answered with
real data (specifically: does sign-level granularity get past the
"41 samples, 3 asymmetric utterances" sample-size problem, or not).
Neither this file nor the original is the "final" answer.

--- Per-sign windowing: same remapping convention as
    src/preprocessing/boundary_labels.py's build_boundary_targets() ---
sign_spans entries are (start_frame, end_frame), RAW, 1-INDEXED, in
ORIGINAL VIDEO frame space. frame_indices is the exact
sampled_frame_indices for this utterance -- absolute original-video
frame numbers, aligned 1:1 with each landmark array's T positions. A
sign's window is the set of sampled positions whose frame number falls
within [start_frame - 1, end_frame - 1] (converting to 0-indexed);
signs that collapse to a single raw frame (start == end, a "point
anchor") get a small tolerance window around the nearest sampled
position instead, exactly like build_boundary_targets. This script
reimplements that same mapping (not import-shared with
boundary_labels.py, since build_boundary_targets returns one aggregated
per-timestep array, not the per-sign window breakdown this script
needs) but documents it as the same convention deliberately, so the two
don't silently drift apart.

Usage:
    uv run python -m scripts.curate_symmetric_asymmetric_split_sign_level
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from scripts.curate_symmetric_asymmetric_split import (
    DEFAULT_ASYMMETRIC_RATIO_THRESHOLD,
    DEFAULT_MIN_MOTION_EPS,
    DEFAULT_MIN_PRESENCE_FRACTION,
    DEFAULT_SYMMETRIC_RATIO_THRESHOLD,
    classify_symmetry,
)
from src.preprocessing.sample_writer import load_sample

DEFAULT_NPZ_DIRS = [
    "data/processed/asllrp/train",
    "data/processed/asllrp/val",
]
DEFAULT_OUTPUT_PATH = "data/eval_splits/symmetric_asymmetric_split_sign_level.json"
POINT_ANCHOR_TOLERANCE = 2  # sampled-sequence positions, same default as build_boundary_targets

# --- Minimum-frame reliability threshold for the motion-ratio heuristic ---
#
# classify_symmetry's ratio is min(motion_L, motion_R) / max(motion_L,
# motion_R), where each hand's "motion" is the SUM of frame-to-frame
# centroid displacements over consecutive present frames (see
# curate_symmetric_asymmetric_split._hand_motion) -- i.e. a window of N
# present frames yields (N - 1) displacement terms per hand.
#
# At N=2 (1 displacement term per hand), the ratio compares exactly ONE
# raw frame-to-frame displacement per hand, with zero averaging -- a
# single sample of a quantity dominated by per-frame landmark-tracking
# jitter, not an aggregate motion signal. Whichever hand's single jump
# happens to be marginally larger "wins" the classification with no
# statistical basis for that being the hand's true relative motion.
#
# MIN_RELIABLE_FRAMES = 5 is picked as the point where each hand has at
# least 4 consecutive displacement terms to sum -- some real averaging
# against single-frame jitter, rather than 0-2 terms -- AND matches this
# project's own existing minimum-temporal-unit convention: the
# point-anchor tolerance window (POINT_ANCHOR_TOLERANCE=2, i.e. a
# nearest_pos +/- 2 window = 5 frames) is already the smallest span this
# pipeline treats as a legitimate sign region elsewhere (see
# extract_sign_windows above and build_boundary_targets). Reusing that
# existing anchor keeps this threshold internally consistent with the
# rest of the pipeline instead of introducing a second, unrelated round
# number.
#
# This is stated plainly as a reasoned judgment call, not a rigorously
# derived statistical bound -- there is no measured landmark-jitter
# variance behind it, only the structural argument above (N=2 has zero
# averaging; N=5 has some) plus the existing project convention.
MIN_RELIABLE_FRAMES = 5


def extract_sign_windows(meta) -> tuple[list[dict], dict]:
    """
    Returns (windows, stats) for one sample's metadata.

    windows: list of {"sign_index", "gloss", "position_mask"} -- one per
    sign with a usable span (missing-span or entirely-out-of-range signs
    are excluded, counted in stats instead of silently dropped).

    stats: same category names as build_boundary_targets's stats dict,
    so per-sign span coverage is directly comparable/auditable against
    that module's own reporting.
    """
    stats = {
        "num_signs_total": len(meta.sign_spans) if meta.sign_spans else 0,
        "num_signs_missing_span": 0,
        "num_signs_point_anchor": 0,
        "num_signs_real_span": 0,
        "num_spans_dropped_frame": 0,
        "num_spans_partially_covered": 0,
    }
    windows: list[dict] = []

    if not meta.sign_spans or not meta.frame_indices:
        return windows, stats

    sampled_arr = np.asarray(meta.frame_indices)
    T = len(sampled_arr)
    if T == 0:
        return windows, stats
    lo_bound, hi_bound = sampled_arr.min(), sampled_arr.max()

    glosses = meta.gloss.split("|")
    # sign_spans and the "|"-split gloss list are both built, per sign, in
    # ASLLRPReader's SIGN-element iteration order -- but this is an
    # assumption about upstream ordering, not a guarantee this script can
    # verify from the .npz alone. Only trust the alignment when the counts
    # actually match; otherwise fall back to an index label rather than
    # silently mislabeling a sign with the wrong gloss.
    gloss_count_matches = len(glosses) == len(meta.sign_spans)

    for i, span in enumerate(meta.sign_spans):
        gloss_label = glosses[i] if gloss_count_matches else f"sign_{i}(gloss count mismatch)"

        if span is None:
            stats["num_signs_missing_span"] += 1
            continue

        start_raw, end_raw = span
        start_0 = start_raw - 1
        end_0 = end_raw - 1

        if end_0 < lo_bound or start_0 > hi_bound:
            stats["num_spans_dropped_frame"] += 1
            continue

        if start_0 < lo_bound or end_0 > hi_bound:
            stats["num_spans_partially_covered"] += 1

        if start_raw == end_raw:
            stats["num_signs_point_anchor"] += 1
            nearest_pos = int(np.argmin(np.abs(sampled_arr - start_0)))
            lo = max(0, nearest_pos - POINT_ANCHOR_TOLERANCE)
            hi = min(T, nearest_pos + POINT_ANCHOR_TOLERANCE + 1)
            mask = np.zeros(T, dtype=bool)
            mask[lo:hi] = True
        else:
            stats["num_signs_real_span"] += 1
            mask = (sampled_arr >= start_0) & (sampled_arr <= end_0)

        if mask.any():
            windows.append({"sign_index": i, "gloss": gloss_label, "position_mask": mask})
        else:
            # Span passed the range check but produced an empty mask
            # (shouldn't happen given the checks above, but guarded
            # rather than silently emitting a window with zero frames).
            stats["num_spans_dropped_frame"] += 1

    return windows, stats


def build_sign_level_split(
    npz_dirs: list[str],
    symmetric_ratio_threshold: float = DEFAULT_SYMMETRIC_RATIO_THRESHOLD,
    asymmetric_ratio_threshold: float = DEFAULT_ASYMMETRIC_RATIO_THRESHOLD,
    min_presence_fraction: float = DEFAULT_MIN_PRESENCE_FRACTION,
    min_motion_eps: float = DEFAULT_MIN_MOTION_EPS,
) -> tuple[dict, dict]:
    results = {}
    aggregate_span_stats = Counter()
    samples_missing_sign_spans = []
    samples_processed = 0

    for npz_dir in npz_dirs:
        npz_dir_path = Path(npz_dir)
        split_name = npz_dir_path.name
        for npz_path in sorted(npz_dir_path.glob("*.npz")):
            sample = load_sample(npz_path)
            meta = sample.metadata
            samples_processed += 1

            if meta.sign_spans is None or meta.frame_indices is None:
                samples_missing_sign_spans.append(meta.video_id)
                continue

            windows, stats = extract_sign_windows(meta)
            for key, value in stats.items():
                aggregate_span_stats[key] += value

            for window in windows:
                mask = window["position_mask"]

                classification = classify_symmetry(
                    sample.left_hand[mask],
                    sample.right_hand[mask],
                    sample.presence.left_hand[mask],
                    sample.presence.right_hand[mask],
                    symmetric_ratio_threshold=symmetric_ratio_threshold,
                    asymmetric_ratio_threshold=asymmetric_ratio_threshold,
                    min_presence_fraction=min_presence_fraction,
                    min_motion_eps=min_motion_eps,
                )

                num_frames_in_window = int(mask.sum())
                reliability = (
                    "reliable"
                    if num_frames_in_window >= MIN_RELIABLE_FRAMES
                    else "low_confidence_short_window"
                )

                result_key = f"{meta.video_id}::sign{window['sign_index']:02d}"
                results[result_key] = {
                    "video_id": meta.video_id,
                    "sign_index": window["sign_index"],
                    "gloss": window["gloss"],
                    "category": classification["category"],
                    "source": "heuristic_sign_level",
                    "evidence": classification["evidence"],
                    "num_frames_in_window": num_frames_in_window,
                    "reliability": reliability,
                    "split": split_name,
                }

    meta_report = {
        "samples_processed": samples_processed,
        "samples_missing_sign_spans": samples_missing_sign_spans,
        "aggregate_span_stats": dict(aggregate_span_stats),
    }
    return results, meta_report


def _percentile(sorted_values: list[int], pct: float) -> float:
    """Linear-interpolation percentile (numpy-equivalent), no numpy dependency."""
    if not sorted_values:
        return float("nan")
    n = len(sorted_values)
    if n == 1:
        return float(sorted_values[0])
    rank = (pct / 100.0) * (n - 1)
    lo = int(rank)
    hi = min(lo + 1, n - 1)
    frac = rank - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def print_frame_count_distribution(sorted_frame_counts: list[int]) -> None:
    """
    Full distribution of num_frames_in_window across all windows --
    percentiles plus a bucketed histogram -- not just min/median/max.
    """
    n = len(sorted_frame_counts)
    print(f"\nFull num_frames_in_window distribution (n={n} windows):")
    for pct in (1, 5, 10, 25, 50, 75, 90, 95, 99):
        print(f"  p{pct:<3}: {_percentile(sorted_frame_counts, pct):.1f}")
    mean = sum(sorted_frame_counts) / n
    print(f"  mean : {mean:.2f}")

    bucket_edges = [
        (2, 2), (3, 3), (4, 4), (5, 5), (6, 10), (11, 20), (21, sorted_frame_counts[-1]),
    ]
    print("  histogram:")
    for lo, hi in bucket_edges:
        count = sum(1 for f in sorted_frame_counts if lo <= f <= hi)
        label = f"{lo}" if lo == hi else f"{lo}-{hi}"
        bar = "#" * max(1, round(count / n * 60)) if count else ""
        print(f"    {label:>7}: {count:>4}  {bar}")


def print_reliability_breakdown(results: dict) -> None:
    """
    Per-category reliable vs. low_confidence_short_window counts, so the
    headline category counts (e.g. '125 asymmetric') are never reported
    without their confidence composition alongside.
    """
    print(
        f"\nReliability threshold: num_frames_in_window >= {MIN_RELIABLE_FRAMES} "
        f"-> 'reliable', else 'low_confidence_short_window' (see MIN_RELIABLE_FRAMES "
        f"docstring above for the justification)."
    )
    print("Per-category reliability breakdown:")
    for category in ("symmetric", "asymmetric", "ambiguous"):
        cat_entries = [e for e in results.values() if e["category"] == category]
        total = len(cat_entries)
        reliable = sum(1 for e in cat_entries if e["reliability"] == "reliable")
        low_conf = total - reliable
        low_conf_pct = (low_conf / total * 100) if total else 0.0
        print(
            f"  {category:<10}: {total:>3} total -- {reliable:>3} reliable, "
            f"{low_conf:>3} low_confidence_short_window ({low_conf_pct:.1f}%)"
        )
    total_all = len(results)
    total_low_conf = sum(1 for e in results.values() if e["reliability"] == "low_confidence_short_window")
    print(
        f"  {'ALL':<10}: {total_all:>3} total -- {total_all - total_low_conf:>3} reliable, "
        f"{total_low_conf:>3} low_confidence_short_window "
        f"({(total_low_conf / total_all * 100) if total_all else 0.0:.1f}%)"
    )


def print_summary(results: dict, meta_report: dict) -> None:
    total_windows = len(results)
    counts = {"symmetric": 0, "asymmetric": 0, "ambiguous": 0}
    for entry in results.values():
        counts[entry["category"]] += 1

    print("=== Sign-level symmetric/asymmetric curation summary (PROTOTYPE) ===")
    print(f"Samples processed: {meta_report['samples_processed']}")

    missing = meta_report["samples_missing_sign_spans"]
    if missing:
        print(
            f"\n*** {len(missing)}/{meta_report['samples_processed']} samples have NO "
            f"sign_spans/frame_indices metadata at all and were EXCLUDED from "
            f"sign-level windowing entirely: {missing} ***"
        )
    else:
        print("\nAll processed samples had sign_spans/frame_indices metadata present.")

    span_stats = meta_report["aggregate_span_stats"]
    print("\nAggregate per-sign span coverage (across all samples with sign_spans present):")
    for key, value in span_stats.items():
        print(f"  {key}: {value}")
    if span_stats.get("num_spans_dropped_frame", 0) or span_stats.get("num_signs_missing_span", 0):
        print(
            "  -> NOTE: signs counted under num_signs_missing_span or "
            "num_spans_dropped_frame produced NO sign-level window and are "
            "NOT represented in the classification counts below -- this is "
            "real coverage loss, not zero-cost."
        )

    print(f"\nTOTAL SIGN-LEVEL WINDOWS PRODUCED: {total_windows}")
    print("(this is the number that matters for the sample-size question --")
    print(" compare against the utterance-level script's 41 total utterances)")
    for category in ("symmetric", "asymmetric", "ambiguous"):
        pct = (counts[category] / total_windows * 100) if total_windows else 0.0
        print(f"  {category}: {counts[category]} ({pct:.1f}%)")

    frame_counts = sorted(entry["num_frames_in_window"] for entry in results.values())
    if frame_counts:
        print(
            f"\nPer-window frame-count range: min={frame_counts[0]}, "
            f"median={frame_counts[len(frame_counts) // 2]}, max={frame_counts[-1]}"
        )
        print_frame_count_distribution(frame_counts)
        print_reliability_breakdown(results)

    print(
        "\nGRANULARITY CAVEAT: this is a PROTOTYPE, same unconfirmed status as "
        "curate_symmetric_asymmetric_split.py's utterance-level output. Whether "
        "sign-level or utterance-level granularity is the correct unit for the "
        "C3 ablation is an OPEN QUESTION sent to Person C -- do not treat either "
        "file as the final eval subset until that's resolved."
    )
    print(
        "HEURISTIC CAVEAT: 'symmetric'/'asymmetric' labels are still a "
        "motion-ratio proxy, not linguistic ground truth, and now additionally "
        "depend on a frame-window remapping (sign_spans -> sampled positions) "
        "that has not been human-verified for correctness. Every label here "
        "needs spot-checking before being used as an authoritative split, more "
        "so than the utterance-level version."
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

    results, meta_report = build_sign_level_split(
        args.npz_dirs,
        symmetric_ratio_threshold=args.symmetric_ratio_threshold,
        asymmetric_ratio_threshold=args.asymmetric_ratio_threshold,
        min_presence_fraction=args.min_presence_fraction,
        min_motion_eps=args.min_motion_eps,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"windows": results, "meta": meta_report}, f, indent=2)
    print(f"Wrote {len(results)} sign-level classifications to {output_path}\n")

    print_summary(results, meta_report)


if __name__ == "__main__":
    main()
