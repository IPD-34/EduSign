"""
extract_real_dropout_stats.py

Characterizes real tracking-dropout patterns from the actual ASLLRP
samples currently loading cleanly through src/models/dataset.py
(SignLanguageDataset -- 20 train + 21 val samples, 41 total), so the
resulting stats can be fed into robustness/synthetic.py's dropout specs
as realistic parameters instead of guessed ones.

Missingness contract (see src/preprocessing/sample_schema.py): each of
pose/face/left_hand/right_hand carries a separate (T,) boolean
LandmarkPresence array. False means "absent at this frame" -- the
landmark values themselves are zero-filled (not NaN) at that frame, so
presence is the only reliable missingness signal; this script reads
presence directly and never NaN-scans the landmark arrays.

Reports, per channel (pose/face/left_hand/right_hand), aggregated across
all 41 samples:
  1. Fraction of timesteps with presence == False.
  2. Run-length distribution of consecutive missing-timestep spans (min /
     median / max / histogram), not just the overall fraction.
  3. Co-occurrence: what fraction of a channel's missing SPANS overlap,
     at at least one frame, with another channel also being missing
     (vs. spans where this channel is the only one missing throughout).
     A per-frame "how many channels missing at once" histogram is
     included alongside as a complementary, finer-grained view.
  4. Which samples have a channel missing for the ENTIRE sample (total
     dropout) vs. only part of it (partial) vs. never.

Usage:
    uv run python -m scripts.extract_real_dropout_stats
"""

from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path

import numpy as np

from src.preprocessing.sample_writer import load_sample

NPZ_DIRS = [
    "data/processed/asllrp/train",
    "data/processed/asllrp/val",
]
OUTPUT_PATH = "data/real_dropout_stats.json"

CHANNELS = ["pose", "face", "left_hand", "right_hand"]

# Bucket edges for the printed run-length histogram (inclusive upper bound
# per bucket, last bucket is open-ended).
_HIST_BUCKETS = [1, 2, 5, 10, 20, 50, 100]


def _extract_run_lengths(missing: np.ndarray) -> list[int]:
    """Lengths of consecutive True (missing) runs in a 1-D bool array."""
    runs = []
    length = 0
    for v in missing:
        if v:
            length += 1
        else:
            if length > 0:
                runs.append(length)
            length = 0
    if length > 0:
        runs.append(length)
    return runs


def _extract_run_spans(missing: np.ndarray) -> list[tuple[int, int]]:
    """(start, end) inclusive frame-index pairs for each consecutive True run."""
    spans = []
    start = None
    for i, v in enumerate(missing):
        if v and start is None:
            start = i
        elif not v and start is not None:
            spans.append((start, i - 1))
            start = None
    if start is not None:
        spans.append((start, len(missing) - 1))
    return spans


def _bucket_label(run_length: int) -> str:
    for edge in _HIST_BUCKETS:
        if run_length <= edge:
            prev_edges = [0] + _HIST_BUCKETS
            lower = prev_edges[_HIST_BUCKETS.index(edge)] + 1
            return f"{lower}" if lower == edge else f"{lower}-{edge}"
    return f"{_HIST_BUCKETS[-1] + 1}+"


def _histogram(run_lengths: list[int]) -> dict:
    counts = Counter(_bucket_label(rl) for rl in run_lengths)
    # Preserve a stable, human-readable bucket order.
    ordered_labels = []
    prev = 0
    for edge in _HIST_BUCKETS:
        label = f"{prev + 1}" if prev + 1 == edge else f"{prev + 1}-{edge}"
        ordered_labels.append(label)
        prev = edge
    ordered_labels.append(f"{_HIST_BUCKETS[-1] + 1}+")
    return {label: counts.get(label, 0) for label in ordered_labels}


def _run_length_stats(run_lengths: list[int]) -> dict:
    if not run_lengths:
        return {"count": 0, "min": None, "median": None, "max": None, "mean": None}
    return {
        "count": len(run_lengths),
        "min": min(run_lengths),
        "median": statistics.median(run_lengths),
        "max": max(run_lengths),
        "mean": round(statistics.mean(run_lengths), 3),
    }


def collect_stats(npz_dirs: list[str]) -> dict:
    npz_paths = []
    for npz_dir in npz_dirs:
        npz_paths.extend(sorted(Path(npz_dir).glob("*.npz")))

    per_channel_missing_timesteps = {c: 0 for c in CHANNELS}
    per_channel_total_timesteps = {c: 0 for c in CHANNELS}
    per_channel_run_lengths = {c: [] for c in CHANNELS}
    per_channel_fully_missing_samples = {c: [] for c in CHANNELS}
    per_channel_partially_missing_samples = {c: 0 for c in CHANNELS}
    per_channel_never_missing_samples = {c: 0 for c in CHANNELS}

    co_occurring_spans = 0
    isolated_spans = 0
    frame_level_missing_count_histogram = Counter()  # num channels missing -> frame count

    num_samples = 0
    for npz_path in npz_paths:
        sample = load_sample(npz_path)
        num_samples += 1
        video_id = sample.metadata.video_id

        presence_by_channel = {
            "pose": np.asarray(sample.presence.pose, dtype=bool),
            "face": np.asarray(sample.presence.face, dtype=bool),
            "left_hand": np.asarray(sample.presence.left_hand, dtype=bool),
            "right_hand": np.asarray(sample.presence.right_hand, dtype=bool),
        }
        missing_by_channel = {c: ~p for c, p in presence_by_channel.items()}

        T = len(presence_by_channel["pose"])

        # --- per-frame co-occurrence histogram ---
        missing_count_per_frame = np.zeros(T, dtype=int)
        for c in CHANNELS:
            missing_count_per_frame += missing_by_channel[c].astype(int)
        for count in missing_count_per_frame:
            frame_level_missing_count_histogram[int(count)] += 1

        for c in CHANNELS:
            missing = missing_by_channel[c]
            per_channel_total_timesteps[c] += T
            per_channel_missing_timesteps[c] += int(missing.sum())

            frac_missing = missing.mean() if T > 0 else 0.0
            if frac_missing >= 1.0:
                per_channel_fully_missing_samples[c].append(video_id)
            elif frac_missing > 0.0:
                per_channel_partially_missing_samples[c] += 1
            else:
                per_channel_never_missing_samples[c] += 1

            run_lengths = _extract_run_lengths(missing)
            per_channel_run_lengths[c].extend(run_lengths)

            # --- span-level co-occurrence: does ANY other channel overlap
            # this channel's missing span at at least one frame? ---
            spans = _extract_run_spans(missing)
            other_channels = [oc for oc in CHANNELS if oc != c]
            for start, end in spans:
                overlap = any(
                    missing_by_channel[oc][start : end + 1].any() for oc in other_channels
                )
                if overlap:
                    co_occurring_spans += 1
                else:
                    isolated_spans += 1

    per_channel_report = {}
    for c in CHANNELS:
        total = per_channel_total_timesteps[c]
        missing = per_channel_missing_timesteps[c]
        run_lengths = per_channel_run_lengths[c]
        per_channel_report[c] = {
            "total_timesteps": total,
            "missing_timesteps": missing,
            "fraction_missing": (missing / total) if total else 0.0,
            "run_lengths": run_lengths,
            "run_length_stats": _run_length_stats(run_lengths),
            "run_length_histogram": _histogram(run_lengths),
            "samples_fully_missing": per_channel_fully_missing_samples[c],
            "samples_partially_missing_count": per_channel_partially_missing_samples[c],
            "samples_never_missing_count": per_channel_never_missing_samples[c],
        }

    total_spans = co_occurring_spans + isolated_spans
    co_occurrence_report = {
        "total_spans": total_spans,
        "co_occurring_spans": co_occurring_spans,
        "isolated_spans": isolated_spans,
        "fraction_co_occurring": (co_occurring_spans / total_spans) if total_spans else None,
        "frame_level_num_channels_missing_histogram": {
            str(k): v for k, v in sorted(frame_level_missing_count_histogram.items())
        },
    }

    return {
        "num_samples": num_samples,
        "per_channel": per_channel_report,
        "co_occurrence": co_occurrence_report,
    }


def print_summary(stats: dict) -> None:
    print("=== Real tracking-dropout stats: "
          f"{stats['num_samples']} ASLLRP samples (20 train + 21 val) ===\n")

    header = f"{'channel':<12} {'% missing':>10} {'#runs':>7} {'min':>5} {'median':>7} {'max':>5} {'full-drop':>10} {'partial':>8} {'never':>6}"
    print(header)
    print("-" * len(header))
    for c in CHANNELS:
        entry = stats["per_channel"][c]
        rls = entry["run_length_stats"]
        pct = entry["fraction_missing"] * 100
        print(
            f"{c:<12} {pct:>9.2f}% {rls['count']:>7} "
            f"{str(rls['min']):>5} {str(rls['median']):>7} {str(rls['max']):>5} "
            f"{len(entry['samples_fully_missing']):>10} "
            f"{entry['samples_partially_missing_count']:>8} "
            f"{entry['samples_never_missing_count']:>6}"
        )
    print()

    print("Run-length histograms (burst length in consecutive missing frames):")
    for c in CHANNELS:
        hist = stats["per_channel"][c]["run_length_histogram"]
        hist_str = ", ".join(f"{label}:{count}" for label, count in hist.items() if count)
        print(f"  {c:<12} {hist_str if hist_str else '(no missing spans)'}")
    print()

    for c in CHANNELS:
        fully = stats["per_channel"][c]["samples_fully_missing"]
        if fully:
            print(f"Samples with {c} missing for the ENTIRE sample: {fully}")
    print()

    co = stats["co_occurrence"]
    if co["total_spans"]:
        print(
            f"Co-occurrence: {co['co_occurring_spans']}/{co['total_spans']} "
            f"({co['fraction_co_occurring'] * 100:.1f}%) of missing spans overlap "
            f"with at least one other channel also missing; "
            f"{co['isolated_spans']}/{co['total_spans']} are isolated to a single channel."
        )
    else:
        print("Co-occurrence: no missing spans found in this data.")
    print(
        "Per-frame count of simultaneously-missing channels "
        f"(0-4): {co['frame_level_num_channels_missing_histogram']}"
    )


def main() -> None:
    stats = collect_stats(NPZ_DIRS)

    output_path = Path(OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print_summary(stats)
    print(f"\nRaw per-channel run-length arrays + full stats written to {output_path}")


if __name__ == "__main__":
    main()
