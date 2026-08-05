"""
latency_timer.py

Measures per-stage latency (file loading, hold-last-value processing,
normalisation, and total) for the Month 1 pipeline, using
time.perf_counter().

No model / confidence-estimation / evaluation code here -- purely timing
instrumentation around the other robustness/ modules.

Since no real preprocessed .npz data exists yet this month (WLASL/ASLLRP
preprocessing is still in progress on another lane), this module also
provides time_pipeline_on_synthetic(), which generates a synthetic
ProcessedSample, round-trips it through disk, and times the pipeline on
that -- so this script is runnable standalone right now.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from robustness.data_io import load_recording
from robustness.hold_last_value import hold_last_value_sample
from robustness.normalisation import normalize_landmarks_sample
from robustness.synthetic import make_synthetic_sample, round_trip_sample
from src.preprocessing.sample_schema import ProcessedSample


class Stopwatch:
    """
    Minimal reusable timing context manager built on time.perf_counter().

        with Stopwatch() as sw:
            do_work()
        print(sw.elapsed)
    """

    def __enter__(self) -> "Stopwatch":
        self._start = time.perf_counter()
        self.elapsed: Optional[float] = None
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.elapsed = time.perf_counter() - self._start


def time_pipeline(
    sample_path: Union[Path, str],
    *,
    max_hold_frames: Optional[int] = None,
    min_head_size: float = 1e-4,
) -> Tuple[Dict[str, float], ProcessedSample]:
    """
    Run load -> hold_last_value_sample -> normalize_landmarks_sample on
    one .npz file, timing each stage with time.perf_counter().

    Parameters
    ----------
    sample_path:
        Path to a single .npz file (as written by
        src.preprocessing.sample_writer.write_sample), not a
        directory+name pair.
    max_hold_frames:
        Passed through to hold_last_value_sample.
    min_head_size:
        Passed through to normalize_landmarks_sample.

    Returns
    -------
    timings: dict with keys 'load_time', 'hold_last_value_time',
             'normalisation_time', 'total_time' (all floats, seconds).
    normalized_sample: the final ProcessedSample pipeline output.
    """
    sample_path = Path(sample_path)
    total_start = time.perf_counter()

    with Stopwatch() as sw_load:
        sample = load_recording(sample_path)
    load_time = sw_load.elapsed

    with Stopwatch() as sw_hold:
        filled = hold_last_value_sample(sample, max_hold_frames=max_hold_frames)
    hold_time = sw_hold.elapsed

    with Stopwatch() as sw_norm:
        normalized = normalize_landmarks_sample(filled, min_head_size=min_head_size)
    norm_time = sw_norm.elapsed

    total_time = time.perf_counter() - total_start

    timings = {
        "load_time": load_time,
        "hold_last_value_time": hold_time,
        "normalisation_time": norm_time,
        "total_time": total_time,
    }
    return timings, normalized


def time_pipeline_on_synthetic(
    num_frames: int = 100,
    seed: int = 0,
    tmp_dir: Optional[Path] = None,
    *,
    pose_dropout: Any = None,
    left_hand_dropout: Any = None,
    right_hand_dropout: Any = None,
    face_dropout: Any = None,
    **kwargs: Any,
) -> Tuple[Dict[str, float], ProcessedSample]:
    """
    ...
    pose_dropout, left_hand_dropout, right_hand_dropout, face_dropout:
        Per-group dropout specs, forwarded to make_synthetic_sample (NOT
        to time_pipeline -- these control the synthetic input, not the
        pipeline call).
    **kwargs:
        Forwarded to time_pipeline only (e.g. max_hold_frames,
        min_head_size).
    """
    synthetic_sample = make_synthetic_sample(
        num_frames=num_frames,
        seed=seed,
        pose_dropout=pose_dropout,
        left_hand_dropout=left_hand_dropout,
        right_hand_dropout=right_hand_dropout,
        face_dropout=face_dropout,
    )

    def _write_and_time(directory):
        round_trip_sample(synthetic_sample, directory)
        sample_path = Path(directory) / "synthetic_sample.npz"
        return time_pipeline(sample_path, **kwargs)

    if tmp_dir is not None:
        return _write_and_time(tmp_dir)

    with tempfile.TemporaryDirectory() as tmp_dir_str:
        return _write_and_time(tmp_dir_str)