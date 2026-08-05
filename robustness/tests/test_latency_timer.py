"""
test_latency_timer.py

Tests for robustness.latency_timer: Stopwatch and the
time_pipeline / time_pipeline_on_synthetic timing wrappers.

No real preprocessed data exists yet this month, so these tests run
entirely against synthetic samples via time_pipeline_on_synthetic,
which builds a synthetic ProcessedSample, round-trips it through disk,
and times load -> hold_last_value -> normalisation on it.
"""

from __future__ import annotations

import numpy as np
import pytest

from robustness.latency_timer import Stopwatch, time_pipeline_on_synthetic
from robustness.synthetic import make_synthetic_sample


def test_stopwatch_measures_nonnegative_elapsed():
    with Stopwatch() as sw:
        sum(range(10000))
    assert sw.elapsed is not None
    assert sw.elapsed >= 0.0


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_time_pipeline_returns_expected_keys(seed):
    timings, normalized = time_pipeline_on_synthetic(num_frames=50, seed=seed)

    for key in (
        "load_time",
        "hold_last_value_time",
        "normalisation_time",
        "total_time",
    ):
        assert key in timings
        assert isinstance(timings[key], float)
        assert timings[key] >= 0.0


def test_time_pipeline_total_at_least_sum_of_stages():
    timings, _ = time_pipeline_on_synthetic(num_frames=50, seed=0)
    stage_sum = (
        timings["load_time"]
        + timings["hold_last_value_time"]
        + timings["normalisation_time"]
    )
    # total_time wraps the same three stages, so it should be >= their sum
    # (equal modulo negligible overhead of the outer perf_counter calls).
    assert timings["total_time"] >= stage_sum - 1e-6


def test_time_pipeline_output_shape_matches_source_recording():
    source = make_synthetic_sample(num_frames=50, seed=0)
    _, normalized = time_pipeline_on_synthetic(num_frames=50, seed=0)
    assert normalized.pose.shape == source.pose.shape
    assert normalized.left_hand.shape == source.left_hand.shape
    assert normalized.right_hand.shape == source.right_hand.shape
    assert normalized.face.shape == source.face.shape


def test_time_pipeline_respects_max_hold_frames():
    timings_a, out_a = time_pipeline_on_synthetic(
        num_frames=50,
        seed=3,
        face_dropout=[10, 11, 12, 13, 14],
        max_hold_frames=0,
    )
    timings_b, out_b = time_pipeline_on_synthetic(
        num_frames=50,
        seed=3,
        face_dropout=[10, 11, 12, 13, 14],
        max_hold_frames=None,
    )
    # Unlimited hold should leave no more NaNs than zero-hold.
    assert np.isnan(out_b.face).sum() <= np.isnan(out_a.face).sum()