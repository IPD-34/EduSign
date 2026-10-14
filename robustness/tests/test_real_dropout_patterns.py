"""
test_real_dropout_patterns.py

Month 2 plan deliverable: "re-run month-1 robustness unit tests against
real dropout patterns observed in actual footage." Month 1's
hold_last_value.py / normalisation.py tests only ever used synthetic
fixtures (make_synthetic_sample, generate_realistic_dropout, hand-built
tiny arrays) -- this file runs the same no-NaN / no-inf / correct-fill
guarantees directly against the 41 REAL ASLLRP .npz samples under
data/processed/asllrp/{train,val}/, parametrized per sample so CI reports
exactly which real sample (if any) fails, not just a bulk pass/fail.

Why this is a meaningfully different test than test_robustness_real_patterns.py
(also in this suite): that file tests generate_realistic_dropout, which
SAMPLES synthetic data from the real per-channel run-length/co-occurrence
DISTRIBUTION -- statistically similar to real data, but not the actual
real samples. This file loads the real .npz files directly.

Real dropout characteristics found (see the docstrings on individual test
functions below for the specific numbers each one checks), summarized
here for reference:
  - pose:       0.57% missing, max gap 2 frames
  - face:       0.73% missing, max gap 4 frames
  - left_hand:  7.27% missing, max gap 22 frames (asllrp_1169) <-- by far
                the longest real gap, and dramatically longer than
                anything Month 1's synthetic tests exercised (see below)
  - right_hand: 4.56% missing, max gap 7 frames
  - 31 frames (spread across 30/41 samples, one sample has 2) are a
    GENUINE LEADING gap: all four groups missing simultaneously at the
    very start of the clip, before tracking locks on -- confirmed these
    are the same 31 "4 channels missing at once" frames found by
    scripts/extract_real_dropout_stats.py's co-occurrence analysis.

Month 1 coverage gap this exposes (finding, not a bug -- see task report):
  - test_hold_last_value.py's longest explicit contiguous gap was 5 frames
    (left_hand_dropout=[1,2,3,4,5]); real data reaches 22 frames (4.4x
    longer) and was never exercised against this function before now.
  - test_normalisation.py's "broad synthetic sample" checks
    (test_normalize_synthetic_sample_no_infinities/_preserves_shapes) used
    ZERO dropout (fully-present sequences) -- only the T=1 hand-built
    cases exercised an absent-face frame at all, never a multi-frame gap
    embedded in a longer real-shaped sequence. Running the real 41
    samples through normalize_landmarks_sample for the first time (this
    file) found zero failures -- the implementation was already correct
    against real gap shapes, this just makes that a permanent, checked
    guarantee instead of an unverified assumption.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from robustness.hold_last_value import (
    compute_fill_validity,
    compute_fill_validity_sample,
    hold_last_value_sample,
)
from robustness.normalisation import compute_head_size, normalize_landmarks_sample
from src.preprocessing.sample_writer import load_sample

_GROUPS = ("pose", "left_hand", "right_hand", "face")
_NPZ_DIRS = ["data/processed/asllrp/train", "data/processed/asllrp/val"]


def _discover_real_samples() -> list[Path]:
    paths = []
    for npz_dir in _NPZ_DIRS:
        d = Path(npz_dir)
        if d.exists():
            paths.extend(sorted(d.glob("*.npz")))
    return paths


_REAL_SAMPLE_PATHS = _discover_real_samples()

pytestmark = pytest.mark.skipif(
    not _REAL_SAMPLE_PATHS,
    reason="real ASLLRP samples not found under data/processed/asllrp/ -- "
    "this test file requires the actual real data, not synthetic fixtures",
)

# All 41 real samples are used, not a subset: each is small (a few hundred
# KB of landmark arrays), the checks below are pure numpy (no model
# loading), and the whole parametrized file runs in a couple of seconds --
# there's no performance reason to sample down, and every one of the 41 is
# a genuinely distinct real occlusion pattern worth checking individually.
_SAMPLE_IDS = [p.stem for p in _REAL_SAMPLE_PATHS]


@pytest.fixture(scope="module")
def _loaded_samples():
    """Loads each real sample once per test session, keyed by path, so the
    41 parametrized tests below don't each re-read the same .npz files
    from disk redundantly across every test function."""
    return {p: load_sample(p) for p in _REAL_SAMPLE_PATHS}


# ---------------------------------------------------------------------
# hold_last_value_sample against real samples
# ---------------------------------------------------------------------


@pytest.mark.parametrize("npz_path", _REAL_SAMPLE_PATHS, ids=_SAMPLE_IDS)
def test_hold_last_value_no_nan_on_real_sample(npz_path, _loaded_samples):
    sample = _loaded_samples[npz_path]
    filled = hold_last_value_sample(sample)

    for group in _GROUPS:
        assert not np.isnan(getattr(filled, group)).any(), (
            f"{npz_path.name}/{group}: hold_last_value_sample must never "
            f"produce NaN -- it only copies real observed values or leaves "
            f"the original zero-fill"
        )


@pytest.mark.parametrize("npz_path", _REAL_SAMPLE_PATHS, ids=_SAMPLE_IDS)
def test_hold_last_value_no_inf_on_real_sample(npz_path, _loaded_samples):
    sample = _loaded_samples[npz_path]
    filled = hold_last_value_sample(sample)

    for group in _GROUPS:
        assert not np.isinf(getattr(filled, group)).any(), (
            f"{npz_path.name}/{group}: hold_last_value_sample must never "
            f"produce +/-inf"
        )


@pytest.mark.parametrize("npz_path", _REAL_SAMPLE_PATHS, ids=_SAMPLE_IDS)
def test_hold_last_value_spot_check_exact_fill_on_real_sample(npz_path, _loaded_samples):
    """
    Not just 'no NaN' -- confirms actual VALUES. For up to 3 missing frames
    per group per sample that had a real prior valid frame to hold from,
    the filled output must equal that prior frame's raw values EXACTLY
    (hold-last-value is a copy, not an approximation).
    """
    sample = _loaded_samples[npz_path]
    filled = hold_last_value_sample(sample)

    checked_any = False
    for group in _GROUPS:
        presence = getattr(sample.presence, group)
        raw = getattr(sample, group)
        out = getattr(filled, group)
        missing_idx = np.where(~presence)[0]

        for t in missing_idx[:3]:
            prior_true = np.where(presence[:t])[0]
            if len(prior_true) == 0:
                continue  # leading gap, covered by a separate test below
            last_valid_idx = prior_true[-1]
            checked_any = True
            np.testing.assert_array_equal(
                out[t], raw[last_valid_idx],
                err_msg=f"{npz_path.name}/{group}: frame {t} should hold "
                        f"frame {last_valid_idx}'s exact values",
            )
    # Not every sample necessarily has a fillable (non-leading) gap in
    # every group, but across 41 real samples at least this one should.
    if npz_path.stem == "asllrp_1169":
        assert checked_any, "expected at least one fillable gap in this sample"


@pytest.mark.parametrize("npz_path", _REAL_SAMPLE_PATHS, ids=_SAMPLE_IDS)
def test_hold_last_value_leading_gap_stays_zero_filled_on_real_sample(npz_path, _loaded_samples):
    """
    A leading gap (no valid observation yet anywhere before it) must stay
    exactly as its original (zero-filled) value -- hold_last_value_sample
    must not fabricate a value there. Across the real dataset, 31 such
    frames exist (30 samples with 1, one sample with 2), all of them
    frames where pose/face/left_hand/right_hand are ALL missing at once
    (the very first tracked frame(s) of the clip, before detection locks
    on) -- confirmed to be the exact same frames scripts/extract_real_dropout_stats.py's
    co-occurrence analysis found under "4 channels missing simultaneously".
    """
    sample = _loaded_samples[npz_path]
    filled = hold_last_value_sample(sample)
    fill_validity = compute_fill_validity_sample(sample)

    for group in _GROUPS:
        validity = getattr(fill_validity, group)
        raw = getattr(sample, group)
        out = getattr(filled, group)
        leading_gap_idx = np.where(~validity)[0]
        for t in leading_gap_idx:
            np.testing.assert_array_equal(
                out[t], raw[t],
                err_msg=f"{npz_path.name}/{group}: leading-gap frame {t} "
                        f"must remain exactly its original zero-fill, not "
                        f"be fabricated or turned into NaN",
            )


def test_real_dataset_leading_gap_total_matches_known_count(_loaded_samples):
    """
    Dataset-level pin: exactly 31 leading-gap frames across all 41 real
    samples (30 samples with 1, one sample -- asllrp_638 -- with 2),
    identical across all four groups (confirming these are synchronized
    whole-frame dropouts, not independent per-channel gaps). A change
    here means either the real data files changed or something about
    fill-validity's leading-gap detection changed -- either is worth a
    fresh look, not a silent update to this number.
    """
    total_leading = {g: 0 for g in _GROUPS}
    for sample in _loaded_samples.values():
        fv = compute_fill_validity_sample(sample)
        for g in _GROUPS:
            total_leading[g] += int((~getattr(fv, g)).sum())

    for g in _GROUPS:
        assert total_leading[g] == 31, (
            f"{g}: expected 31 total leading-gap frames across the real "
            f"dataset, got {total_leading[g]}"
        )


def test_real_left_hand_has_a_gap_longer_than_month1_synthetic_coverage(_loaded_samples):
    """
    Documents-as-code the actual coverage gap found: Month 1's longest
    EXPLICIT contiguous gap in test_hold_last_value.py was 5 frames
    (left_hand_dropout=[1,2,3,4,5] in
    test_hold_last_value_sample_fill_actually_changes_values). Real data
    has a 22-frame left_hand gap in asllrp_1169 -- 4.4x longer. This test
    pins that the real data still actually contains a gap this long (so
    the "Month 1 under-covered this" claim stays checked, not just
    asserted in a docstring) and that hold_last_value_sample fills all 22
    frames correctly from the same prior valid value.
    """
    sample = next(s for s in _loaded_samples.values() if s.metadata.video_id == "asllrp_1169")
    presence = sample.presence.left_hand
    missing = ~presence

    # Find the run of at least 20 consecutive missing frames.
    run_lengths, run_starts = [], []
    start = None
    for t, v in enumerate(missing):
        if v and start is None:
            start = t
        elif not v and start is not None:
            run_lengths.append(t - start)
            run_starts.append(start)
            start = None
    if start is not None:
        run_lengths.append(len(missing) - start)
        run_starts.append(start)

    assert max(run_lengths) >= 20, (
        "expected asllrp_1169's known ~22-frame left_hand gap -- if this "
        "fails, the real data file changed and the Month-1-coverage-gap "
        "claim above needs re-checking against whatever the new longest "
        "gap is"
    )

    long_run_idx = run_lengths.index(max(run_lengths))
    gap_start = run_starts[long_run_idx]
    gap_len = run_lengths[long_run_idx]
    last_valid_idx = gap_start - 1
    assert presence[last_valid_idx], "frame just before the gap should be present"

    filled = hold_last_value_sample(sample)
    for t in range(gap_start, gap_start + gap_len):
        np.testing.assert_array_equal(
            filled.left_hand[t], sample.left_hand[last_valid_idx],
            err_msg=f"22-frame gap: frame {t} should hold frame {last_valid_idx}'s value",
        )


# ---------------------------------------------------------------------
# normalize_landmarks_sample against real (hold-last-value-filled) samples
#
# Uses face_validity=compute_fill_validity(...), matching the actual
# composition src/models/dataset.py's __getitem__ uses in production --
# not the bare presence-only default, since that's what real inference
# actually runs.
# ---------------------------------------------------------------------


@pytest.mark.parametrize("npz_path", _REAL_SAMPLE_PATHS, ids=_SAMPLE_IDS)
def test_normalize_no_unexpected_nan_on_real_sample(npz_path, _loaded_samples):
    """
    NaN must appear EXACTLY where face_validity is False (leading gap) or
    head_size is degenerate for that frame -- checked against the actual
    computed head_size array for THIS real sample, not assumed. Across
    all 41 real samples, zero degenerate-head-size frames exist beyond
    the known leading gaps (confirmed once, empirically, when this test
    was written) -- if that ever changes, this test will catch it.
    """
    sample = _loaded_samples[npz_path]
    filled = hold_last_value_sample(sample)
    face_validity = compute_fill_validity(sample.presence.face)
    out = normalize_landmarks_sample(filled, face_validity=face_validity)

    head_size = compute_head_size(filled.face, face_validity)
    expected_nan = np.isnan(head_size)
    actual_nan = np.isnan(out.pose[:, 0, 0])

    np.testing.assert_array_equal(
        actual_nan, expected_nan,
        err_msg=f"{npz_path.name}: NaN pattern in normalized output doesn't "
                f"match compute_head_size's own NaN pattern -- either an "
                f"unexpected degenerate frame or a masking bug",
    )


@pytest.mark.parametrize("npz_path", _REAL_SAMPLE_PATHS, ids=_SAMPLE_IDS)
def test_normalize_no_inf_on_real_sample(npz_path, _loaded_samples):
    sample = _loaded_samples[npz_path]
    filled = hold_last_value_sample(sample)
    face_validity = compute_fill_validity(sample.presence.face)
    out = normalize_landmarks_sample(filled, face_validity=face_validity)

    for group in _GROUPS:
        arr = getattr(out, group)
        assert np.all(np.isnan(arr) | np.isfinite(arr)), (
            f"{npz_path.name}/{group}: +/-inf found in normalized output"
        )


@pytest.mark.parametrize("npz_path", _REAL_SAMPLE_PATHS, ids=_SAMPLE_IDS)
def test_normalize_pose_visibility_untouched_on_real_sample(npz_path, _loaded_samples):
    """
    pose's visibility channel (index 3) must be identical before and
    after normalize_landmarks_sample. Compared against `filled.pose`
    (hold_last_value_sample's output), which is the actual input
    normalize_landmarks_sample receives here -- NOT the raw pre-fill
    sample, since hold_last_value_sample fills the visibility channel
    too (it treats it like any other channel; only normalisation is
    required to leave it alone). Comparing against the raw sample instead
    would falsely "fail" on every sample with any pose gap -- confirmed
    this exact mistake while writing this test (compared against raw
    first, got a false positive on asllrp_565, traced it to comparing
    against the wrong baseline, not a real bug).
    """
    sample = _loaded_samples[npz_path]
    filled = hold_last_value_sample(sample)
    face_validity = compute_fill_validity(sample.presence.face)
    out = normalize_landmarks_sample(filled, face_validity=face_validity)

    np.testing.assert_array_equal(
        out.pose[:, :, 3], filled.pose[:, :, 3],
        err_msg=f"{npz_path.name}: pose visibility channel changed by normalization",
    )
