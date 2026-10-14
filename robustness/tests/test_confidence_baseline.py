"""
test_confidence_baseline.py

Tests for robustness.confidence_baseline: entropy/top-token-probability
math, mask-aware aggregation, and ECE on a synthetic known-calibration
case. Does not load a real checkpoint or dataset -- these are pure
function tests against hand-built tensors, matching how
test_normalisation.py tests normalisation.py's math directly.
"""

from __future__ import annotations

import math

import pytest
import torch

from robustness.confidence_baseline import (
    DEFAULT_ECE_THRESHOLD,
    DEFAULT_MIN_CORRECT_FOR_DECISION,
    DEFAULT_MIN_INCORRECT_FOR_DECISION,
    DEFAULT_MIN_N_FOR_DECISION,
    aggregate_per_utterance,
    compute_ece,
    ctc_greedy_decode,
    entropy_from_log_probs,
    evaluate_calibration_decision,
    top_token_probability,
)


# ---------------------------------------------------------------------
# entropy_from_log_probs / top_token_probability
# ---------------------------------------------------------------------


def test_entropy_known_uniform_distribution():
    # Uniform over 4 classes: H = ln(4).
    probs = torch.full((1, 1, 4), 0.25)
    log_probs = probs.log()

    entropy = entropy_from_log_probs(log_probs)

    assert entropy.shape == (1, 1)
    torch.testing.assert_close(entropy[0, 0], torch.tensor(math.log(4)))


def test_entropy_known_deterministic_distribution():
    # All mass on one class -> entropy 0.
    probs = torch.tensor([[1e-12, 1e-12, 1.0 - 2e-12]])
    log_probs = probs.log().unsqueeze(0)

    entropy = entropy_from_log_probs(log_probs)

    assert entropy.item() < 1e-6


def test_entropy_known_two_class_distribution():
    # p = [0.9, 0.1] -> H = -0.9*ln(0.9) - 0.1*ln(0.1)
    probs = torch.tensor([[0.9, 0.1]])
    log_probs = probs.log().unsqueeze(0)
    expected = -(0.9 * math.log(0.9) + 0.1 * math.log(0.1))

    entropy = entropy_from_log_probs(log_probs)

    torch.testing.assert_close(entropy[0, 0], torch.tensor(expected))


def test_top_token_probability_known_distribution():
    probs = torch.tensor([[0.1, 0.7, 0.2]])
    log_probs = probs.log().unsqueeze(0)

    top_prob = top_token_probability(log_probs)

    torch.testing.assert_close(top_prob[0, 0], torch.tensor(0.7))


# ---------------------------------------------------------------------
# aggregate_per_utterance: padding must be excluded, not zeroed-in
# ---------------------------------------------------------------------


def test_aggregate_per_utterance_masks_out_padding():
    # Utterance 0: real values [1.0, 3.0] over 2 valid timesteps, then
    # padding with a value (100.0) that would badly skew a naive mean.
    per_timestep = torch.tensor([[1.0, 3.0, 100.0]])
    mask = torch.tensor([[True, True, False]])

    result = aggregate_per_utterance(per_timestep, mask)

    torch.testing.assert_close(result, torch.tensor([2.0]))  # mean(1.0, 3.0), not mean(1,3,100)


def test_aggregate_per_utterance_full_length_no_padding():
    per_timestep = torch.tensor([[2.0, 4.0, 6.0]])
    mask = torch.tensor([[True, True, True]])

    result = aggregate_per_utterance(per_timestep, mask)

    torch.testing.assert_close(result, torch.tensor([4.0]))


def test_aggregate_per_utterance_batched_independent_masks():
    per_timestep = torch.tensor([[1.0, 3.0, 100.0], [5.0, 100.0, 100.0]])
    mask = torch.tensor([[True, True, False], [True, False, False]])

    result = aggregate_per_utterance(per_timestep, mask)

    torch.testing.assert_close(result, torch.tensor([2.0, 5.0]))


# ---------------------------------------------------------------------
# ctc_greedy_decode
# ---------------------------------------------------------------------


def _log_probs_for_ids(ids, vocab_size, blank_id):
    """Builds log_probs (T, vocab_size) that argmax to exactly `ids`."""
    T = len(ids)
    logits = torch.full((T, vocab_size), -10.0)
    for t, tok in enumerate(ids):
        logits[t, tok] = 10.0
    return torch.log_softmax(logits, dim=-1)


def test_ctc_greedy_decode_collapses_repeats_and_drops_blank():
    blank_id = 5
    # a a blank b b b a -> collapse repeats -> a blank b a -> drop blank -> [a, b, a]
    ids = [0, 0, blank_id, 1, 1, 1, 0]
    log_probs = _log_probs_for_ids(ids, vocab_size=6, blank_id=blank_id)
    valid_mask = torch.ones(len(ids), dtype=torch.bool)

    decoded = ctc_greedy_decode(log_probs, valid_mask=valid_mask, blank_id=blank_id)

    assert decoded == [0, 1, 0]


def test_ctc_greedy_decode_respects_valid_mask_contiguous_prefix():
    blank_id = 5
    ids = [0, 0, 1, 1]  # only first 2 are "valid"; rest is padding-region noise
    log_probs = _log_probs_for_ids(ids, vocab_size=6, blank_id=blank_id)
    valid_mask = torch.tensor([True, True, False, False])

    decoded = ctc_greedy_decode(log_probs, valid_mask=valid_mask, blank_id=blank_id)

    assert decoded == [0]


def test_ctc_greedy_decode_all_blank_gives_empty():
    blank_id = 3
    ids = [blank_id, blank_id, blank_id]
    log_probs = _log_probs_for_ids(ids, vocab_size=4, blank_id=blank_id)
    valid_mask = torch.ones(len(ids), dtype=torch.bool)

    decoded = ctc_greedy_decode(log_probs, valid_mask=valid_mask, blank_id=blank_id)

    assert decoded == []


def test_ctc_greedy_decode_non_contiguous_mask_selects_correct_frames():
    """
    Regression for the leading-dropout CTC decode bug: a NON-contiguous
    mask (leading invalid frame, matching what encoder_attention_mask can
    now contain per src/models/dataset.py) must select the actual True
    positions, not a mask.sum()-length prefix. Frame 0 is invalid and
    deliberately holds NON-blank, non-matching content (class 4) -- if the
    old `log_probs[:valid_length]` prefix-slice bug were still present,
    it would read this frame as if valid and produce a DIFFERENT (wrong)
    decode than the correct mask-aware selection.
    """
    blank_id = 5
    # Position:      0        1  2  3        4
    # mask:      False(invalid) T  T  False(invalid) T
    # content:      4(garbage) 0  1  4(garbage)      blank
    ids = [4, 0, 1, 4, blank_id]
    log_probs = _log_probs_for_ids(ids, vocab_size=6, blank_id=blank_id)
    valid_mask = torch.tensor([False, True, True, False, True])

    decoded = ctc_greedy_decode(log_probs, valid_mask=valid_mask, blank_id=blank_id)

    # Correct: selects positions 1, 2, 4 (in order) -> ids [0, 1, blank] ->
    # collapse -> [0, 1]. The buggy prefix-slice version (mask.sum()=3,
    # log_probs[:3]) would instead read positions 0,1,2 -> ids [4,0,1] ->
    # decode [4, 0, 1] -- clearly different, proving this case is sensitive
    # to the bug.
    assert decoded == [0, 1]

    # Confirm the buggy prefix-slice reading really would have differed
    # (guards against this test silently becoming non-discriminating if
    # the helper data ever changes).
    buggy_valid_length = int(valid_mask.sum().item())
    buggy_ids = log_probs[:buggy_valid_length].argmax(dim=-1).tolist()
    assert buggy_ids != [ids[i] for i in range(len(ids)) if valid_mask[i]]


# ---------------------------------------------------------------------
# compute_ece: synthetic known-calibration cases
# ---------------------------------------------------------------------


def test_ece_perfectly_calibrated_gives_zero():
    # Each confidence bucket's accuracy matches its confidence exactly.
    # Bucket 0.9: 10 predictions, 9 correct (accuracy 0.9).
    # Bucket 0.5: 10 predictions, 5 correct (accuracy 0.5).
    confidences = [0.9] * 10 + [0.5] * 10
    correct = [1] * 9 + [0] * 1 + [1] * 5 + [0] * 5

    result = compute_ece(confidences, correct, num_bins=10)

    assert result.ece == pytest.approx(0.0, abs=1e-9)


def test_ece_maximally_miscalibrated_confident_but_wrong():
    # Every prediction is maximally confident (1.0) but always wrong
    # (accuracy 0.0) -> ECE should equal the confidence-accuracy gap, 1.0.
    confidences = [0.999] * 5
    correct = [0] * 5

    result = compute_ece(confidences, correct, num_bins=10)

    assert result.ece == pytest.approx(0.999, abs=1e-6)


def test_ece_empty_bins_reported_as_none_not_zero():
    # Only the top bin is populated; all others should show None, not 0.0
    # (0.0 would look like "this bin really did average confidence 0").
    confidences = [0.95, 0.96]
    correct = [1, 1]

    result = compute_ece(confidences, correct, num_bins=10)

    populated = [i for i, c in enumerate(result.bin_confidence) if c is not None]
    assert populated == [9]
    for i in range(9):
        assert result.bin_confidence[i] is None
        assert result.bin_accuracy[i] is None
        assert result.bin_count[i] == 0


def test_ece_bin_weighting_by_count():
    # Two bins: one with many samples slightly miscalibrated, one with a
    # single sample wildly miscalibrated. ECE should weight by count, so
    # the large well-populated bin dominates over the tiny outlier bin.
    confidences = [0.85] * 100 + [0.15]
    correct = [1] * 80 + [0] * 20 + [1]  # bin 0.8: acc=0.8 vs conf=0.85; bin 0.1: acc=1.0 vs conf=0.15

    result = compute_ece(confidences, correct, num_bins=10)

    expected = (100 / 101) * abs(0.8 - 0.85) + (1 / 101) * abs(1.0 - 0.15)
    assert result.ece == pytest.approx(expected, abs=1e-6)


# ---------------------------------------------------------------------
# evaluate_calibration_decision -- see docs/ece_threshold_policy.md for
# the full research-backed justification of the threshold (0.10) and the
# minimum-evidence guard (n>=30, n_correct>=5, n_incorrect>=5).
# ---------------------------------------------------------------------


def test_calibration_decision_needs_calibration_above_threshold_with_sufficient_n():
    result = evaluate_calibration_decision(ece=0.25, n_samples=50, n_correct=25)

    assert result.decision == "NEEDS_CALIBRATION"
    assert "0.25" in result.reason


def test_calibration_decision_acceptable_below_threshold_with_sufficient_n():
    result = evaluate_calibration_decision(ece=0.03, n_samples=50, n_correct=25)

    assert result.decision == "ACCEPTABLE"
    assert "0.03" in result.reason


def test_calibration_decision_at_exact_threshold_is_needs_calibration():
    # >= threshold, not >, per the documented policy ("ECE >= 0.10 -> build").
    result = evaluate_calibration_decision(ece=DEFAULT_ECE_THRESHOLD, n_samples=50, n_correct=25)

    assert result.decision == "NEEDS_CALIBRATION"


def test_calibration_decision_insufficient_n_regardless_of_ece():
    # n_samples below the floor -- even a dramatically miscalibrated ECE
    # with plenty of correct/incorrect variety must not be trusted yet.
    result = evaluate_calibration_decision(
        ece=0.50, n_samples=DEFAULT_MIN_N_FOR_DECISION - 1, n_correct=10
    )

    assert result.decision == "INSUFFICIENT_EVIDENCE"
    assert "n_samples" in result.reason


def test_calibration_decision_insufficient_correct_regardless_of_ece():
    # Plenty of samples, but too few correct predictions -- ECE=0.02
    # (looks great) must still not be trusted as "ACCEPTABLE" evidence.
    result = evaluate_calibration_decision(
        ece=0.02, n_samples=50, n_correct=DEFAULT_MIN_CORRECT_FOR_DECISION - 1
    )

    assert result.decision == "INSUFFICIENT_EVIDENCE"
    assert "n_correct" in result.reason


def test_calibration_decision_insufficient_incorrect_regardless_of_ece():
    # Plenty of samples, but too few INCORRECT predictions (near-perfect
    # accuracy) -- symmetric case to the above, same guard.
    n_samples = 50
    n_correct = n_samples - (DEFAULT_MIN_INCORRECT_FOR_DECISION - 1)
    result = evaluate_calibration_decision(ece=0.02, n_samples=n_samples, n_correct=n_correct)

    assert result.decision == "INSUFFICIENT_EVIDENCE"
    assert "n_incorrect" in result.reason


def test_calibration_decision_current_real_case_is_insufficient_evidence():
    """
    The actual case this whole guard exists for: the step-4000 checkpoint's
    real reported numbers (ECE=0.9814, n_samples=21, n_correct=0). Despite
    ECE being dramatically above the 0.10 threshold, this must resolve to
    INSUFFICIENT_EVIDENCE, not NEEDS_CALIBRATION -- a 0/21 all-wrong result
    is not real calibration evidence, per the Month 2 plan's own intent.
    """
    result = evaluate_calibration_decision(ece=0.9814, n_samples=21, n_correct=0)

    assert result.decision == "INSUFFICIENT_EVIDENCE"
    assert "n_samples=21" in result.reason
    assert "n_correct=0" in result.reason
