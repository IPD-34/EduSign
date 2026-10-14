"""
test_mc_dropout.py

Tests for robustness.mc_dropout: verifies the MC-dropout machinery itself
is correct, independent of any real checkpoint's quality --
  - dropout is genuinely stochastic across passes when active (not
    accidentally deterministic due to eval-mode leaking through)
  - variance collapses to ~0 when dropout probability is 0 (sanity check
    that the K-pass variance computation is doing real work, not just
    always reporting nonzero noise)
  - the correlation function is proven correct on synthetic data with a
    KNOWN correlation, independent of the current checkpoint's degenerate
    (all-blank) state
"""

from __future__ import annotations

import torch

import pytest

from robustness.mc_dropout import (
    correlate_variance_with_error,
    enable_dropout,
    mc_dropout_forward,
    run_mc_dropout_baseline,
    spearman_correlation,
)
from src.models.sign_transformer import SignTransformer

# Small model config -- fast to construct/run, no checkpoint needed since
# these tests only care about dropout stochasticity and variance math, not
# trained weights.
_VOCAB_SIZE = 50
_CTC_VOCAB_SIZE = 30
_HIDDEN_DIM = 16
_SEQ_LEN = 5
_BATCH_SIZE = 2


def _make_model(dropout: float) -> SignTransformer:
    torch.manual_seed(0)
    return SignTransformer(
        vocab_size=_VOCAB_SIZE,
        hidden_dim=_HIDDEN_DIM,
        num_encoder_layers=2,
        num_decoder_layers=1,
        num_heads=2,
        dim_feedforward=32,
        dropout=dropout,
        ctc_vocab_size=_CTC_VOCAB_SIZE,
    )


def _make_inputs():
    torch.manual_seed(1)
    from src.models.encoder import RAW_FEATURE_DIM

    landmarks = torch.randn(_BATCH_SIZE, _SEQ_LEN, RAW_FEATURE_DIM)
    encoder_attention_mask = torch.ones(_BATCH_SIZE, _SEQ_LEN, dtype=torch.bool)
    return landmarks, encoder_attention_mask


# ---------------------------------------------------------------------
# enable_dropout / stochasticity
# ---------------------------------------------------------------------


def test_enable_dropout_produces_different_outputs_across_passes():
    model = _make_model(dropout=0.5)
    landmarks, mask = _make_inputs()

    outputs = []
    with enable_dropout(model):
        for _ in range(5):
            log_probs, _ = model.forward_ctc_and_boundary(landmarks, mask)
            outputs.append(log_probs.clone())

    # At least one pair of passes must differ -- if dropout were leaking
    # through as eval-mode (identity), all 5 passes would be bit-identical.
    all_identical = all(torch.equal(outputs[0], o) for o in outputs[1:])
    assert not all_identical


def test_enable_dropout_restores_original_mode():
    model = _make_model(dropout=0.5)
    model.eval()
    with enable_dropout(model):
        pass
    assert not model.training

    model.train()
    with enable_dropout(model):
        pass
    assert model.training


def test_enable_dropout_only_toggles_dropout_modules():
    model = _make_model(dropout=0.5)
    model.eval()
    with enable_dropout(model):
        for m in model.modules():
            if isinstance(m, torch.nn.Dropout):
                assert m.training
            else:
                assert not m.training


# ---------------------------------------------------------------------
# mc_dropout_forward: variance behavior
# ---------------------------------------------------------------------


def test_mc_dropout_forward_variance_near_zero_when_dropout_disabled():
    model = _make_model(dropout=0.0)
    landmarks, mask = _make_inputs()

    result = mc_dropout_forward(model, landmarks, mask, K=10)

    assert result.ctc_variance.abs().max().item() < 1e-10
    assert result.boundary_variance.abs().max().item() < 1e-10


def test_mc_dropout_forward_variance_nonzero_when_dropout_active():
    model = _make_model(dropout=0.5)
    landmarks, mask = _make_inputs()

    result = mc_dropout_forward(model, landmarks, mask, K=20)

    assert result.ctc_variance.mean().item() > 0.0
    assert result.boundary_variance.mean().item() > 0.0


def test_mc_dropout_forward_rejects_k_less_than_2():
    model = _make_model(dropout=0.5)
    landmarks, mask = _make_inputs()

    import pytest

    with pytest.raises(ValueError):
        mc_dropout_forward(model, landmarks, mask, K=1)


def test_mc_dropout_forward_shapes():
    model = _make_model(dropout=0.3)
    landmarks, mask = _make_inputs()

    result = mc_dropout_forward(model, landmarks, mask, K=8)

    assert result.mean_ctc_log_probs.shape == (_BATCH_SIZE, _SEQ_LEN, _CTC_VOCAB_SIZE + 1)
    assert result.ctc_variance.shape == (_BATCH_SIZE, _SEQ_LEN)
    assert result.mean_boundary_prob.shape == (_BATCH_SIZE, _SEQ_LEN)
    assert result.boundary_variance.shape == (_BATCH_SIZE, _SEQ_LEN)
    assert result.K == 8


def test_mc_dropout_forward_restores_model_mode():
    model = _make_model(dropout=0.5)
    model.eval()
    landmarks, mask = _make_inputs()

    mc_dropout_forward(model, landmarks, mask, K=4)

    assert not model.training


# ---------------------------------------------------------------------
# spearman_correlation / correlate_variance_with_error: known synthetic cases
# ---------------------------------------------------------------------


def test_spearman_correlation_perfect_positive():
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [10.0, 20.0, 30.0, 40.0, 50.0]

    corr = spearman_correlation(x, y)

    assert corr == pytest_approx(1.0)


def test_spearman_correlation_perfect_negative():
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [50.0, 40.0, 30.0, 20.0, 10.0]

    corr = spearman_correlation(x, y)

    assert corr == pytest_approx(-1.0)


def test_spearman_correlation_monotonic_not_linear_still_perfect():
    # y = x^3 is monotonic but nonlinear -- Spearman should still read 1.0,
    # unlike Pearson which would be < 1.0 for a nonlinear relationship.
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [xi**3 for xi in x]

    corr = spearman_correlation(x, y)

    assert corr == pytest_approx(1.0)


def test_spearman_correlation_constant_input_is_none():
    x = [5.0, 5.0, 5.0]
    y = [1.0, 2.0, 3.0]

    corr = spearman_correlation(x, y)

    assert corr is None


# ---------------------------------------------------------------------
# Independent hand-verification of spearman_correlation against a
# textbook-computed ground truth (Pearson correlation of the rank
# arrays -- the same method spearman_correlation itself uses, but
# derived here by hand/independently, not by calling the function under
# test or scipy). Expected values are hardcoded literals with the rank
# derivation shown in each comment, so a future reader can check the
# arithmetic without re-running anything.
# ---------------------------------------------------------------------


def test_spearman_correlation_ties_in_both_x_and_y_matches_hand_derivation():
    """
    x = [1, 2, 2, 3, 4, 4, 4, 5], y = [2, 1, 3, 4, 5, 6, 5, 7]
    Two distinct tied groups in x (two 2's, three 4's) AND one tied group
    in y (two 5's) -- ties on both sides simultaneously.

    Ranks (average-rank method, 1-based):
      x sorted: 1(pos1), 2,2(pos2,3 -> avg 2.5), 3(pos4), 4,4,4(pos5,6,7 -> avg 6), 5(pos8)
        rank_x = [1, 2.5, 2.5, 4, 6, 6, 6, 8]
      y sorted: 1(pos1), 2(pos2), 3(pos3), 4(pos4), 5,5(pos5,6 -> avg 5.5), 6(pos7), 7(pos8)
        rank_y (in original x-order [2,1,3,4,5,6,5,7]) = [2, 1, 3, 4, 5.5, 7, 5.5, 8]

    Spearman rho = Pearson(rank_x, rank_y):
      mean_rx = mean_ry = 4.5 (both are permutations summing to n(n+1)/2=36, n=8)
      cov = sum((rx-4.5)(ry-4.5)) = 38.0
      var_x = sum((rx-4.5)^2) = 39.5
      var_y = sum((ry-4.5)^2) = 41.5
      rho = 38.0 / sqrt(39.5 * 41.5) = 38.0 / sqrt(1639.25) = 0.938557751...
    """
    x = [1, 2, 2, 3, 4, 4, 4, 5]
    y = [2, 1, 3, 4, 5, 6, 5, 7]

    corr = spearman_correlation(x, y)

    assert corr == pytest_approx(0.938557751)


def test_spearman_correlation_ties_in_x_only_y_strictly_increasing_matches_hand_derivation():
    """
    Stress case for the tie-handling specifically: ties concentrated in x
    ONLY, y strictly increasing (no ties at all) -- the configuration most
    likely to expose an incomplete/naive tie-correction implementation
    (e.g. a simplified d^2-based Spearman formula with a tie-correction
    term applied to only one variable, or applied incorrectly when only
    one side has ties). spearman_correlation computes Spearman as the
    Pearson correlation of the rank arrays directly (see its docstring),
    which handles this case correctly by construction -- no separate
    correction term exists to get subtly wrong -- but this is verified
    here rather than assumed.

    x = [1, 1, 1, 2, 3, 4, 5, 5], y = [1, 2, 3, 4, 5, 6, 7, 8]

    Ranks:
      x: three 1's tie for ranks 1,2,3 -> avg 2; 2->rank4; 3->rank5; 4->rank6;
         two 5's tie for ranks 7,8 -> avg 7.5
         rank_x = [2, 2, 2, 4, 5, 6, 7.5, 7.5]
      y: strictly increasing, no ties -> rank_y = [1, 2, 3, 4, 5, 6, 7, 8]

    Spearman rho = Pearson(rank_x, rank_y):
      mean_rx = mean_ry = 4.5 (n=8, sum of ranks always n(n+1)/2=36)
      cov = sum((rx-4.5)(ry-4.5)) = 39.5
      var_x = sum((rx-4.5)^2) = 39.5
      var_y = sum((ry-4.5)^2) = 42.0
      rho = 39.5 / sqrt(39.5 * 42.0) = 39.5 / sqrt(1659.0) = 0.969781517...
    """
    x = [1, 1, 1, 2, 3, 4, 5, 5]
    y = [1, 2, 3, 4, 5, 6, 7, 8]

    corr = spearman_correlation(x, y)

    assert corr == pytest_approx(0.969781517)


def test_spearman_correlation_original_variance_error_case_hand_verified():
    """
    Hand-verification of the exact case behind the historical "0.878"
    result reported for MC-dropout's variance-vs-error correlation (see
    test_correlate_variance_with_error_known_perfect_correlation below,
    which asserts a loose >0.85 bound with the same data) -- this test
    pins the PRECISE value, independently derived, confirming 0.878 was
    actually correct rather than merely "looked plausible."

    variances = [0.01, 0.02, 0.03, 0.5, 0.6, 0.7] (all distinct)
    errors    = [0, 0, 0, 1, 1, 1] (three 0's, three 1's -- two tied groups)

    Ranks:
      variances: all distinct, already sorted -> rank_x = [1, 2, 3, 4, 5, 6]
      errors: three 0's tie for ranks 1,2,3 -> avg 2; three 1's tie for
              ranks 4,5,6 -> avg 5
              rank_y = [2, 2, 2, 5, 5, 5]

    Spearman rho = Pearson(rank_x, rank_y):
      mean_rx = mean_ry = 3.5 (n=6, sum of ranks = 21)
      cov = sum((rx-3.5)(ry-3.5)) = 13.5
      var_x = sum((rx-3.5)^2) = 17.5
      var_y = sum((ry-3.5)^2) = 13.5
      rho = 13.5 / sqrt(17.5 * 13.5) = 13.5 / sqrt(236.25) = 0.878310066...
    """
    variances = [0.01, 0.02, 0.03, 0.5, 0.6, 0.7]
    errors = [0, 0, 0, 1, 1, 1]

    corr = spearman_correlation(variances, errors)

    assert corr == pytest_approx(0.878310066)


def test_correlate_variance_with_error_known_perfect_correlation():
    # Constructed so variance strictly increases with whether the
    # utterance was wrong: low-variance utterances are always correct,
    # high-variance utterances are always incorrect -> should read as a
    # strong positive correlation between variance and ERROR. Not exactly
    # 1.0 -- the three "correct" and three "incorrect" utterances tie in
    # the error ranking even though their variances differ, which caps
    # achievable rank correlation below 1.0; this asserts "strong" (>0.85)
    # rather than a specific decimal, since the exact value is a
    # tie-correction artifact, not the property under test.
    variances = [0.01, 0.02, 0.03, 0.5, 0.6, 0.7]
    correct = [1, 1, 1, 0, 0, 0]

    result = correlate_variance_with_error(variances, correct)

    assert not result.degenerate
    assert result.spearman_correlation > 0.85
    assert result.n == 6


def test_correlate_variance_with_error_shuffled_gives_near_zero():
    # Same variances, but correctness shuffled to break any relationship
    # with variance -- interleaved so neither the low half nor the high
    # half of variances is uniformly correct/incorrect.
    variances = [0.01, 0.02, 0.03, 0.5, 0.6, 0.7]
    correct = [1, 0, 1, 0, 1, 0]

    result = correlate_variance_with_error(variances, correct)

    assert not result.degenerate
    assert abs(result.spearman_correlation) < 0.6


def test_correlate_variance_with_error_all_correct_is_degenerate():
    variances = [0.1, 0.2, 0.3]
    correct = [1, 1, 1]

    result = correlate_variance_with_error(variances, correct)

    assert result.degenerate
    assert result.spearman_correlation is None
    assert "no variation in outcome" in result.degenerate_reason


def test_correlate_variance_with_error_all_incorrect_is_degenerate():
    # This is exactly the failure mode found against the real, non-converged
    # checkpoint (0/21 correct) -- confirming the function reports it as
    # explicitly degenerate rather than a misleading 0.0.
    variances = [0.1, 0.2, 0.3]
    correct = [0, 0, 0]

    result = correlate_variance_with_error(variances, correct)

    assert result.degenerate
    assert result.spearman_correlation is None


def test_correlate_variance_with_error_constant_variance_is_degenerate():
    variances = [0.5, 0.5, 0.5]
    correct = [1, 0, 1]

    result = correlate_variance_with_error(variances, correct)

    assert result.degenerate
    assert "identical variance" in result.degenerate_reason


def pytest_approx(expected, abs=1e-9):
    import pytest

    return pytest.approx(expected, abs=abs)


# ---------------------------------------------------------------------
# run_mc_dropout_baseline: NaN input must raise loudly, not silently
# produce a "nan" summary statistic.
#
# Regression for a real gap: confidence_baseline.py's compute_ece already
# rejects a NaN confidence value (every comparison against NaN is False,
# so `0.0 <= conf <= 1.0` fails and it raises). mc_dropout.py had no
# equivalent -- a NaN per-utterance variance used to average silently into
# "Mean per-utterance CTC predictive variance: nan" with no error raised
# at all, discovered when re-running against a pipeline that (at the time)
# still had an unmasked NaN input timestep.
# ---------------------------------------------------------------------


def _make_poisoned_batch():
    """One batch, one utterance, with a NaN landmark timestep that is NOT
    excluded from encoder_attention_mask -- reproducing the exact real
    condition (an invalid timestep reaching self-attention unmasked) that
    used to silently poison mc_dropout's per-utterance variance to NaN."""
    from src.models.encoder import RAW_FEATURE_DIM

    torch.manual_seed(2)
    landmarks = torch.randn(1, _SEQ_LEN, RAW_FEATURE_DIM)
    landmarks[0, 0, :] = float("nan")  # poisoned timestep, left unmasked below
    encoder_attention_mask = torch.ones(1, _SEQ_LEN, dtype=torch.bool)  # NOT masking it out

    bos_id, eos_id = 1, 2
    target_tokens = torch.tensor([[bos_id, 5, eos_id]])
    target_padding_mask = torch.ones(1, 3, dtype=torch.bool)

    batch = {
        "landmarks": landmarks,
        "encoder_attention_mask": encoder_attention_mask,
        "target_tokens": target_tokens,
        "target_padding_mask": target_padding_mask,
        "video_ids": ["poisoned_sample"],
    }
    return [batch], bos_id, eos_id


def test_run_mc_dropout_baseline_raises_on_nan_variance_instead_of_silent_nan():
    model = _make_model(dropout=0.3)
    dataloader, bos_id, eos_id = _make_poisoned_batch()
    device = torch.device("cpu")

    with pytest.raises(ValueError, match="not finite"):
        run_mc_dropout_baseline(
            model, dataloader, blank_id=_CTC_VOCAB_SIZE, bos_id=bos_id, eos_id=eos_id,
            device=device, K=4,
        )


def test_run_mc_dropout_baseline_error_names_the_offending_utterance():
    model = _make_model(dropout=0.3)
    dataloader, bos_id, eos_id = _make_poisoned_batch()
    device = torch.device("cpu")

    with pytest.raises(ValueError, match="poisoned_sample"):
        run_mc_dropout_baseline(
            model, dataloader, blank_id=_CTC_VOCAB_SIZE, bos_id=bos_id, eos_id=eos_id,
            device=device, K=4,
        )
