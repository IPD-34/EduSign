"""
mc_dropout.py

Month 2 Step 2 (per the plan): MC-dropout uncertainty estimation, composed
on top of robustness/confidence_baseline.py rather than duplicating it.
This module is independent of checkpoint quality -- it exercises the
model's dropout layers K times and measures how much the predictive
distribution moves around, which is a property of the model's
architecture and weights, not of whether its predictions happen to be
correct yet.

Regularization-layer audit (done before choosing an activation strategy,
per the investigation's ask): SignEncoder (src/models/encoder.py) uses
nn.TransformerEncoderLayer + a PositionalEncoding module, both of which
place their stochasticity entirely in nn.Dropout sublayers (attention
dropout, feedforward dropout, positional-encoding dropout). SignDecoder
is not touched by forward_ctc_and_boundary() at all. The CTC head and
boundary head (sign_transformer.py) are plain nn.Linear, no dropout.
There is NO BatchNorm (or any other layer with train/eval-dependent
running statistics) anywhere in this path -- only nn.Dropout and
nn.LayerNorm, and LayerNorm's behavior does not depend on train/eval
mode (it always normalizes per-sample, no running stats). So a blind
model.train() would, today, be harmless for this specific architecture.
It is not used anyway: enable_dropout() below targets nn.Dropout modules
specifically and leaves every other module's mode untouched, so this
stays correct even if a future architecture change adds a running-stats
layer.

Averaging space, stated explicitly (per the investigation's ask):
MC-dropout's K stochastic forward passes are Monte Carlo samples from an
approximate posterior over the network's weights; the quantity being
estimated is the predictive distribution p(y|x) = E_weights[p(y|x,
weights)] -- an average of *probabilities*, not of log-probabilities.
mean(log p_1, ..., log p_K) is the log of a *geometric* mean of
probabilities, which is not what MC-dropout approximates and is
additionally dominated by whichever pass assigns near-zero probability
to a class (a single confident-wrong pass can crater the geometric
mean). The correct, standard MC-dropout estimator is: convert each
pass's log-probs to probs, arithmetic-mean across the K passes, then
(if log-probs are needed downstream) take the log of that mean. The
same reasoning applies to the boundary head: sigmoid(logit) is a
probability, so it's averaged and its variance measured in
sigmoid-probability space, not raw-logit space -- logit averaging
would suffer the same nonlinearity problem softmax averaging would.
(This is unrelated to, and does not change, the separate documented
convention that the *calibration MLP*, not yet built, should consume
raw logits as its input feature -- that's about what feeds a not-yet-
built classifier, not about how MC-dropout should average stochastic
samples.)
"""

from __future__ import annotations

import argparse
import json
import math
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from robustness.confidence_baseline import (
    DEFAULT_ASLLRP_TOKEN_TO_ID_PATH,
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_VAL_NPZ_DIR,
    DEFAULT_VOCAB_PATH,
    aggregate_per_utterance,
    ctc_greedy_decode,
    _load_model,
)
from src.models.dataset import SignLanguageDataset, collate_fn
from src.models.sign_transformer import SignTransformer

DEFAULT_MC_DROPOUT_K = 20


# ---------------------------------------------------------------------
# Targeted "dropout only" activation
# ---------------------------------------------------------------------


@contextmanager
def enable_dropout(model: nn.Module):
    """
    Puts the model in eval() (so anything with train/eval-dependent
    behavior other than dropout -- e.g. a future BatchNorm -- stays in
    inference mode, using its learned/running statistics rather than
    per-batch ones), then flips ONLY nn.Dropout submodules back to
    train() so they sample instead of acting as identity. Restores the
    model's original training/eval mode on exit, whatever it was.

    See this module's docstring for the audit confirming that, today,
    nn.Dropout is the only train/eval-sensitive layer in the encoder ->
    CTC-head / boundary-head path -- this context manager doesn't rely
    on that staying true, since it targets nn.Dropout explicitly rather
    than assuming model.train() is safe.
    """
    was_training = model.training
    model.eval()
    dropout_modules = [m for m in model.modules() if isinstance(m, nn.Dropout)]
    for m in dropout_modules:
        m.train()
    try:
        yield
    finally:
        model.train(was_training)


# ---------------------------------------------------------------------
# K stochastic forward passes
# ---------------------------------------------------------------------


@dataclass
class MCDropoutResult:
    mean_ctc_log_probs: torch.Tensor  # (B, T, V) -- log(mean_k probs), see module docstring
    ctc_variance: torch.Tensor  # (B, T) -- per-timestep scalar: mean over vocab of per-class variance across K
    mean_boundary_prob: torch.Tensor  # (B, T) -- mean_k sigmoid(boundary_logits)
    boundary_variance: torch.Tensor  # (B, T) -- variance across K of sigmoid(boundary_logits)
    K: int


def mc_dropout_forward(
    model: SignTransformer,
    landmarks: torch.Tensor,
    encoder_attention_mask: torch.Tensor,
    K: int = DEFAULT_MC_DROPOUT_K,
) -> MCDropoutResult:
    """
    Runs K stochastic forward passes of model.forward_ctc_and_boundary
    with dropout active (via enable_dropout), and returns the predictive
    mean and variance for both the CTC output and the boundary head.

    K is configurable (default DEFAULT_MC_DROPOUT_K = 20, a standard
    MC-dropout sample count) and must be >= 2 for the variance to be a
    real estimate rather than trivially 0 -- K=1 is rejected explicitly
    rather than silently returning a meaningless all-zero variance.
    """
    if K < 2:
        raise ValueError(f"mc_dropout_forward: K must be >= 2 to estimate variance, got K={K}")

    ctc_prob_samples = []
    boundary_prob_samples = []

    with enable_dropout(model):
        for _ in range(K):
            ctc_log_probs, boundary_logits = model.forward_ctc_and_boundary(
                landmarks, encoder_attention_mask
            )
            ctc_prob_samples.append(ctc_log_probs.exp())
            boundary_prob_samples.append(torch.sigmoid(boundary_logits))

    ctc_probs_stack = torch.stack(ctc_prob_samples, dim=0)  # (K, B, T, V)
    boundary_probs_stack = torch.stack(boundary_prob_samples, dim=0)  # (K, B, T)

    mean_ctc_probs = ctc_probs_stack.mean(dim=0)  # (B, T, V)
    # clamp before log: mean of K softmax outputs is provably in (0, 1] for
    # any class with nonzero probability in at least one pass, but a class
    # that is exactly 0 in every one of K passes (possible in float32) would
    # otherwise produce -inf here.
    mean_ctc_log_probs = torch.log(mean_ctc_probs.clamp_min(1e-12))

    # unbiased=False (population variance, i.e. divide by K not K-1): K here
    # is the full set of MC samples actually drawn, not a sample from a
    # larger population we're trying to generalize from, so there is no
    # small-sample correction to make -- and unbiased=True would divide by
    # zero and return NaN if K ever were 1.
    ctc_class_variance = ctc_probs_stack.var(dim=0, unbiased=False)  # (B, T, V)
    ctc_variance = ctc_class_variance.mean(dim=-1)  # (B, T)

    mean_boundary_prob = boundary_probs_stack.mean(dim=0)  # (B, T)
    boundary_variance = boundary_probs_stack.var(dim=0, unbiased=False)  # (B, T)

    return MCDropoutResult(
        mean_ctc_log_probs=mean_ctc_log_probs,
        ctc_variance=ctc_variance,
        mean_boundary_prob=mean_boundary_prob,
        boundary_variance=boundary_variance,
        K=K,
    )


def aggregate_variance_per_utterance(
    ctc_variance: torch.Tensor, encoder_attention_mask: torch.Tensor
) -> torch.Tensor:
    """
    Thin, intentional pass-through to confidence_baseline.aggregate_per_utterance
    -- same masked-mean-over-valid-timesteps aggregation used for entropy and
    top-token probability, so a per-utterance variance number is directly
    comparable to those (same aggregation policy, not a second one invented
    here).
    """
    return aggregate_per_utterance(ctc_variance, encoder_attention_mask)


# ---------------------------------------------------------------------
# Correlation: does MC-dropout variance track actual error rate?
# ---------------------------------------------------------------------


def _rankdata(values: list[float]) -> list[float]:
    """Average ranks (1-based), ties get the mean of the ranks they'd span.
    No scipy dependency: this project's pyproject.toml doesn't declare scipy
    as a direct dependency (only picked up transitively via fiftyone), so a
    hand-rolled implementation keeps this module's own dependency footprint
    honest."""
    n = len(values)
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2 + 1  # 1-based
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def _pearson_correlation(x: list[float], y: list[float]) -> float | None:
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    var_x = sum((xi - mean_x) ** 2 for xi in x)
    var_y = sum((yi - mean_y) ** 2 for yi in y)
    if var_x == 0.0 or var_y == 0.0:
        return None  # degenerate: at least one variable is constant
    return cov / (var_x**0.5 * var_y**0.5)


def spearman_correlation(x: list[float], y: list[float]) -> float | None:
    """
    Spearman rank correlation, computed as the Pearson correlation of each
    variable's ranks. Returns None if either variable is constant (rank
    correlation is undefined, not 0, when there's no variation to rank).

    Spearman rather than Pearson, because: (1) MC-dropout variance is a
    nonnegative, typically right-skewed quantity -- the "done right if" bar
    ("variance correlates with error rate") is a monotonic claim (more
    uncertain -> more likely wrong), not a claim that the relationship is
    linear in raw variance units, and Spearman measures exactly the
    monotonic relationship without assuming linearity; (2) the other side
    of the pairing is a binary correctness/error label, and Pearson's
    linear-correlation assumption is a poor fit against a variable with
    only two possible values combined with a skewed continuous one; (3)
    Spearman is the standard choice in the uncertainty-calibration
    literature for exactly this "does my uncertainty signal track errors"
    check.
    """
    if len(x) != len(y):
        raise ValueError("spearman_correlation: x and y must be the same length")
    if len(x) < 2:
        raise ValueError("spearman_correlation: need at least 2 points")
    return _pearson_correlation(_rankdata(x), _rankdata(y))


@dataclass
class VarianceErrorCorrelation:
    spearman_correlation: float | None
    n: int
    degenerate: bool
    degenerate_reason: str | None = None

    def as_dict(self) -> dict:
        return {
            "spearman_correlation": self.spearman_correlation,
            "n": self.n,
            "degenerate": self.degenerate,
            "degenerate_reason": self.degenerate_reason,
        }


def correlate_variance_with_error(
    variances: list[float], correct: list[int]
) -> VarianceErrorCorrelation:
    """
    Correlates per-utterance MC-dropout variance against per-utterance
    ERROR (1 - correct, using the same CTC-greedy-decode exact-match
    correctness definition as confidence_baseline.compute_ece), so a
    positive correlation has the intuitive reading "higher variance,
    more likely wrong" -- the "done right if" bar from the Month 2 plan.

    Degenerate inputs (variance has zero spread, or every utterance has
    the same correctness outcome -- e.g. all-correct or, as with the
    current non-converged checkpoint, all-incorrect) are reported
    explicitly via `degenerate`/`degenerate_reason` with correlation=None,
    rather than silently returning 0.0, which would misleadingly read as
    "measured zero correlation" instead of "correlation is undefined
    here."
    """
    if len(variances) != len(correct):
        raise ValueError("correlate_variance_with_error: variances and correct must be same length")
    n = len(variances)
    if n < 2:
        raise ValueError("correlate_variance_with_error: need at least 2 utterances")

    errors = [1 - c for c in correct]

    if len(set(variances)) == 1:
        return VarianceErrorCorrelation(
            spearman_correlation=None, n=n, degenerate=True,
            degenerate_reason="all utterances have identical variance -- no spread to correlate",
        )
    if len(set(errors)) == 1:
        outcome = "correct" if errors[0] == 0 else "incorrect"
        return VarianceErrorCorrelation(
            spearman_correlation=None, n=n, degenerate=True,
            degenerate_reason=f"every utterance is {outcome} -- no variation in outcome to correlate against",
        )

    corr = spearman_correlation(variances, errors)
    return VarianceErrorCorrelation(spearman_correlation=corr, n=n, degenerate=False)


# ---------------------------------------------------------------------
# Full report over a val dataloader
# ---------------------------------------------------------------------


@dataclass
class MCDropoutReport:
    num_utterances: int
    K: int
    mean_ctc_variance: float
    mean_boundary_variance: float
    correlation: VarianceErrorCorrelation
    per_utterance: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "num_utterances": self.num_utterances,
            "K": self.K,
            "mean_ctc_variance": self.mean_ctc_variance,
            "mean_boundary_variance": self.mean_boundary_variance,
            "correlation": self.correlation.as_dict(),
            "per_utterance": self.per_utterance,
        }

    def print_report(self) -> None:
        print("=== MC-dropout report (Month 2, Step 2) ===")
        print(
            "CAVEAT: computed on a small preliminary val set "
            f"({self.num_utterances} utterances) -- these are preliminary "
            "numbers, not final Month 2 evidence."
        )
        print(f"K (stochastic forward passes): {self.K}")
        print(f"Mean per-utterance CTC predictive variance: {self.mean_ctc_variance:.6f}")
        print(f"Mean per-utterance boundary predictive variance: {self.mean_boundary_variance:.6f}")
        if self.correlation.degenerate:
            print(
                f"Variance-vs-error correlation: DEGENERATE "
                f"({self.correlation.degenerate_reason}), n={self.correlation.n}"
            )
        else:
            print(
                f"Variance-vs-error Spearman correlation: "
                f"{self.correlation.spearman_correlation:.4f} (n={self.correlation.n})"
            )


def _validate_finite(value: float, name: str, video_id: str) -> None:
    """
    Raises a clear error if `value` is NaN or +/-inf, instead of letting it
    silently propagate into a meaningless "nan" summary statistic.

    Mirrors confidence_baseline.compute_ece's validation of each confidence
    value (`if not (0.0 <= conf <= 1.0): raise ValueError(...)`, which
    itself rejects NaN since every comparison against NaN is False) -- that
    check happened to be what caught NaN-poisoned utterances in
    confidence_baseline.py's report. run_mc_dropout_baseline had no
    equivalent: a NaN per-utterance variance value used to average silently
    into "Mean per-utterance CTC predictive variance: nan" with no error at
    all, which is worse than a crash -- it looks like a valid (if oddly
    small-looking) number if nobody reads closely enough to notice "nan".
    """
    if not math.isfinite(value):
        raise ValueError(
            f"{name} for utterance {video_id!r} is {value} (not finite) -- "
            f"refusing to silently average this into the report's summary "
            f"statistics. This usually means the model produced NaN/Inf "
            f"output for this utterance (e.g. an unmasked invalid input "
            f"timestep) -- investigate that utterance directly rather than "
            f"suppressing this check."
        )


@torch.no_grad()
def run_mc_dropout_baseline(
    model: SignTransformer,
    dataloader: DataLoader,
    blank_id: int,
    bos_id: int,
    eos_id: int,
    device: torch.device,
    K: int = DEFAULT_MC_DROPOUT_K,
) -> MCDropoutReport:
    """
    Runs MC-dropout over every batch in dataloader, aggregates predictive
    variance per utterance, decodes the MC-dropout mean CTC distribution
    with the same ctc_greedy_decode used by the cheap baseline to get an
    exact-match correctness label per utterance, and correlates variance
    against error rate.

    Explicitly restores whatever training mode the model was in
    beforehand, same convention as run_confidence_baseline.
    """
    was_training = model.training
    model.eval()

    all_ctc_variance: list[float] = []
    all_boundary_variance: list[float] = []
    all_correct: list[int] = []
    per_utterance: list[dict] = []

    for batch in dataloader:
        landmarks = batch["landmarks"].to(device)
        enc_mask = batch["encoder_attention_mask"].to(device)

        result = mc_dropout_forward(model, landmarks, enc_mask, K=K)

        utt_ctc_variance = aggregate_variance_per_utterance(result.ctc_variance, enc_mask)
        utt_boundary_variance = aggregate_variance_per_utterance(result.boundary_variance, enc_mask)

        target_tokens = batch["target_tokens"]
        target_mask = batch["target_padding_mask"]
        video_ids = batch["video_ids"]

        for i in range(landmarks.shape[0]):
            video_id = video_ids[i]
            real_targets = target_tokens[i][target_mask[i]].tolist()
            if len(real_targets) < 2 or real_targets[0] != bos_id or real_targets[-1] != eos_id:
                continue
            target_gloss_ids = real_targets[1:-1]

            decoded = ctc_greedy_decode(
                result.mean_ctc_log_probs[i].cpu(), enc_mask[i].cpu(), blank_id
            )
            is_correct = int(decoded == target_gloss_ids)

            ctc_variance_value = utt_ctc_variance[i].item()
            boundary_variance_value = utt_boundary_variance[i].item()
            _validate_finite(ctc_variance_value, "ctc_variance", video_id)
            _validate_finite(boundary_variance_value, "boundary_variance", video_id)

            all_ctc_variance.append(ctc_variance_value)
            all_boundary_variance.append(boundary_variance_value)
            all_correct.append(is_correct)
            per_utterance.append(
                {
                    "video_id": video_id,
                    "ctc_variance": utt_ctc_variance[i].item(),
                    "boundary_variance": utt_boundary_variance[i].item(),
                    "ctc_greedy_decoded": decoded,
                    "target": target_gloss_ids,
                    "correct": bool(is_correct),
                }
            )

    if was_training:
        model.train()

    if not all_ctc_variance:
        raise RuntimeError("run_mc_dropout_baseline: no utterances produced a usable result")

    correlation = correlate_variance_with_error(all_ctc_variance, all_correct)

    return MCDropoutReport(
        num_utterances=len(all_ctc_variance),
        K=K,
        mean_ctc_variance=sum(all_ctc_variance) / len(all_ctc_variance),
        mean_boundary_variance=sum(all_boundary_variance) / len(all_boundary_variance),
        correlation=correlation,
        per_utterance=per_utterance,
    )


# ---------------------------------------------------------------------
# CLI (separate from confidence_baseline.py's CLI -- see module docstring)
# ---------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--val-npz-dir", default=DEFAULT_VAL_NPZ_DIR)
    parser.add_argument("--vocab-path", default=DEFAULT_VOCAB_PATH)
    parser.add_argument("--asllrp-token-to-id-path", default=DEFAULT_ASLLRP_TOKEN_TO_ID_PATH)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--K", type=int, default=DEFAULT_MC_DROPOUT_K)
    parser.add_argument(
        "--output-json", default=None, help="Optional path to also write the full report as JSON."
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.vocab_path, "r", encoding="utf-8") as f:
        vocab_merged = json.load(f)
    ctc_vocab_size = len(vocab_merged)
    blank_id = ctc_vocab_size
    bos_id = vocab_merged["<BOS>"]
    eos_id = vocab_merged["<EOS>"]

    model = _load_model(args.checkpoint, args.vocab_path, device)

    val_dataset = SignLanguageDataset(
        npz_dir=args.val_npz_dir,
        vocab_path=args.vocab_path,
        asllrp_token_to_id_path=args.asllrp_token_to_id_path,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False
    )
    print(f"Val set: {len(val_dataset)} utterances from {args.val_npz_dir}")

    report = run_mc_dropout_baseline(
        model, val_loader, blank_id, bos_id, eos_id, device, K=args.K
    )
    report.print_report()

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(report.as_dict(), f, indent=2)
        print(f"Full report written to {args.output_json}")


if __name__ == "__main__":
    main()
