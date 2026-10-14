"""
confidence_baseline.py

Month 2, Step 1: the cheap confidence-estimator baseline -- top-token
probability and entropy read directly off the CTC head's log-probs, with
no extra model calls (no MC-dropout, no calibration MLP; those are later,
separate steps in the same Month 2 plan).

Lives under robustness/, not src/models/: everything here consumes a
trained model's outputs to answer "how much should we trust this
prediction," rather than defining or training the model itself. That
matches the existing split -- src/models/ owns the architecture and
training loop (sign_transformer.py, dataset.py, train_utils.py,
combined_loss.py), robustness/ owns analysis run against it
(normalisation.py, latency_timer.py, hold_last_value.py all read model-
adjacent data without altering how the model is built or trained).

Two design decisions were flagged as ambiguous by the Month 2 plan and
confirmed with the project owner before implementation:

1. Per-utterance aggregation (top-token probability, entropy): mean over
   valid timesteps only, per utterance, using encoder_attention_mask to
   exclude padded timesteps. Padding must never dilute the average --
   see aggregate_per_utterance().

2. "Correct" for ECE: CTC has no per-frame ground truth (the target is an
   unaligned gloss sequence, not a per-timestep label), so a framewise
   notion of "correct" would require forced alignment this baseline does
   not implement. Instead, correctness is utterance-level: greedy-decode
   the CTC output (argmax per timestep, then the standard CTC collapse --
   merge consecutive repeats, drop blank), and compare the resulting
   gloss sequence to the target (BOS/EOS stripped) for exact match.
   Confidence paired with each utterance's correctness label is that
   utterance's mean top-token probability from decision 1. This makes ECE
   a standard binary-outcome calibration measure over one point per
   utterance, not per timestep.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models.combined_loss import reorder_valid_frames_first
from src.models.dataset import SignLanguageDataset, collate_fn
from src.models.sign_transformer import SignTransformer

DEFAULT_CHECKPOINT_PATH = "checkpoints/month2_asllrp/best_val.pt"
DEFAULT_VAL_NPZ_DIR = "data/processed/asllrp/val"
DEFAULT_VOCAB_PATH = "data/vocab_merged.json"
DEFAULT_ASLLRP_TOKEN_TO_ID_PATH = "data/asllrp_token_to_id.json"
DEFAULT_NUM_ECE_BINS = 10


# ---------------------------------------------------------------------
# Step 1: per-timestep entropy / top-token probability, per-utterance
# aggregation
# ---------------------------------------------------------------------


def top_token_probability(log_probs: torch.Tensor) -> torch.Tensor:
    """
    log_probs: (..., vocab) log-softmax'd log-probabilities.
    Returns (...,) the probability of the highest-probability token at
    each position: max(exp(log_probs)) == exp(max(log_probs)), computed
    the second way so nothing is exponentiated except the single winning
    value.
    """
    return log_probs.max(dim=-1).values.exp()


def entropy_from_log_probs(log_probs: torch.Tensor) -> torch.Tensor:
    """
    Shannon entropy (in nats) of the distribution at each position,
    computed directly from log-probabilities: H = -sum(p * log p) =
    -sum(exp(log_p) * log_p). This never round-trips through
    log(exp(log_p)) or re-normalizes, so it stays numerically safe even
    for the very small probabilities a large vocab produces -- the only
    exponentiation is exp(log_p) itself, which is always in [0, 1].
    """
    probs = log_probs.exp()
    return -(probs * log_probs).sum(dim=-1)


def aggregate_per_utterance(
    per_timestep: torch.Tensor, encoder_attention_mask: torch.Tensor
) -> torch.Tensor:
    """
    per_timestep: (batch, seq_len) a per-timestep scalar (top-token prob
      or entropy).
    encoder_attention_mask: (batch, seq_len) bool, True = real frame.

    Returns (batch,): mean over valid (unmasked) timesteps per utterance.
    Padded timesteps are excluded from both the sum and the denominator,
    not zeroed-and-averaged-in -- a zero-and-average approach would pull
    every utterance's aggregate toward zero in proportion to how much
    padding its batch happened to contain, which has nothing to do with
    the model's actual confidence.
    """
    mask = encoder_attention_mask.to(per_timestep.dtype)
    valid_counts = mask.sum(dim=1)
    if (valid_counts == 0).any():
        raise ValueError("aggregate_per_utterance: an utterance has zero valid timesteps.")
    return (per_timestep * mask).sum(dim=1) / valid_counts


# ---------------------------------------------------------------------
# Step 2: utterance-level CTC-greedy-decode correctness + ECE
# ---------------------------------------------------------------------


def ctc_greedy_decode(log_probs: torch.Tensor, valid_mask: torch.Tensor, blank_id: int) -> list[int]:
    """
    Standard CTC greedy decode for one utterance: argmax per timestep
    over the valid frames selected by `valid_mask`, then collapse
    consecutive repeats and drop the blank token. Does not use a language
    model or beam search -- this is the cheap baseline's decode, matching
    the "cheap baseline" framing of Step 1.

    valid_mask: (T,) bool, True = real, valid frame (same convention as
    encoder_attention_mask). Takes a MASK, not an integer length: a
    `[:valid_length]` prefix slice silently assumes valid frames are a
    contiguous block starting at index 0, which stopped being true once
    encoder_attention_mask could carry a leading-dropout False (see
    src/models/dataset.py's collate_fn docstring) -- a mask.sum()-length
    prefix slice would then read a shifted window that includes an
    invalid placeholder frame and drops a real one. Reuses
    reorder_valid_frames_first (src/models/combined_loss.py, the same
    fix already applied to the CTC training loss for exactly this
    reason) rather than reimplementing the same selection logic here --
    it's built for a batched (B, T, ...) tensor, so a size-1 batch
    dimension is added/removed around the single-utterance call.
    """
    reordered = reorder_valid_frames_first(
        log_probs.unsqueeze(0), valid_mask.unsqueeze(0)
    )[0]
    valid_length = int(valid_mask.sum().item())
    ids = reordered[:valid_length].argmax(dim=-1).tolist()
    collapsed = []
    prev = None
    for tok in ids:
        if tok != prev:
            if tok != blank_id:
                collapsed.append(tok)
        prev = tok
    return collapsed


@dataclass
class ECEResult:
    ece: float
    bin_edges: list[float]
    bin_confidence: list[float | None]
    bin_accuracy: list[float | None]
    bin_count: list[int]

    def as_dict(self) -> dict:
        return {
            "ece": self.ece,
            "bin_edges": self.bin_edges,
            "bin_confidence": self.bin_confidence,
            "bin_accuracy": self.bin_accuracy,
            "bin_count": self.bin_count,
        }


def compute_ece(
    confidences: list[float], correct: list[int], num_bins: int = DEFAULT_NUM_ECE_BINS
) -> ECEResult:
    """
    Standard equal-width-bin ECE: partition [0, 1] into num_bins bins,
    assign each (confidence, correct) pair to a bin by its confidence,
    and compute sum_b (|bin_b| / N) * |accuracy_b - mean_confidence_b|.

    Empty bins contribute 0 to the ECE sum (nothing to weight) and are
    reported with confidence/accuracy = None, not 0.0 -- 0.0 would be
    indistinguishable from "this bin's average confidence really was
    zero," which is a real, different thing.

    confidence == 1.0 falls into the last bin (right-closed on the final
    edge only), so a perfectly confident prediction isn't silently
    dropped into an out-of-range 11th bin.
    """
    if len(confidences) != len(correct):
        raise ValueError("confidences and correct must be the same length")
    if len(confidences) == 0:
        raise ValueError("compute_ece: no predictions given")

    bin_edges = [i / num_bins for i in range(num_bins + 1)]
    bin_conf_sum = [0.0] * num_bins
    bin_correct_sum = [0.0] * num_bins
    bin_count = [0] * num_bins

    n = len(confidences)
    for conf, is_correct in zip(confidences, correct):
        if not (0.0 <= conf <= 1.0):
            raise ValueError(f"confidence {conf} outside [0, 1]")
        idx = min(int(conf * num_bins), num_bins - 1)
        bin_conf_sum[idx] += conf
        bin_correct_sum[idx] += is_correct
        bin_count[idx] += 1

    ece = 0.0
    bin_confidence: list[float | None] = []
    bin_accuracy: list[float | None] = []
    for b in range(num_bins):
        if bin_count[b] == 0:
            bin_confidence.append(None)
            bin_accuracy.append(None)
            continue
        mean_conf = bin_conf_sum[b] / bin_count[b]
        acc = bin_correct_sum[b] / bin_count[b]
        bin_confidence.append(mean_conf)
        bin_accuracy.append(acc)
        ece += (bin_count[b] / n) * abs(acc - mean_conf)

    return ECEResult(
        ece=ece,
        bin_edges=bin_edges,
        bin_confidence=bin_confidence,
        bin_accuracy=bin_accuracy,
        bin_count=bin_count,
    )


# ---------------------------------------------------------------------
# Step 3: full report over a val dataloader
# ---------------------------------------------------------------------

# --- ECE threshold policy (see docs/ece_threshold_policy.md for the full
# research-backed justification) ---
# ECE >= this triggers building the calibration MLP; below it, raw
# entropy/top-token confidence is used as-is.
DEFAULT_ECE_THRESHOLD = 0.10

# Minimum-evidence guard: below these, the decision is INSUFFICIENT_EVIDENCE
# regardless of the raw ECE value. Directly motivated by the degenerate-ECE
# case already found this session (ECE=0.9814 from 0/21 correct, which is
# mathematically just the mean confidence, not a real calibration signal).
# min_n=30 is the standard introductory-statistics rule-of-thumb minimum for
# a sample proportion/mean to start being reasonably estimable under a
# normal approximation -- not an ECE-specific derivation (none is standard
# in the literature), but a defensible, commonly-cited floor. min_correct/
# min_incorrect=5 each mirrors the "expected cell count >= 5" convention
# from categorical statistics (e.g. chi-square test validity) -- applied
# here to ensure ECE's confidence-vs-accuracy comparison actually has both
# outcomes represented with a minimally stable count, not just one.
DEFAULT_MIN_N_FOR_DECISION = 30
DEFAULT_MIN_CORRECT_FOR_DECISION = 5
DEFAULT_MIN_INCORRECT_FOR_DECISION = 5


@dataclass
class CalibrationDecision:
    decision: str  # "NEEDS_CALIBRATION" | "ACCEPTABLE" | "INSUFFICIENT_EVIDENCE"
    reason: str
    ece: float
    n_samples: int
    n_correct: int
    threshold: float

    def as_dict(self) -> dict:
        return {
            "decision": self.decision,
            "reason": self.reason,
            "ece": self.ece,
            "n_samples": self.n_samples,
            "n_correct": self.n_correct,
            "threshold": self.threshold,
        }


def evaluate_calibration_decision(
    ece: float,
    n_samples: int,
    n_correct: int,
    threshold: float = DEFAULT_ECE_THRESHOLD,
    min_n: int = DEFAULT_MIN_N_FOR_DECISION,
    min_correct: int = DEFAULT_MIN_CORRECT_FOR_DECISION,
    min_incorrect: int = DEFAULT_MIN_INCORRECT_FOR_DECISION,
) -> CalibrationDecision:
    """
    Applies the Month 2 calibration-MLP decision policy (docs/ece_threshold_policy.md)
    to one ECE measurement, returning one of three states rather than
    requiring a human to eyeball the ECE number and remember the policy
    separately each time.

    INSUFFICIENT_EVIDENCE takes priority over the threshold comparison:
    an ECE computed from too few samples, or from an outcome with too
    little variance (e.g. 0 correct, or 0 incorrect), is not treated as
    proof of either "needs calibration" or "acceptable" -- it means the
    eval set can't answer the question yet. This is the guard that
    correctly classifies this project's current real case (ECE=0.9814,
    n_correct=0 out of the checkpoint's total blank-collapse) as
    INSUFFICIENT_EVIDENCE, not as a (misleadingly confident) NEEDS_CALIBRATION
    verdict -- a degenerate all-wrong result is not real calibration
    evidence either way.
    """
    n_incorrect = n_samples - n_correct

    if n_samples < min_n or n_correct < min_correct or n_incorrect < min_incorrect:
        reason = (
            f"insufficient evidence: n_samples={n_samples} (need >= {min_n}), "
            f"n_correct={n_correct} (need >= {min_correct}), "
            f"n_incorrect={n_incorrect} (need >= {min_incorrect})"
        )
        return CalibrationDecision("INSUFFICIENT_EVIDENCE", reason, ece, n_samples, n_correct, threshold)

    if ece >= threshold:
        reason = f"ECE {ece:.4f} >= threshold {threshold:.2f}"
        return CalibrationDecision("NEEDS_CALIBRATION", reason, ece, n_samples, n_correct, threshold)

    reason = f"ECE {ece:.4f} < threshold {threshold:.2f}"
    return CalibrationDecision("ACCEPTABLE", reason, ece, n_samples, n_correct, threshold)


@dataclass
class ConfidenceBaselineReport:
    num_utterances: int
    num_excluded: int
    exclusion_reasons: list[str]
    mean_top_token_probability: float
    mean_entropy: float
    ece_result: ECEResult
    num_correct: int
    calibration_decision: CalibrationDecision
    per_utterance: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "num_utterances": self.num_utterances,
            "num_excluded": self.num_excluded,
            "exclusion_reasons": self.exclusion_reasons,
            "mean_top_token_probability": self.mean_top_token_probability,
            "mean_entropy": self.mean_entropy,
            "ece": self.ece_result.as_dict(),
            "num_correct": self.num_correct,
            "calibration_decision": self.calibration_decision.as_dict(),
            "per_utterance": self.per_utterance,
        }

    def print_report(self) -> None:
        print("=== Confidence baseline report (Month 2, Step 1-3) ===")
        print(
            "CAVEAT: computed on a small preliminary val set "
            f"({self.num_utterances} utterances) -- these are preliminary "
            "numbers, not final Month 2 evidence. The project doc calls for "
            "a larger held-out set later."
        )
        if self.num_excluded:
            print(f"Excluded {self.num_excluded} sample(s): {self.exclusion_reasons}")
        print(f"Mean top-token probability: {self.mean_top_token_probability:.4f}")
        print(f"Mean entropy (nats): {self.mean_entropy:.4f}")
        print(f"ECE ({len(self.ece_result.bin_edges) - 1} bins): {self.ece_result.ece:.4f}")
        print(f"Correct: {self.num_correct}/{self.num_utterances}")
        print(
            f"Calibration decision: {self.calibration_decision.decision} "
            f"({self.calibration_decision.reason})"
        )
        print("Reliability diagram (bin_range: confidence vs accuracy, count):")
        edges = self.ece_result.bin_edges
        for b in range(len(edges) - 1):
            conf = self.ece_result.bin_confidence[b]
            acc = self.ece_result.bin_accuracy[b]
            count = self.ece_result.bin_count[b]
            conf_str = f"{conf:.4f}" if conf is not None else "  n/a "
            acc_str = f"{acc:.4f}" if acc is not None else "  n/a "
            print(
                f"  [{edges[b]:.1f}, {edges[b + 1]:.1f}): "
                f"confidence={conf_str} accuracy={acc_str} n={count}"
            )


@torch.no_grad()
def run_confidence_baseline(
    model: SignTransformer,
    dataloader: DataLoader,
    blank_id: int,
    bos_id: int,
    eos_id: int,
    device: torch.device,
    num_ece_bins: int = DEFAULT_NUM_ECE_BINS,
) -> ConfidenceBaselineReport:
    """
    Runs the cheap confidence baseline over every batch in dataloader:
    per-timestep top-token probability + entropy from the CTC head,
    aggregated per utterance (mean over valid timesteps), plus
    utterance-level CTC-greedy-decode-vs-target exact match for ECE.

    Explicitly restores whatever training mode the model was in
    beforehand, matching the convention already used by
    compute_val_ctc_loss in scripts/train_month2_asllrp.py.
    """
    was_training = model.training
    model.eval()

    all_top_token_prob: list[float] = []
    all_entropy: list[float] = []
    all_confidence: list[float] = []
    all_correct: list[int] = []
    per_utterance: list[dict] = []
    num_excluded = 0
    exclusion_reasons: list[str] = []

    for batch in dataloader:
        landmarks = batch["landmarks"].to(device)
        enc_mask = batch["encoder_attention_mask"].to(device)

        ctc_log_probs, _boundary_logits = model.forward_ctc_and_boundary(landmarks, enc_mask)

        top_prob = top_token_probability(ctc_log_probs)  # (B, T)
        entropy = entropy_from_log_probs(ctc_log_probs)  # (B, T)

        utt_top_prob = aggregate_per_utterance(top_prob, enc_mask)  # (B,)
        utt_entropy = aggregate_per_utterance(entropy, enc_mask)  # (B,)

        target_tokens = batch["target_tokens"]
        target_mask = batch["target_padding_mask"]
        video_ids = batch["video_ids"]

        for i in range(landmarks.shape[0]):
            video_id = video_ids[i]
            real_targets = target_tokens[i][target_mask[i]].tolist()
            if len(real_targets) < 2 or real_targets[0] != bos_id or real_targets[-1] != eos_id:
                num_excluded += 1
                exclusion_reasons.append(f"{video_id}: malformed target sequence (no BOS/EOS)")
                continue
            target_gloss_ids = real_targets[1:-1]

            decoded = ctc_greedy_decode(
                ctc_log_probs[i].cpu(), enc_mask[i].cpu(), blank_id
            )

            is_correct = int(decoded == target_gloss_ids)
            confidence = utt_top_prob[i].item()

            all_top_token_prob.append(utt_top_prob[i].item())
            all_entropy.append(utt_entropy[i].item())
            all_confidence.append(confidence)
            all_correct.append(is_correct)
            per_utterance.append(
                {
                    "video_id": video_id,
                    "top_token_probability": utt_top_prob[i].item(),
                    "entropy": utt_entropy[i].item(),
                    "ctc_greedy_decoded": decoded,
                    "target": target_gloss_ids,
                    "correct": bool(is_correct),
                }
            )

    if was_training:
        model.train()

    if not all_confidence:
        raise RuntimeError(
            "run_confidence_baseline: no utterances survived to compute a report "
            f"(excluded {num_excluded}: {exclusion_reasons})"
        )

    ece_result = compute_ece(all_confidence, all_correct, num_bins=num_ece_bins)
    num_correct = sum(all_correct)
    calibration_decision = evaluate_calibration_decision(
        ece_result.ece, n_samples=len(all_correct), n_correct=num_correct
    )

    return ConfidenceBaselineReport(
        num_utterances=len(all_confidence),
        num_excluded=num_excluded,
        exclusion_reasons=exclusion_reasons,
        mean_top_token_probability=sum(all_top_token_prob) / len(all_top_token_prob),
        mean_entropy=sum(all_entropy) / len(all_entropy),
        ece_result=ece_result,
        num_correct=num_correct,
        calibration_decision=calibration_decision,
        per_utterance=per_utterance,
    )


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def _load_model(checkpoint_path: str, vocab_path: str, device: torch.device) -> SignTransformer:
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_merged = json.load(f)
    ctc_vocab_size = len(vocab_merged)

    model = SignTransformer(vocab_size=1850, ctc_vocab_size=ctc_vocab_size).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(
        f"Loaded checkpoint {checkpoint_path} (step {checkpoint.get('step', '?')}) "
        f"onto {device}"
    )
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--val-npz-dir", default=DEFAULT_VAL_NPZ_DIR)
    parser.add_argument("--vocab-path", default=DEFAULT_VOCAB_PATH)
    parser.add_argument("--asllrp-token-to-id-path", default=DEFAULT_ASLLRP_TOKEN_TO_ID_PATH)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-ece-bins", type=int, default=DEFAULT_NUM_ECE_BINS)
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

    report = run_confidence_baseline(
        model, val_loader, blank_id, bos_id, eos_id, device, num_ece_bins=args.num_ece_bins
    )
    report.print_report()

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(report.as_dict(), f, indent=2)
        print(f"Full report written to {args.output_json}")


if __name__ == "__main__":
    main()
