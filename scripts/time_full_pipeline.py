"""
time_full_pipeline.py

End-to-end per-stage latency measurement: landmark loading -> robustness
layer (hold_last_value + normalisation) -> flatten-to-model-input ->
transformer forward pass -> MC-dropout's K-pass overhead, run against all
41 real ASLLRP samples (train + val).

Reuses robustness.latency_timer as the instrumentation base (its
Stopwatch context manager, and time_pipeline() for the load/hold/
normalisation stages) rather than reimplementing that timing logic --
latency_timer.py's own docstring deliberately keeps model/confidence
code out of that module ("No model / confidence-estimation / evaluation
code here"), so the transformer + MC-dropout stages are added here
instead, on top of the reused instrumentation.

PURELY a timing measurement -- no decode-correctness, ECE, or
calibration reporting here. best_val.pt's forward pass is only used
because it is already confirmed to run clean (real ctc_log_probs /
boundary_logits tensors, no NaN/Inf); its output QUALITY is irrelevant
to this script and is not inspected.

One real finding surfaced while wiring this: the actual training/data
path (src/models/dataset.py's SignLanguageDataset.__getitem__) does NOT
currently call hold_last_value_sample or normalize_landmarks_sample --
it flattens the RAW loaded sample directly. So the "robustness layer"
timed here is a designed-but-not-yet-wired-in stage for today's live
model path, not a currently-live one. This script measures the
full designed pipeline (load -> hold -> normalize -> flatten -> model)
per the request; it does not change or fix that wiring gap -- flagged
here explicitly, not chased further (out of scope for a timing script).

Usage:
    uv run python -m scripts.time_full_pipeline
"""

from __future__ import annotations

import platform
from pathlib import Path

import numpy as np
import torch

from robustness.confidence_baseline import (
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_VOCAB_PATH,
    _load_model,
)
from robustness.latency_timer import Stopwatch, time_pipeline
from robustness.mc_dropout import DEFAULT_MC_DROPOUT_K, mc_dropout_forward
from src.models.feature_utils import flatten_frame_features

NPZ_DIRS = [
    "data/processed/asllrp/train",
    "data/processed/asllrp/val",
]

STAGES = [
    "load_time",
    "hold_last_value_time",
    "normalisation_time",
    "flatten_time",
    "transformer_forward_time",
    "mc_dropout_k20_time",
]


def _sync(device: torch.device) -> None:
    """CUDA ops are async -- must synchronize before/after a GPU-bound
    stage's Stopwatch boundary, or the recorded time is meaningless
    (it would just measure how fast the CPU can enqueue kernels, not how
    long they actually take to run)."""
    if device.type == "cuda":
        torch.cuda.synchronize()


def time_one_sample(npz_path: Path, model, device: torch.device) -> dict:
    timings, normalized_sample = time_pipeline(npz_path)  # reused: load/hold/normalisation

    with Stopwatch() as sw_flatten:
        features = flatten_frame_features(normalized_sample)  # (T, RAW_FEATURE_DIM)
        landmarks = torch.tensor(features, dtype=torch.float32, device=device).unsqueeze(0)
        encoder_attention_mask = torch.ones(1, features.shape[0], dtype=torch.bool, device=device)
    timings["flatten_time"] = sw_flatten.elapsed

    _sync(device)
    with Stopwatch() as sw_fwd, torch.no_grad():
        model.eval()
        model.forward_ctc_and_boundary(landmarks, encoder_attention_mask)
    _sync(device)
    timings["transformer_forward_time"] = sw_fwd.elapsed

    # K=1 is represented by transformer_forward_time above (a single
    # deterministic eval-mode pass) -- mc_dropout_forward requires K >= 2
    # by design (K=1 can't estimate a variance), so there is no separate
    # "K=1 via MC-dropout" call to make; the base forward pass IS that
    # measurement.
    _sync(device)
    with Stopwatch() as sw_mc:
        mc_dropout_forward(model, landmarks, encoder_attention_mask, K=DEFAULT_MC_DROPOUT_K)
    _sync(device)
    timings["mc_dropout_k20_time"] = sw_mc.elapsed

    return timings


def summarize(values: list[float]) -> dict:
    arr = np.asarray(values)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=== Hardware / environment ===")
    print(f"device: {device} "
          f"({torch.cuda.get_device_name(0) if device.type == 'cuda' else platform.processor()})")
    print(f"platform: {platform.platform()}")
    print(f"torch: {torch.__version__}")
    print(
        "*** This is a dev laptop GPU (RTX 3050 6GB Laptop GPU), NOT the eventual "
        "deployment server Month 3's plan calls for. These numbers characterize "
        "relative per-stage cost and the MC-dropout multiplier, which should carry "
        "over qualitatively, but absolute latency numbers here are NOT representative "
        "of production hardware and should be re-measured on the actual deployment "
        "target before being used for any latency budget/SLA decision. ***\n"
    )

    model = _load_model(DEFAULT_CHECKPOINT_PATH, DEFAULT_VOCAB_PATH, device)

    npz_paths = []
    for npz_dir in NPZ_DIRS:
        npz_paths.extend(sorted(Path(npz_dir).glob("*.npz")))
    print(f"Timing {len(npz_paths)} real samples from {NPZ_DIRS}")

    # Explicit CUDA/cuDNN warm-up: the FIRST GPU call in a process pays a
    # one-time context-init/kernel-selection cost that has nothing to do
    # with steady-state per-sample latency (confirmed directly: an
    # unwarmed first sample measured 0.22s forward-pass / 1.00s
    # mc_dropout_k20 vs. ~0.01s / ~0.15s for every sample after it -- a
    # >10x one-off artifact that would otherwise silently dominate the
    # mean). Run one throwaway pass on the first sample before the timed
    # loop starts, exactly like the real per-sample stages, so the 41
    # measurements below reflect steady-state cost only.
    if npz_paths:
        time_one_sample(npz_paths[0], model, device)

    per_stage_values = {stage: [] for stage in STAGES}
    total_values = []

    for npz_path in npz_paths:
        timings = time_one_sample(npz_path, model, device)
        for stage in STAGES:
            per_stage_values[stage].append(timings[stage])
        total_values.append(sum(timings[s] for s in STAGES))

    print("=== Per-stage latency across all 41 samples (seconds) ===")
    header = f"{'stage':<28} {'mean':>10} {'median':>10} {'p95':>10} {'min':>10} {'max':>10}"
    print(header)
    for stage in STAGES:
        s = summarize(per_stage_values[stage])
        print(f"{stage:<28} {s['mean']:>10.5f} {s['median']:>10.5f} {s['p95']:>10.5f} "
              f"{s['min']:>10.5f} {s['max']:>10.5f}")

    total_summary = summarize(total_values)
    print(f"{'TOTAL end-to-end':<28} {total_summary['mean']:>10.5f} "
          f"{total_summary['median']:>10.5f} {total_summary['p95']:>10.5f} "
          f"{total_summary['min']:>10.5f} {total_summary['max']:>10.5f}")

    fwd_summary = summarize(per_stage_values["transformer_forward_time"])
    mc_summary = summarize(per_stage_values["mc_dropout_k20_time"])
    multiplier_mean = mc_summary["mean"] / fwd_summary["mean"]
    multiplier_median = mc_summary["median"] / fwd_summary["median"]

    print(f"\n=== MC-dropout (K={DEFAULT_MC_DROPOUT_K}) cost vs. single forward pass ===")
    print(f"single forward pass (K=1 equivalent): mean={fwd_summary['mean']:.5f}s, "
          f"median={fwd_summary['median']:.5f}s")
    print(f"mc_dropout_forward(K={DEFAULT_MC_DROPOUT_K}):            mean={mc_summary['mean']:.5f}s, "
          f"median={mc_summary['median']:.5f}s")
    print(f"multiplier: mean={multiplier_mean:.2f}x, median={multiplier_median:.2f}x "
          f"(expected ~{DEFAULT_MC_DROPOUT_K}x if MC-dropout cost is dominated by "
          f"K sequential forward passes)")
    if abs(multiplier_mean - DEFAULT_MC_DROPOUT_K) / DEFAULT_MC_DROPOUT_K > 0.15:
        print(
            f"  -> DEVIATES from the expected ~{DEFAULT_MC_DROPOUT_K}x by more than 15% "
            f"-- worth investigating (e.g. fixed per-call Python/dropout-toggling "
            f"overhead not amortizing away, or GPU underutilization at this batch "
            f"size=1 making per-pass cost non-linear)."
        )
    else:
        print(f"  -> within 15% of the expected ~{DEFAULT_MC_DROPOUT_K}x -- MC-dropout cost "
              f"is essentially K sequential forward passes, no significant extra overhead.")


if __name__ == "__main__":
    main()
