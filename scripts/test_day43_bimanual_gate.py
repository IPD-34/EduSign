"""
scripts/test_day43_bimanual_gate.py -- Day 43 standalone validation of
src/models/bimanual_gate.py against a real ASLLRP batch.

Does NOT touch SignEncoder/SignTransformer/combined_loss -- this only
validates the new module in isolation, per Month 1 Day 9's discipline
(shapes + gradients first, wiring into the pipeline is Day 44).

CPU-scale, no Kaggle quota spent. Checks:
  1. Slice-index math in bimanual_gate.py is actually consistent with
     RAW_FEATURE_DIM (not just internally self-consistent -- checked
     against the real constant encoder.py exports).
  2. Forward pass on a real ASLLRP training batch produces the expected
     shapes.
  3. Backward pass on a dummy loss completes, and EVERY parameter's
     gradient is confirmed present and finite (not just "no crash") --
     same bar Day 9 used for the original encoder validation.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader

from src.models.dataset import SignLanguageDataset, collate_fn
from src.models.encoder import RAW_FEATURE_DIM
from src.models.bimanual_gate import (
    BimanualGateHead,
    LEFT_HAND_SLICE,
    RIGHT_HAND_SLICE,
    POSE_FACE_SLICE,
    HAND_DIM,
)

VOCAB_MERGED_PATH = "data/vocab_merged.json"
ASLLRP_TOKEN_TO_ID_PATH = "data/asllrp_token_to_id.json"
ASLLRP_NPZ_DIR = "data/processed/asllrp/train"


def check_slice_math():
    print("=== Step 1: slice-index sanity check (against real RAW_FEATURE_DIM) ===")
    print(f"  RAW_FEATURE_DIM (from encoder.py) = {RAW_FEATURE_DIM}")
    print(f"  POSE_FACE_SLICE  = {POSE_FACE_SLICE.start}:{POSE_FACE_SLICE.stop}")
    print(f"  LEFT_HAND_SLICE  = {LEFT_HAND_SLICE.start}:{LEFT_HAND_SLICE.stop}  "
          f"(width={LEFT_HAND_SLICE.stop - LEFT_HAND_SLICE.start}, expected HAND_DIM={HAND_DIM})")
    print(f"  RIGHT_HAND_SLICE = {RIGHT_HAND_SLICE.start}:{RIGHT_HAND_SLICE.stop}  "
          f"(width={RIGHT_HAND_SLICE.stop - RIGHT_HAND_SLICE.start}, expected HAND_DIM={HAND_DIM})")
    assert LEFT_HAND_SLICE.stop - LEFT_HAND_SLICE.start == HAND_DIM
    assert RIGHT_HAND_SLICE.stop - RIGHT_HAND_SLICE.start == HAND_DIM
    assert RIGHT_HAND_SLICE.stop == RAW_FEATURE_DIM, (
        f"right_hand slice should end exactly at RAW_FEATURE_DIM "
        f"({RIGHT_HAND_SLICE.stop} != {RAW_FEATURE_DIM})"
    )
    assert RIGHT_HAND_SLICE.start == LEFT_HAND_SLICE.stop, "hands must be contiguous, no gap/overlap"
    print("  PASS: slices are contiguous, non-overlapping, and exactly span the two hands.\n")


def main():
    check_slice_math()

    dataset = SignLanguageDataset(
        npz_dir=ASLLRP_NPZ_DIR,
        vocab_path=VOCAB_MERGED_PATH,
        asllrp_token_to_id_path=ASLLRP_TOKEN_TO_ID_PATH,
    )
    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    batch = next(iter(loader))

    landmarks = batch["landmarks"]  # (B, T, RAW_FEATURE_DIM)
    print("=== Step 2: forward pass on a real ASLLRP batch ===")
    print(f"  input landmarks shape:  {tuple(landmarks.shape)}  dtype={landmarks.dtype}")
    assert landmarks.shape[-1] == RAW_FEATURE_DIM

    model = BimanualGateHead(hidden_dim=256)
    routed_features, gate_logits = model(landmarks)

    B, T, _ = landmarks.shape
    print(f"  routed_features shape:  {tuple(routed_features.shape)}  dtype={routed_features.dtype}  "
          f"(expected ({B}, {T}, 256))")
    print(f"  gate_logits shape:      {tuple(gate_logits.shape)}  dtype={gate_logits.dtype}  "
          f"(expected ({B}, {T}))")
    assert routed_features.shape == (B, T, 256)
    assert gate_logits.shape == (B, T)
    assert torch.isfinite(routed_features).all(), "routed_features contains non-finite values"
    assert torch.isfinite(gate_logits).all(), "gate_logits contains non-finite values"
    print("  PASS: shapes match spec, all outputs finite.\n")

    print("=== Step 3: backward pass + per-parameter gradient check ===")
    dummy_loss = routed_features.sum() + gate_logits.sum()
    dummy_loss.backward()
    print(f"  dummy_loss = {dummy_loss.item():.4f}")

    all_ok = True
    for name, p in model.named_parameters():
        if p.grad is None:
            print(f"  FAIL: {name} has grad=None")
            all_ok = False
            continue
        finite = torch.isfinite(p.grad).all().item()
        grad_norm = p.grad.norm().item()
        status = "ok" if finite else "NON-FINITE"
        print(f"  {name:35s} shape={tuple(p.shape)!s:16s} grad_norm={grad_norm:.6f}  [{status}]")
        if not finite:
            all_ok = False

    print()
    if all_ok:
        print("=== RESULT: PASS -- all parameters have finite, non-None gradients. ===")
    else:
        print("=== RESULT: FAIL -- see flagged parameters above. ===")
        sys.exit(1)


if __name__ == "__main__":
    main()
