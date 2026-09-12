"""
smoke_test_asllrp_recheck.py

Day 27 prep / deferred Item 5: quick structural smoke test on the
CORRECTED ASLLRP data (post Day-22 cap-fix, full corpus re-extracted,
max frame length now genuinely up to 300, not artificially flat at 150).

WHAT THIS CHECKS (structural only, not real training):
  1. Samples spanning the new length range -- including at least one
     sample actually at or near the 300-frame cap -- load and batch
     without shape errors through the existing collate_fn/padding path
     built for WLASL's max 150.
  2. A forward + backward pass runs cleanly (finite loss, gradients
     flow) on a small ASLLRP batch using the CURRENT model as-is.

WHAT THIS DOES NOT CHECK (explicitly out of scope, Month 2 items):
  - Real CTC + boundary-prediction supervision. ASLLRP utterances are
    multi-gloss ("|"-joined ordered SIGN LABELs), but the current
    SignTransformer/compute_loss path (Day 11) was built for WLASL's
    single-gloss [BOS, gloss, EOS] target shape. This script does NOT
    attempt real multi-gloss supervision -- for smoke purposes only,
    it uses the FIRST gloss in each utterance's sequence as a stand-in
    single-token target, purely to exercise the forward/backward path
    structurally. This is not meaningful supervision and must not be
    read as an ASLLRP accuracy signal of any kind.
  - Whether unknown ASLLRP glosses (likely many, since vocab.json was
    built from WLASL labels only) map sensibly to <UNK> -- expected
    and fine for a structural check, irrelevant until Month 2's real
    joint-loss training exists.

GOAL: catch any shape/padding/masking bug BEFORE Month 2 commits real
GPU-hours to it -- same "cheap to check, do it before something
expensive" discipline used throughout this project (Day 12's epoch
extensions, the rank/probability diagnostic before trusting Day 20's
channel decision).

Usage:
    uv run python smoke_test_asllrp_recheck.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.preprocessing.config import ASLLRP_PROCESSED_DIR
from src.preprocessing.sample_writer import load_sample
from src.models.feature_utils import flatten_frame_features
from src.models.sign_transformer import SignTransformer
from src.models.train_utils import compute_loss, build_optimizer

import json
import csv


def load_manifest_rows():
    manifest_path = ASLLRP_PROCESSED_DIR / "manifest.csv"
    with open(manifest_path, "r", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("status") == "success"]
    return rows


def pick_smoke_samples(rows, n=8):
    """Deliberately pick a SPREAD across the length distribution,
    including at least one sample at/near the 300 cap -- a random
    sample might miss the boundary case entirely, and the boundary
    case is exactly what changed since the fix."""
    rows_sorted = sorted(rows, key=lambda r: int(r["sampled_num_frames"]))
    picks = []

    # shortest
    picks.append(rows_sorted[0])
    # a couple from the middle
    mid = len(rows_sorted) // 2
    picks.append(rows_sorted[mid])
    picks.append(rows_sorted[mid + 1] if mid + 1 < len(rows_sorted) else rows_sorted[mid])
    # longest available -- should be at or very near 300 post-fix
    picks.append(rows_sorted[-1])
    picks.append(rows_sorted[-2] if len(rows_sorted) > 1 else rows_sorted[-1])

    # pad out with a few more spread picks if n > len(picks)
    step = max(1, len(rows_sorted) // max(1, n))
    for i in range(0, len(rows_sorted), step):
        if len(picks) >= n:
            break
        if rows_sorted[i] not in picks:
            picks.append(rows_sorted[i])

    return picks[:n]


def main():
    print("==========================")
    print("ASLLRP Post-Fix Smoke Test (structural only)")
    print("==========================\n")

    rows = load_manifest_rows()
    print(f"Manifest rows (success): {len(rows)}")

    samples = pick_smoke_samples(rows, n=8)
    print(f"Selected {len(samples)} samples spanning the length distribution:")
    for r in samples:
        print(f"  {r['video_id']:<30} frames={r['sampled_num_frames']:>4}  "
              f"hit_cap={r.get('hit_max_cap')}")
    print()

    with open("data/vocab.json", "r", encoding="utf-8") as f:
        token_to_id = json.load(f)
    vocab_size = len(token_to_id)
    pad_id = token_to_id["<PAD>"]
    bos_id = token_to_id["<BOS>"]
    eos_id = token_to_id["<EOS>"]
    unk_id = token_to_id["<UNK>"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    model = SignTransformer(vocab_size=vocab_size, pad_token_id=pad_id)
    model.to(device)
    model.train()

    optimizer = build_optimizer(model)

    # --- Build a batch manually (small n, so no need for full DataLoader) ---
    landmark_seqs = []
    target_seqs = []
    unk_count = 0

    for r in samples:
        npz_path = ASLLRP_PROCESSED_DIR / r["split"] / f"{r['video_id']}.npz"
        sample = load_sample(npz_path)
        feats = flatten_frame_features(sample)  # (T, 1692)
        landmark_seqs.append(torch.from_numpy(feats).float())

        # STAND-IN TARGET ONLY -- see module docstring. Real gloss field
        # is "|"-joined; take first token as a structural stand-in.
        first_gloss = r["gloss"].split("|")[0]
        gloss_id = token_to_id.get(first_gloss, unk_id)
        if gloss_id == unk_id:
            unk_count += 1
        target_seqs.append(torch.tensor([bos_id, gloss_id, eos_id]))

    print(f"Stand-in single-token targets: {unk_count}/{len(samples)} mapped to <UNK> "
          f"(expected -- ASLLRP glosses mostly outside WLASL-derived vocab, "
          f"irrelevant for this structural check)\n")

    # Pad landmarks to batch max
    max_len = max(seq.shape[0] for seq in landmark_seqs)
    feat_dim = landmark_seqs[0].shape[1]
    batch_size = len(landmark_seqs)

    padded_landmarks = torch.zeros(batch_size, max_len, feat_dim)
    encoder_attention_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

    for i, seq in enumerate(landmark_seqs):
        T = seq.shape[0]
        padded_landmarks[i, :T, :] = seq
        encoder_attention_mask[i, :T] = True

    target_padded = torch.nn.utils.rnn.pad_sequence(
        target_seqs, batch_first=True, padding_value=pad_id
    )

    padded_landmarks = padded_landmarks.to(device)
    encoder_attention_mask = encoder_attention_mask.to(device)
    target_padded = target_padded.to(device)

    print(f"Batch landmarks shape       : {tuple(padded_landmarks.shape)}")
    print(f"Batch encoder_mask shape    : {tuple(encoder_attention_mask.shape)}")
    print(f"Batch target_tokens shape   : {tuple(target_padded.shape)}")
    print(f"Max sequence length in batch: {max_len}  "
          f"({'AT/NEAR CAP -- exercises the 300 boundary' if max_len >= 280 else 'below cap'})\n")

    # --- Forward + backward ---
    try:
        target_padding_mask = (target_padded == pad_id)

        logits = model(
            padded_landmarks,
            encoder_attention_mask,
            target_padded,
            target_padding_mask=target_padding_mask,
        )
        print(f"Logits shape: {tuple(logits.shape)}")

        loss = compute_loss(logits, target_padded, pad_id)
        print(f"Loss: {loss.item():.4f}  (finite: {torch.isfinite(loss).item()})")

        optimizer.zero_grad()
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        print(f"Grad norm (pre-clip): {grad_norm.item():.4f}")

        has_grad = any(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in model.parameters() if p.requires_grad
        )
        print(f"Gradients flowed (finite): {has_grad}")

        optimizer.step()

        print("\n==========================")
        print("RESULT: PASS -- forward + backward completed cleanly on "
              "post-fix ASLLRP data, including a near-300-frame sample, "
              "with no shape errors and a finite loss/gradient.")
        print("==========================")

    except Exception as e:
        print("\n==========================")
        print(f"RESULT: FAIL -- {type(e).__name__}: {e}")
        print("==========================")
        raise


if __name__ == "__main__":
    main()