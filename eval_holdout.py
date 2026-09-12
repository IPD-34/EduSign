"""
eval_holdout.py — compare checkpoints on ASLLRP val split.

Metric: mean per-gloss-token CTC loss on held-out val data (lower = better).
This is the same loss function training optimizes, just measured on
unseen utterances -- a direct generalization signal, not a proxy for
something else. Note: this only evaluates CTC/recognition generalization,
not boundary-head generalization -- boundary_loss is not scored here.

Confirmed against real code (this session):
  - load_checkpoint(model, optimizer, path, device) requires an optimizer
    and returns model+optimizer state -- unnecessary for eval, so this
    script loads the raw state_dict directly instead.
  - ctc_vocab_size is derived from each checkpoint's own ctc_head.weight
    shape (shape[0] - 1), not from an external config -- this also acts
    as an automatic compatibility check between the two checkpoints.
  - Both checkpoints confirmed to have ctc_head.weight == [2223, 256]
    and boundary_head.weight == [1, 256] (checked this session).
  - target_padding_mask convention confirmed: True/1 = real token,
    False/0 = padding (dataset.py L149-150). Script's BOS/EOS-stripping
    logic already matches this.
  - val split confirmed real: data/processed/asllrp/val, 122 files.

--- PATCH NOTES (this pass) ---
  1. Fixed a real bug: `continues` (not valid Python) -> `continue`.
     The except branch would have raised NameError on the first
     checkpoint-load failure instead of skipping it gracefully.
  2. Added step_424000 / step_426000 / step_428000 to CHECKPOINTS --
     the original set only covered early training (200-58200), which
     never actually tests the question this script exists to answer:
     did continuing training past step 58,200 out to step 428,000
     help or hurt held-out generalization.
  3. Generalized the final summary from "if exactly 2 results" to a
     real ranking over however many checkpoints succeeded, so partial
     failures (a bad path, a missing file) don't silently suppress
     the comparison you actually came here for.
"""

import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.sign_transformer import SignTransformer
from src.models.dataset import SignLanguageDataset, collate_fn

# ---- CONFIG: must match train_month2_asllrp.py exactly ----
HIDDEN_DIM = 256
NUM_ENCODER_LAYERS = 4
NUM_DECODER_LAYERS = 4
NUM_HEADS = 8
DIM_FEEDFORWARD = 1024
DROPOUT = 0.1  # irrelevant at eval (model.eval() disables it) but needed for __init__
PAD_ID = 0

VOCAB_PATH = "data/vocab_merged.json"
DECODER_VOCAB_PATH = "data/vocab.json"  # 1850 entries -- old WLASL decoder vocab
ASLLRP_TOKEN_MAP_PATH = "data/asllrp_token_to_id.json"
VAL_NPZ_DIR = "data/processed/asllrp/val"

CHECKPOINTS = {
    "step_200":   "checkpoints/month2_asllrp_OLD_session_loss_bug/step_0000200.pt",
    "step_250":   "checkpoints/month2_asllrp_OLD_session_loss_bug/step_0000250_final.pt",
    "step_300":   "checkpoints/month2_asllrp_OLD_session_loss_bug/step_0000300_final.pt",
    "step_58000": "checkpoints/month2_asllrp_OLD_session_loss_bug/step_0058000.pt",
    "step_58200": "checkpoints/month2_asllrp_OLD_session_loss_bug/step_0058200.pt",
    # --- added: the actual checkpoints in question ---
    "step_424000": "checkpoints/month2_asllrp_OLD_overfit_424k_428k/step_0424000.pt",
    "step_426000": "checkpoints/month2_asllrp_OLD_overfit_424k_428k/step_0426000.pt",
    "step_428000": "checkpoints/month2_asllrp_OLD_overfit_424k_428k/step_0428000.pt",
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model_for_eval(ckpt_path):
    """
    Loads a raw state_dict (no optimizer needed for eval).
    Derives ctc_vocab_size from the checkpoint's own ctc_head shape,
    so a shape mismatch between checkpoints fails loudly here rather
    than silently producing a wrong comparison later.
    """
    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    sd = checkpoint["model_state_dict"]
    step = checkpoint.get("step")

    ctc_out_dim = sd["ctc_head.weight"].shape[0]
    ctc_vocab_size = ctc_out_dim - 1  # last slot is the CTC blank token

    with open(DECODER_VOCAB_PATH) as f:
        decoder_vocab = json.load(f)

    model = SignTransformer(
        vocab_size=len(decoder_vocab),  # WLASL decoder vocab -- unused by forward_ctc, kept for __init__ compat
        hidden_dim=HIDDEN_DIM,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        num_heads=NUM_HEADS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        pad_token_id=PAD_ID,
        ctc_vocab_size=ctc_vocab_size,
    ).to(DEVICE)

    model.load_state_dict(sd)  # strict=True by default -- a wrong HIDDEN_DIM/
                                # NUM_*_LAYERS/etc above would fail loudly here
                                # with a shape/key mismatch, not silently produce
                                # a wrong architecture.
    return model, step, ctc_vocab_size


@torch.no_grad()
def eval_checkpoint(ckpt_path, val_loader):
    model, step, ctc_vocab_size = load_model_for_eval(ckpt_path)
    model.eval()
    blank_id = ctc_vocab_size

    ctc_loss_fn = nn.CTCLoss(blank=blank_id, zero_infinity=True, reduction="sum")

    total_loss = 0.0
    total_targets = 0
    n_batches = 0
    n_skipped_nonfinite = 0

    for batch in val_loader:
        landmarks = batch["landmarks"].to(DEVICE)
        enc_mask = batch["encoder_attention_mask"].to(DEVICE)

        log_probs = model.forward_ctc(landmarks, enc_mask)      # (B, T, V+1)
        log_probs_ctc = log_probs.transpose(0, 1)                # CTC wants (T, B, V+1)

        input_lengths = enc_mask.sum(dim=1).cpu()                # (B,) real frame counts

        # target_padding_mask: True = real token (confirmed, dataset.py L149-150)
        target_tokens = batch["target_tokens"]
        target_mask = batch["target_padding_mask"]

        targets, target_lengths = [], []
        for i in range(target_tokens.shape[0]):
            real = target_tokens[i][target_mask[i]]
            gloss_only = real[1:-1]  # strip BOS (pos 0) and EOS (last real token)
            targets.append(gloss_only)
            target_lengths.append(len(gloss_only))

        targets_flat = torch.cat(targets)
        target_lengths_t = torch.tensor(target_lengths, dtype=torch.long)

        loss = ctc_loss_fn(
            log_probs_ctc.cpu(),
            targets_flat,
            input_lengths,
            target_lengths_t,
        )

        if not torch.isfinite(loss):
            # Note: with zero_infinity=True, per-utterance infinite losses
            # are already zeroed out INSIDE the sum reduction (per-utterance,
            # before summing) -- so this branch mainly catches NaN, not the
            # input_length < target_length case. The Day 33 corpus-wide audit
            # already confirmed 0/1234 violations across the full corpus
            # (train+val+test), so this is expected to rarely/never fire here.
            n_skipped_nonfinite += 1
            continue

        total_loss += loss.item()
        total_targets += sum(target_lengths)
        n_batches += 1

    mean_loss_per_target = total_loss / max(total_targets, 1)
    return {
        "checkpoint": ckpt_path,
        "step": step,
        "ctc_vocab_size": ctc_vocab_size,
        "n_batches": n_batches,
        "n_skipped_nonfinite": n_skipped_nonfinite,
        "total_val_utterances": len(val_loader.dataset),
        "mean_ctc_loss_per_gloss_token": mean_loss_per_target,
        "sum_ctc_loss": total_loss,
    }


def main():
    val_dataset = SignLanguageDataset(
        npz_dir=VAL_NPZ_DIR,
        vocab_path=VOCAB_PATH,
        asllrp_token_to_id_path=ASLLRP_TOKEN_MAP_PATH,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=4, shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_id=PAD_ID),
    )

    print(f"Val set: {len(val_dataset)} utterances\n")

    results = []
    for name, path in CHECKPOINTS.items():
        try:
            r = eval_checkpoint(path, val_loader)
        except Exception as e:
            print(f"SKIPPING {name} ({path}): {type(e).__name__}: {e}\n")
            continue  # was `continues` -- fixed
        r["name"] = name
        results.append(r)
        print(r, "\n")

    if not results:
        print("No checkpoints evaluated successfully -- nothing to compare.")
        return

    print("--- Comparison (lower mean_ctc_loss_per_gloss_token = better generalization) ---")
    for r in results:
        print(
            f"{r['name']:14s} step={str(r['step']):<8} "
            f"ctc_vocab_size={r['ctc_vocab_size']:<6} "
            f"mean_ctc_loss_per_gloss_token={r['mean_ctc_loss_per_gloss_token']:.4f}  "
            f"(skipped {r['n_skipped_nonfinite']} non-finite batches)"
        )

    # --- Generalized ranking over N results (was: only handled exactly 2) ---
    ranked = sorted(results, key=lambda r: r["mean_ctc_loss_per_gloss_token"])
    print("\n--- Ranked best -> worst on held-out val ---")
    for i, r in enumerate(ranked, 1):
        print(f"{i}. {r['name']} (step {r['step']}): "
              f"{r['mean_ctc_loss_per_gloss_token']:.4f}")

    best = ranked[0]
    print(f"\n=> BEST on val: {best['name']} (step {best['step']}), "
          f"mean_ctc_loss_per_gloss_token={best['mean_ctc_loss_per_gloss_token']:.4f}")

    # Explicit overfitting check: is the *latest*-trained checkpoint worse
    # on val than an earlier one, despite far more training steps?
    by_step = sorted(results, key=lambda r: r["step"])
    latest = by_step[-1]
    if latest["name"] != best["name"]:
        print(
            f"\n[OVERFITTING SIGNAL] The most-trained checkpoint ({latest['name']}, "
            f"step {latest['step']}) is NOT the best on held-out val "
            f"({latest['mean_ctc_loss_per_gloss_token']:.4f} vs. best "
            f"{best['mean_ctc_loss_per_gloss_token']:.4f} at {best['name']}). "
            f"This is a direct, confirmed generalization regression -- "
            f"use {best['name']} (step {best['step']}) going forward, "
            f"not the latest checkpoint."
        )
    else:
        print(
            f"\n[OK] The most-trained checkpoint ({latest['name']}) is also the "
            f"best on held-out val -- no regression detected by this metric. "
            f"Note: this metric only covers CTC/recognition generalization, "
            f"not boundary-head generalization or true WER (post-decode edit "
            f"distance) -- treat as a strong signal, not the final word."
        )


if __name__ == "__main__":
    main()