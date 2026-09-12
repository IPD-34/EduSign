"""
test_boundary_weight_2x_smoke.py

Day 39 follow-up: validate boundary_loss_weight=2.0 locally, cheaply,
BEFORE committing real Kaggle hours to a re-run with the new weight.

Same discipline as Day 33's extended smoke test ("two checks before
spending real Kaggle quota") -- this is that same check, applied to a
hyperparameter change instead of a fresh training kickoff.

Why 2.0: Day 39's fine-grained log review showed boundary_loss declining
steadily and continuously from step 300 all the way to step 14000 --
predating and continuing through the post-step-4000 overfitting/val-
divergence point, not just an artifact of it. ctc_loss dominates the
combined loss the entire run (ratio never drops below ~1.8x even at the
end), so boundary_loss's gradient contribution to total_loss has been
proportionally small throughout. The decision: it's declining despite
being under-weighted, so doubling its weight is a reasonable first step
to test whether that accelerates real learning -- not a blind guess.

What this script checks, resuming from best_val.pt (step 4000, the
checkpoint of record) rather than day12_latest.pt:
  1. Does ctc_loss stay stable (not destabilized) when boundary_loss's
     gradient contribution doubles?
  2. Does grad_norm stay in the same healthy band Day 39 characterized
     (no new sustained spike introduced by the reweighting)?
  3. Does boundary_loss's rate of decline visibly change (faster / same /
     slower) compared to the boundary_loss_weight=1.0 baseline already
     documented in Day 33 (0.7733->0.6856 over 150 steps) and Day 39
     (0.5895->0.5329 over the final 4000 real-training steps)?

IMPORTANT -- checkpoint loading assumption, not yet verified:
This uses load_checkpoint() (full-model load), not load_checkpoint_partial(),
because best_val.pt already contains trained ctc_head/boundary_head weights
(unlike Day 33's day12_latest.pt, which was encoder-only and needed the
partial/prefix-filtered loader). This mirrors Day 36's stated pattern
("load_checkpoint(), not load_checkpoint_partial, since this is a
same-run resume, not a weight transfer") but load_checkpoint()'s exact
signature (return value, whether it returns the resumed step number,
optimizer-state handling) has not been directly re-confirmed against
train_utils.py in this session. Flagged as an assumption to verify on
first run, not silently trusted.

Usage:
    uv run python -m scripts.test_boundary_weight_2x_smoke
"""

import json
from pathlib import Path
-
import torch
from torch.utils.data import DataLoader

from src.models.sign_transformer import SignTransformer
from src.models.train_utils import load_checkpoint, get_device
from src.models.dataset import SignLanguageDataset, collate_fn
from src.models.combined_loss import compute_combined_loss

VOCAB_MERGED_PATH = "data/vocab_merged.json"
ASLLRP_TOKEN_TO_ID_PATH = "data/asllrp_token_to_id.json"
DECODER_VOCAB_PATH = "data/vocab.json"  # 1850 entries -- see Day 38 finding
BEST_VAL_CHECKPOINT_PATH = "checkpoints/month2_asllrp/best_val.pt"
ASLLRP_NPZ_DIR = "data/processed/asllrp/train"

NUM_STEPS = 150  # same length as Day 33's baseline run, for direct comparison

BASELINE_WEIGHT = 1.0   # what the real training run has used throughout
CANDIDATE_WEIGHT = 2.0  # Day 39's proposed new weight, being validated here


def run_smoke(boundary_loss_weight: float, label: str):
    """Runs NUM_STEPS locally at the given boundary_loss_weight, resumed
    from best_val.pt. Returns (ctc_losses, boundary_losses, grad_norms)
    for comparison. Does NOT save any checkpoint or mutate best_val.pt --
    read-only against the checkpoint, this is a validation run only."""
    device = get_device()

    with open(VOCAB_MERGED_PATH) as f:
        vocab_merged = json.load(f)
    with open(DECODER_VOCAB_PATH) as f:
        decoder_vocab = json.load(f)

    ctc_vocab_size = len(vocab_merged)
    blank_id = ctc_vocab_size
    bos_id = decoder_vocab["<BOS>"]
    eos_id = decoder_vocab["<EOS>"]
    pad_id = decoder_vocab["<PAD>"]

    # NOTE: vocab_size=len(decoder_vocab) here is derived, not the bare
    # 1850 literal train_month2_asllrp.py currently hardcodes (Day 38
    # finding, still not fixed at the source). Deriving it here so this
    # script doesn't propagate the same un-derived-literal issue.
    model = SignTransformer(
        vocab_size=len(decoder_vocab), ctc_vocab_size=ctc_vocab_size
    ).to(device)

    # ASSUMPTION FLAGGED (see module docstring): exact return signature
    # of load_checkpoint() not independently re-confirmed this session.
    # If this errors or the model doesn't come back finite, that itself
    # is the first thing to check -- don't silently proceed past it.
    load_checkpoint(model, BEST_VAL_CHECKPOINT_PATH)

    dataset = SignLanguageDataset(
        npz_dir=ASLLRP_NPZ_DIR,
        vocab_path=DECODER_VOCAB_PATH,
        asllrp_token_to_id_path=ASLLRP_TOKEN_TO_ID_PATH,
    )
    loader = DataLoader(
        dataset, batch_size=4, collate_fn=collate_fn,
        num_workers=2, persistent_workers=True, shuffle=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    ctc_losses, boundary_losses, grad_norms = [], [], []
    skipped_inf_batches = 0
    step = 0

    print(f"\n=== {label}: boundary_loss_weight={boundary_loss_weight}, "
          f"{NUM_STEPS} steps, resumed from {BEST_VAL_CHECKPOINT_PATH} ===\n")

    while step < NUM_STEPS:
        for batch in loader:
            if step >= NUM_STEPS:
                break

            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

            ctc_log_probs, boundary_logits = model.forward_ctc_and_boundary(
                batch["landmarks"], batch["encoder_attention_mask"]
            )

            total_loss, ctc_loss, boundary_loss = compute_combined_loss(
                ctc_log_probs, boundary_logits,
                batch["target_tokens"], batch["boundary_target"],
                batch["encoder_attention_mask"],
                bos_id, eos_id, pad_id, blank_id,
                boundary_loss_weight=boundary_loss_weight,
            )

            if not torch.isfinite(total_loss):
                skipped_inf_batches += 1
                step += 1
                continue

            optimizer.zero_grad()
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            ctc_losses.append(ctc_loss.item())
            boundary_losses.append(boundary_loss.item())
            grad_norms.append(float(grad_norm))

            if step % 25 == 0 or step == NUM_STEPS - 1:
                print(f"Step {step}: ctc_loss={ctc_loss.item():.4f}  "
                      f"boundary_loss={boundary_loss.item():.4f}  "
                      f"grad_norm={float(grad_norm):.3f}")

            step += 1

    print(f"\n--- {label} summary ({len(ctc_losses)} valid steps, "
          f"{skipped_inf_batches} skipped as non-finite) ---")
    print(f"ctc_loss:      start={ctc_losses[0]:.4f}  end={ctc_losses[-1]:.4f}")
    print(f"boundary_loss: start={boundary_losses[0]:.4f}  end={boundary_losses[-1]:.4f}  "
          f"delta={boundary_losses[-1]-boundary_losses[0]:.4f}")
    print(f"grad_norm:     mean={sum(grad_norms)/len(grad_norms):.3f}  "
          f"peak={max(grad_norms):.3f}")

    return ctc_losses, boundary_losses, grad_norms


def main():
    baseline = run_smoke(BASELINE_WEIGHT, "BASELINE (weight=1.0)")
    candidate = run_smoke(CANDIDATE_WEIGHT, "CANDIDATE (weight=2.0)")

    base_ctc, base_bnd, base_grad = baseline
    cand_ctc, cand_bnd, cand_grad = candidate

    print("\n" + "=" * 78)
    print("COMPARISON: weight=1.0 (baseline) vs weight=2.0 (candidate)")
    print("=" * 78)
    print(f"{'metric':<30} {'baseline':>15} {'candidate':>15}")
    print(f"{'ctc_loss end':<30} {base_ctc[-1]:>15.4f} {cand_ctc[-1]:>15.4f}")
    print(f"{'boundary_loss delta':<30} "
          f"{base_bnd[-1]-base_bnd[0]:>15.4f} {cand_bnd[-1]-cand_bnd[0]:>15.4f}")
    print(f"{'grad_norm mean':<30} "
          f"{sum(base_grad)/len(base_grad):>15.3f} {sum(cand_grad)/len(cand_grad):>15.3f}")
    print(f"{'grad_norm peak':<30} {max(base_grad):>15.3f} {max(cand_grad):>15.3f}")
    print("\nRead this before deciding anything:")
    print("- ctc_loss end should be roughly comparable between the two --")
    print("  a large regression means boundary reweighting is actively hurting")
    print("  CTC learning, not just sharing gradient budget reasonably.")
    print("- boundary_loss delta should be MORE negative (faster decline) for")
    print("  candidate than baseline, if the reweighting is doing its job.")
    print("- grad_norm peak/mean for candidate should stay in the same rough")
    print("  band as baseline -- a new sustained elevation would mean 2.0 is")
    print("  destabilizing, not just reweighting.")
    print("=" * 78)


if __name__ == "__main__":
    main()