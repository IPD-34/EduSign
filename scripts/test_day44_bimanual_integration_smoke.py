"""
scripts/test_day44_bimanual_integration_smoke.py

Day 44: local smoke test for the C3 bimanual gate wired into the real
training step (SignTransformer.forward_ctc_boundary_and_gate() +
combined_loss.compute_combined_loss_with_gate()).

No historical "Day 33 reference trajectory" file exists anywhere in this
repo (checked -- grep found nothing, and test_day33_extended_smoke.py
itself only prints to console, it never persisted numbers to a log file).
Rather than trust a remembered/pasted figure, this script runs the
UNMODIFIED baseline path (use_bimanual_gate=False -- mathematically the
exact Day 33 model) and the gate-enabled path back-to-back, in the same
session, with the same seed and the same batch order, so the comparison
isolates exactly one variable: the gate routing mechanism itself.

Both runs:
  - start from the same encoder init (RNG order is identical up to the
    point the gate's extra parameters are constructed, which happens
    AFTER decoder/ctc_head/boundary_head in SignTransformer.__init__)
  - load the same WLASL encoder transfer (checkpoints/day12_latest.pt)
  - see batches in the same order (fixed DataLoader generator seed)

CPU/local scale only -- no Kaggle quota spent, same discipline as Days
32-33's local validation.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader

from src.models.sign_transformer import SignTransformer
from src.models.train_utils import load_checkpoint_partial, get_device
from src.models.dataset import SignLanguageDataset, collate_fn
from src.models.combined_loss import compute_combined_loss, compute_combined_loss_with_gate

VOCAB_MERGED_PATH = "data/vocab_merged.json"
DECODER_VOCAB_PATH = "data/vocab.json"  # 1850 entries -- old WLASL decoder vocab
ASLLRP_TOKEN_TO_ID_PATH = "data/asllrp_token_to_id.json"
OLD_CHECKPOINT_PATH = "checkpoints/day12_latest.pt"
ASLLRP_NPZ_DIR = "data/processed/asllrp/train"
NUM_STEPS = 60  # "a few dozen to ~100 steps" per Day 44's brief
BOUNDARY_LOSS_WEIGHT = 1.0
SEED = 42


def run_smoke(use_gate: bool, ctc_vocab_size, decoder_vocab_size, bos_id, eos_id, pad_id, blank_id, device):
    torch.manual_seed(SEED)
    model = SignTransformer(
        vocab_size=decoder_vocab_size, ctc_vocab_size=ctc_vocab_size, use_bimanual_gate=use_gate,
    ).to(device)
    loaded, skipped, new_only = load_checkpoint_partial(
        model, OLD_CHECKPOINT_PATH, prefixes_to_load=("encoder.",)
    )
    print(f"  encoder transfer: {len(loaded)} loaded, {len(skipped)} skipped, "
          f"{len(new_only)} new (fresh init)")

    dataset = SignLanguageDataset(
        npz_dir=ASLLRP_NPZ_DIR, vocab_path=VOCAB_MERGED_PATH,
        asllrp_token_to_id_path=ASLLRP_TOKEN_TO_ID_PATH,
    )
    gen = torch.Generator()
    gen.manual_seed(SEED)
    loader = DataLoader(
        dataset, batch_size=4, collate_fn=collate_fn, shuffle=True, generator=gen,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    ctc_losses, boundary_losses, gate_entropies = [], [], []
    gate_logit_samples = []
    skipped_inf = 0
    step = 0

    while step < NUM_STEPS:
        for batch in loader:
            if step >= NUM_STEPS:
                break
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

            if use_gate:
                ctc_log_probs, boundary_logits, gate_logits = model.forward_ctc_boundary_and_gate(
                    batch["landmarks"], batch["encoder_attention_mask"]
                )
                total_loss, ctc_loss, boundary_loss, gate_entropy = compute_combined_loss_with_gate(
                    ctc_log_probs, boundary_logits, gate_logits,
                    batch["target_tokens"], batch["boundary_target"],
                    batch["encoder_attention_mask"],
                    bos_id, eos_id, pad_id, blank_id,
                    boundary_loss_weight=BOUNDARY_LOSS_WEIGHT,
                    gate_reg_weight=0.0,  # OFF by default -- see combined_loss.py docstring
                )
                mask = batch["encoder_attention_mask"]
                gate_logit_samples.append(gate_logits[mask].detach())
                gate_entropies.append(gate_entropy.item())
            else:
                ctc_log_probs, boundary_logits = model.forward_ctc_and_boundary(
                    batch["landmarks"], batch["encoder_attention_mask"]
                )
                total_loss, ctc_loss, boundary_loss = compute_combined_loss(
                    ctc_log_probs, boundary_logits,
                    batch["target_tokens"], batch["boundary_target"],
                    batch["encoder_attention_mask"],
                    bos_id, eos_id, pad_id, blank_id,
                    boundary_loss_weight=BOUNDARY_LOSS_WEIGHT,
                )

            if not torch.isfinite(total_loss):
                skipped_inf += 1
                step += 1
                continue

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            ctc_losses.append(ctc_loss.item())
            boundary_losses.append(boundary_loss.item())

            if step % 25 == 0 or step == NUM_STEPS - 1:
                extra = f"  gate_entropy={gate_entropies[-1]:.4f}" if use_gate else ""
                print(f"  step {step}: ctc_loss={ctc_loss.item():.4f}  "
                      f"boundary_loss={boundary_loss.item():.4f}{extra}")
            step += 1

    result = {
        "ctc_losses": ctc_losses,
        "boundary_losses": boundary_losses,
        "skipped_inf": skipped_inf,
    }
    if use_gate:
        all_gate_logits = torch.cat(gate_logit_samples)
        p_symmetric = torch.sigmoid(all_gate_logits)
        result["gate_p_mean"] = p_symmetric.mean().item()
        result["gate_p_std"] = p_symmetric.std().item()
        result["gate_pct_symmetric_leaning"] = (p_symmetric > 0.5).float().mean().item() * 100
        result["gate_entropy_mean"] = sum(gate_entropies) / len(gate_entropies)
    return result


def summarize(name, r):
    ctc, bnd = r["ctc_losses"], r["boundary_losses"]
    print(f"\n--- {name} summary ({len(ctc)} valid steps, {r['skipped_inf']} skipped non-finite) ---")
    print(f"  ctc_loss:      start={ctc[0]:.2f}  end={ctc[-1]:.2f}  "
          f"mean_last_10={sum(ctc[-10:])/len(ctc[-10:]):.2f}")
    print(f"  boundary_loss: start={bnd[0]:.4f}  end={bnd[-1]:.4f}  "
          f"mean_last_10={sum(bnd[-10:])/len(bnd[-10:]):.4f}")
    if "gate_p_mean" in r:
        print(f"  gate P(symmetric): mean={r['gate_p_mean']:.4f}  std={r['gate_p_std']:.4f}  "
              f"symmetric-leaning(P>0.5)={r['gate_pct_symmetric_leaning']:.1f}% of positions")
        print(f"  gate_entropy (mean over run, max=ln(2)=0.6931): {r['gate_entropy_mean']:.4f}")


def main():
    device = get_device()
    with open(VOCAB_MERGED_PATH) as f:
        vocab_merged = json.load(f)
    with open(DECODER_VOCAB_PATH) as f:
        decoder_vocab = json.load(f)
    ctc_vocab_size = len(vocab_merged)
    decoder_vocab_size = len(decoder_vocab)
    blank_id = ctc_vocab_size
    bos_id, eos_id, pad_id = vocab_merged["<BOS>"], vocab_merged["<EOS>"], vocab_merged["<PAD>"]

    print(f"=== Day 44 smoke test: {NUM_STEPS} steps each, seed={SEED}, device={device} ===\n")

    print("[1/2] BASELINE (use_bimanual_gate=False -- mathematically = pre-Day-44 model)")
    baseline = run_smoke(False, ctc_vocab_size, decoder_vocab_size, bos_id, eos_id, pad_id, blank_id, device)

    print("\n[2/2] GATED (use_bimanual_gate=True)")
    gated = run_smoke(True, ctc_vocab_size, decoder_vocab_size, bos_id, eos_id, pad_id, blank_id, device)

    summarize("BASELINE (no gate)", baseline)
    summarize("GATED (Day 44 integration)", gated)

    print("\n=== Regression check ===")
    ctc_end_diff = gated["ctc_losses"][-1] - baseline["ctc_losses"][-1]
    bnd_end_diff = gated["boundary_losses"][-1] - baseline["boundary_losses"][-1]
    print(f"  ctc_loss end-of-run:      baseline={baseline['ctc_losses'][-1]:.4f}  "
          f"gated={gated['ctc_losses'][-1]:.4f}  diff={ctc_end_diff:+.4f}")
    print(f"  boundary_loss end-of-run: baseline={baseline['boundary_losses'][-1]:.4f}  "
          f"gated={gated['boundary_losses'][-1]:.4f}  diff={bnd_end_diff:+.4f}")
    print("  (boundary_loss near ln(2)=0.6931 at this step count is expected/unchanged --")
    print("   boundary head has barely trained yet in either run, same as Day 33's own finding.)")


if __name__ == "__main__":
    main()
