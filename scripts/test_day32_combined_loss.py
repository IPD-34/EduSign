"""
test_day32_combined_loss.py

Day 32 combined-loss smoke test: real ASLLRP batch, both loss terms
computed and printed SEPARATELY (not just the combined total, per
Day 32's own plan), backward pass, gradients checked for NaN/Inf.

Deliberately small-scale (a handful of batches, not a real training
run) -- Day 33 is where a corpus-wide length-mismatch audit and an
extended loss-balance check happen before real Kaggle quota is spent.
"""

import json
import torch
from torch.utils.data import DataLoader

from src.models.sign_transformer import SignTransformer
from src.models.train_utils import load_checkpoint_partial, get_device
from src.models.dataset import SignLanguageDataset, collate_fn
from src.models.combined_loss import compute_combined_loss

VOCAB_MERGED_PATH = "data/vocab_merged.json"
DECODER_VOCAB_PATH = "data/vocab.json"  # 1850 entries -- old WLASL decoder vocab
ASLLRP_TOKEN_TO_ID_PATH = "data/asllrp_token_to_id.json"
OLD_CHECKPOINT_PATH = "checkpoints/day12_latest.pt"
ASLLRP_NPZ_DIR = "data/processed/asllrp/train"


def main():
    device = get_device()

    with open(VOCAB_MERGED_PATH) as f:
        vocab_merged = json.load(f)
    with open(DECODER_VOCAB_PATH) as f:
        decoder_vocab = json.load(f)
    ctc_vocab_size = len(vocab_merged)
    blank_id = ctc_vocab_size
    bos_id = vocab_merged["<BOS>"]
    eos_id = vocab_merged["<EOS>"]
    pad_id = vocab_merged["<PAD>"]

    model = SignTransformer(vocab_size=len(decoder_vocab), ctc_vocab_size=ctc_vocab_size).to(device)
    load_checkpoint_partial(model, OLD_CHECKPOINT_PATH, prefixes_to_load=("encoder.",))

    dataset = SignLanguageDataset(
        npz_dir=ASLLRP_NPZ_DIR,
        vocab_path=VOCAB_MERGED_PATH,
        asllrp_token_to_id_path=ASLLRP_TOKEN_TO_ID_PATH,
    )
    # persistent_workers=True requires num_workers > 0
    loader = DataLoader(
        dataset, batch_size=4, collate_fn=collate_fn,
        num_workers=2, persistent_workers=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    print("Running 3 smoke-test steps...\n")
    for step, batch in enumerate(loader):
        if step >= 3:
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
            boundary_loss_weight=1.0,
        )

        optimizer.zero_grad()
        total_loss.backward()

        pre_clip_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        any_nonfinite = any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        optimizer.step()

        print(f"Step {step}: ctc_loss={ctc_loss.item():.4f}  "
              f"boundary_loss={boundary_loss.item():.4f}  "
              f"total={total_loss.item():.4f}  "
              f"max_grad_norm={pre_clip_norm:.4f}  "
              f"nonfinite_grads={any_nonfinite}")

        assert torch.isfinite(total_loss), "Non-finite total loss!"
        assert not any_nonfinite, "Non-finite gradients found!"

    print("\n[OK] 3 combined-loss steps completed, all finite, backward+step succeeded.")


if __name__ == "__main__":
    main()