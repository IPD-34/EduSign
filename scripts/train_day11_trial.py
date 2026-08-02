"""
train_day11_trial.py

Day 11 validation: a short trial run (20-50 steps) confirming the
training loop actually executes, logs correctly, and loss is a real
(finite, non-NaN) number at every step. Not real training -- too few
steps to expect a decreasing loss, just checking nothing crashes or
immediately diverges.

Uses gradient clipping (max norm 1.0) as a standard stabilization
measure from the start, per Day 13's note that this is a reasonable
default, not something to add reactively only after seeing divergence.
"""

import json
import sys
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.config import WLASL_PROCESSED_DIR
from src.models.dataset import SignLanguageDataset, collate_fn
from src.models.sign_transformer import SignTransformer
from src.models.train_utils import compute_loss, build_optimizer, build_scheduler, save_checkpoint

TRIAL_STEPS = 30
LOG_EVERY = 5
CHECKPOINT_PATH = "checkpoints/day11_trial.pt"
GRAD_CLIP_NORM = 1.0


def main():
    with open("data/vocab.json", "r", encoding="utf-8") as f:
        token_to_id = json.load(f)
    vocab_size = len(token_to_id)
    pad_id = token_to_id["<PAD>"]

    dataset = SignLanguageDataset(WLASL_PROCESSED_DIR / "train")
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=partial(collate_fn, pad_id=pad_id),
    )

    model = SignTransformer(vocab_size=vocab_size, pad_token_id=pad_id)
    optimizer = build_optimizer(model)
    scheduler = build_scheduler(optimizer, warmup_steps=10, total_steps=TRIAL_STEPS)

    print(f"Starting {TRIAL_STEPS}-step trial run...")
    print(f"{'Step':>5} | {'Loss':>10} | {'Grad Norm':>10} | {'LR':>10}")
    print("-" * 45)

    data_iter = iter(dataloader)
    step = 0

    while step < TRIAL_STEPS:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        optimizer.zero_grad()

        logits = model(
            batch["landmarks"],
            batch["encoder_attention_mask"],
            batch["target_tokens"],
            batch["target_padding_mask"],
        )

        loss = compute_loss(logits, batch["target_tokens"], batch["target_padding_mask"], pad_id=pad_id)

        if not torch.isfinite(loss):
            print(f"[STEP {step}] NON-FINITE LOSS DETECTED: {loss.item()} -- stopping trial.")
            break

        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)

        optimizer.step()
        scheduler.step()

        if step % LOG_EVERY == 0 or step == TRIAL_STEPS - 1:
            current_lr = scheduler.get_last_lr()[0]
            print(f"{step:>5} | {loss.item():>10.4f} | {grad_norm.item():>10.4f} | {current_lr:>10.6f}")

        step += 1

    save_checkpoint(model, optimizer, step, CHECKPOINT_PATH)
    print(f"\nTrial run complete. Checkpoint saved to {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()