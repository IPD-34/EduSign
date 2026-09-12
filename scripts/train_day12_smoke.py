"""
train_day12_smoke.py

Day 12: real smoke-training run on the FULL WLASL train split (not the
30-step Day 11 correctness check). Goal is to confirm the architecture
trains stably over a longer, real horizon -- this is the run Day 14's
hard non-divergence gate is checked against.

Designed to run unattended (e.g. Kaggle "Save & Run All" background
commit): periodic checkpointing, periodic log flushing to a file (not
just stdout, which is lost if the session disconnects), and safe to
resume from the last checkpoint if interrupted.
"""

import json
import sys
import time
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.config import WLASL_PROCESSED_DIR
from src.models.dataset import SignLanguageDataset, collate_fn
from src.models.sign_transformer import SignTransformer
from src.models.train_utils import (
    compute_loss, build_optimizer, build_scheduler,
    save_checkpoint, load_checkpoint, get_device, move_batch_to_device,
)

# -------------------------------------------------------
# Config -- adjust EPOCHS/BATCH_SIZE if GPU memory is tight
# -------------------------------------------------------
BATCH_SIZE = 16
EPOCHS = 5                      # smoke test, not full convergence -- revisit in Month 2
WARMUP_STEPS = 500
LOG_EVERY = 20                  # steps
CHECKPOINT_EVERY = 200          # steps -- frequent enough to survive a disconnect
GRAD_CLIP_NORM = 1.0

CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_PATH = CHECKPOINT_DIR / "day12_latest.pt"
LOG_PATH = Path("logs") / "day12_training_log.txt"


def log(msg, f):
    """Write to both stdout and a file, flushing immediately -- so an
    unattended/background run leaves a real record even if the
    interactive session disconnects."""
    print(msg)
    f.write(msg + "\n")
    f.flush()


def main():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    with open(LOG_PATH, "a", encoding="utf-8") as logf:
        log(f"\n{'='*60}\nDay 12 run started: {time.ctime()}\n{'='*60}", logf)

        with open("data/vocab.json", "r", encoding="utf-8") as f:
            token_to_id = json.load(f)
        vocab_size = len(token_to_id)
        pad_id = token_to_id["<PAD>"]
        log(f"Vocab size: {vocab_size}", logf)

        dataset = SignLanguageDataset(WLASL_PROCESSED_DIR / "train")
        log(f"Train samples: {len(dataset)}", logf)

        dataloader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=partial(collate_fn, pad_id=pad_id),
            num_workers=2,
        )

        device = get_device()
        log(f"Using device: {device}", logf)
        if device.type == "cuda":
            log(f"GPU: {torch.cuda.get_device_name(0)}", logf)

        model = SignTransformer(vocab_size=vocab_size, pad_token_id=pad_id)
        model.to(device)

        total_steps = len(dataloader) * EPOCHS
        optimizer = build_optimizer(model)
        scheduler = build_scheduler(optimizer, warmup_steps=WARMUP_STEPS, total_steps=total_steps)

        # Resume support: if a checkpoint already exists (e.g. previous
        # run got interrupted), pick up from there rather than restarting.
        start_step = 0
        if CHECKPOINT_PATH.exists():
            start_step = load_checkpoint(model, optimizer, CHECKPOINT_PATH, device=device)
            log(f"Resumed from checkpoint at step {start_step}", logf)

        log(f"Total planned steps: {total_steps} ({EPOCHS} epochs x {len(dataloader)} steps/epoch)", logf)
        log(f"{'Step':>7} | {'Loss':>10} | {'Grad Norm':>10} | {'LR':>10}", logf)
        log("-" * 50, logf)

        step = start_step
        model.train()

        while step < total_steps:
            for batch in dataloader:
                if step >= total_steps:
                    break

                batch = move_batch_to_device(batch, device)
                optimizer.zero_grad()

                logits = model(
                    batch["landmarks"],
                    batch["encoder_attention_mask"],
                    batch["target_tokens"],
                    batch["target_padding_mask"],
                )

                loss = compute_loss(
                    logits, batch["target_tokens"], batch["target_padding_mask"], pad_id=pad_id
                )

                if not torch.isfinite(loss):
                    log(f"[STEP {step}] NON-FINITE LOSS: {loss.item()} -- stopping.", logf)
                    save_checkpoint(model, optimizer, step, CHECKPOINT_PATH)
                    return

                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                optimizer.step()
                scheduler.step()

                if step % LOG_EVERY == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    log(f"{step:>7} | {loss.item():>10.4f} | {grad_norm.item():>10.4f} | {current_lr:>10.6f}", logf)

                if step % CHECKPOINT_EVERY == 0 and step > 0:
                    save_checkpoint(model, optimizer, step, CHECKPOINT_PATH)
                    log(f"[checkpoint saved at step {step}]", logf)

                step += 1

        save_checkpoint(model, optimizer, step, CHECKPOINT_PATH)
        log(f"\nDay 12 run complete at step {step}. Final checkpoint: {CHECKPOINT_PATH}", logf)
        log(f"Run ended: {time.ctime()}", logf)


if __name__ == "__main__":
    main()