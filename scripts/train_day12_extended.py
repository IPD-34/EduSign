"""
train_day12_extended.py

Follow-up to train_day12_smoke.py, after diagnose_gloss_position.py
confirmed the Day 12 checkpoint (step 25,240) is genuinely learning
-- true-gloss rank 12/1850, ~12x random-baseline probability mass on
a real training sample -- but hasn't converged far enough for
top-1 argmax accuracy to be a meaningful metric yet.

This script resumes from checkpoints/day12_latest.pt and runs a
further block of ADDITIONAL_STEPS on the same WLASL train split,
using the same optimizer/loss setup as train_day12_smoke.py. It is
intentionally almost identical to that script -- the only real changes
are: (1) it resumes rather than starting fresh (same as
train_day12_smoke.py's own resume path, just invoked deliberately
rather than as crash recovery), and (2) it targets a fixed number of
ADDITIONAL steps past whatever step the checkpoint is already at,
rather than a fixed total_steps computed from EPOCHS.

Per the Week 2/Day 12 precedent (5 -> 20 -> 40 epoch extensions, each
justified because the run is cheap and each showed continued
improvement with no plateau) -- this follows the same "cheap to check,
extend before concluding" discipline, this time to get top-1 accuracy
into a genuinely testable range before re-running the channel check.

Usage:
    uv run python train_day12_extended.py
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
# Config
# -------------------------------------------------------
BATCH_SIZE = 16
ADDITIONAL_STEPS = 100_000      # cheap on T4 (~14 min per 25k steps observed on Day 12)
WARMUP_STEPS = 500              # only used if total_steps schedule needs recomputing
LOG_EVERY = 200                 # steps -- coarser than Day 12's 20, since this run is longer
CHECKPOINT_EVERY = 2_000        # steps
GRAD_CLIP_NORM = 1.0

CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_PATH = CHECKPOINT_DIR / "day12_latest.pt"
LOG_PATH = Path("logs") / "day12_extended_training_log.txt"


def log(msg, f):
    print(msg)
    f.write(msg + "\n")
    f.flush()


def main():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    with open(LOG_PATH, "a", encoding="utf-8") as logf:
        log(f"\n{'='*60}\nDay 12 EXTENDED run started: {time.ctime()}\n{'='*60}", logf)

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
        optimizer = build_optimizer(model)

        if not CHECKPOINT_PATH.exists():
            log(f"[ERROR] No checkpoint found at {CHECKPOINT_PATH} -- run train_day12_smoke.py first.", logf)
            return

        start_step = load_checkpoint(model, optimizer, CHECKPOINT_PATH, device=device)
        log(f"Resumed from checkpoint at step {start_step}", logf)

        target_step = start_step + ADDITIONAL_STEPS
        # Scheduler is rebuilt fresh against the new target horizon. Warmup
        # is effectively already-satisfied (start_step >> WARMUP_STEPS), so
        # this mainly re-derives a cosine decay curve over the extended
        # horizon rather than re-warming up from near-zero LR.
        scheduler = build_scheduler(optimizer, warmup_steps=WARMUP_STEPS, total_steps=target_step)
        # Fast-forward the scheduler to start_step so LR picks up where a
        # continuous run would have been, rather than restarting the decay.
        for _ in range(start_step):
            scheduler.step()

        log(f"Extending training: {start_step} -> {target_step} ({ADDITIONAL_STEPS} additional steps)", logf)
        log(f"{'Step':>7} | {'Loss':>10} | {'Grad Norm':>10} | {'LR':>10}", logf)
        log("-" * 50, logf)

        step = start_step
        model.train()

        while step < target_step:
            for batch in dataloader:
                if step >= target_step:
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

                if step % CHECKPOINT_EVERY == 0 and step > start_step:
                    save_checkpoint(model, optimizer, step, CHECKPOINT_PATH)
                    log(f"[checkpoint saved at step {step}]", logf)

                step += 1

        save_checkpoint(model, optimizer, step, CHECKPOINT_PATH)
        log(f"\nExtended run complete at step {step}. Final checkpoint: {CHECKPOINT_PATH}", logf)
        log(f"Run ended: {time.ctime()}", logf)
        log(
            "\nNOTE: after this run, re-run diagnose_gloss_position.py on a few "
            "training samples to confirm true-gloss rank/probability has "
            "improved before re-running channel_check_score.py.",
            logf,
        )


if __name__ == "__main__":
    main()