"""
train_utils.py

Training loop infrastructure: optimizer/scheduler setup, loss
computation, checkpointing. Loss is straightforward cross-entropy on
gloss token prediction for this week's WLASL isolated-sign smoke test
-- the combined CTC + boundary-prediction loss (C2's real claim) is a
Month 2/ASLLRP-specific addition, not needed for this smoke test.
"""

import torch
import torch.nn as nn
from pathlib import Path


def compute_loss(logits, target_tokens, target_padding_mask, pad_id=0):
    """
    logits: (batch, tgt_len, vocab_size) -- model output
    target_tokens: (batch, tgt_len) -- ground truth token ids
    target_padding_mask: (batch, tgt_len) -- True/1 for real tokens

    Standard teacher-forced next-token cross-entropy: predict token[t+1]
    given tokens[0:t]. Shift logits/targets by one position accordingly.
    Padding positions are excluded via ignore_index, not the mask directly
    -- CrossEntropyLoss's ignore_index is the simpler, standard way to
    exclude padding from the loss.
    """
    # Shift: predict target_tokens[1:] from logits[:-1]
    shift_logits = logits[:, :-1, :].contiguous()
    shift_targets = target_tokens[:, 1:].contiguous()

    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)
    loss = loss_fn(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_targets.view(-1),
    )
    return loss


def build_optimizer(model, lr=1e-4, weight_decay=0.01):
    """AdamW -- standard, defensible starting point for a from-scratch
    transformer, per Day 11's plan. Not tuned yet; this is a smoke-test
    default, revisit in Month 2 if needed."""
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def build_scheduler(optimizer, warmup_steps=500, total_steps=10000):
    """Linear warmup, then cosine decay -- a common, reasonable default
    for training a transformer from scratch."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159265))).item())

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(model, optimizer, step, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, path)


def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["step"]