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
    """
    Safely save a checkpoint.

    The checkpoint is first written to a temporary file.
    The temporary file is then loaded and validated.
    Only after successful validation is it atomically renamed
    to the final .pt path.

    Therefore, if torch.save() is interrupted or fails, the
    existing valid checkpoint is left untouched.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Temporary file lives beside the final checkpoint.
    tmp_path = path.with_suffix(path.suffix + ".tmp")

    checkpoint = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    # Remove stale temporary file from a previous interrupted save.
    try:
        if tmp_path.exists():
            tmp_path.unlink()
    except FileNotFoundError:
        pass

    try:
        # ---------------------------------------------------------
        # 1. Write checkpoint to temporary file
        # ---------------------------------------------------------
        torch.save(checkpoint, tmp_path)

        # ---------------------------------------------------------
        # 2. Basic file checks
        # ---------------------------------------------------------
        if not tmp_path.exists():
            raise RuntimeError(
                f"Temporary checkpoint was not created: {tmp_path}"
            )

        if tmp_path.stat().st_size == 0:
            raise RuntimeError(
                f"Temporary checkpoint is zero bytes: {tmp_path}"
            )

        # ---------------------------------------------------------
        # 3. Load it back to verify that the file is not corrupted
        # ---------------------------------------------------------
        test = torch.load(tmp_path, map_location="cpu")

        required_keys = {
            "step",
            "model_state_dict",
            "optimizer_state_dict",
        }

        if not required_keys.issubset(test.keys()):
            raise RuntimeError(
                f"Checkpoint validation failed: missing required keys "
                f"in {tmp_path}"
            )

        if test["step"] != step:
            raise RuntimeError(
                f"Checkpoint validation failed: expected step={step}, "
                f"got step={test['step']}"
            )

        # ---------------------------------------------------------
        # 4. ATOMIC REPLACE
        #
        # The existing .pt remains untouched until the new
        # checkpoint has successfully saved and validated.
        # ---------------------------------------------------------
        tmp_path.replace(path)

        print(
            f"[CHECKPOINT] VALID + ATOMIC SAVE: "
            f"{path.name} "
            f"({path.stat().st_size / 1e6:.2f} MB)"
        )

    except Exception:
        # ---------------------------------------------------------
        # Save failed.
        #
        # Delete ONLY the temporary file.
        # NEVER delete or overwrite the existing checkpoint.
        # ---------------------------------------------------------
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass

        raise

def load_checkpoint_partial(model, path, prefixes_to_load=("encoder.",), device=None):
    if device is None:
        device = get_device()
    checkpoint = torch.load(path, map_location=device)
    old_state = checkpoint["model_state_dict"]
    new_state = model.state_dict()

    loaded, skipped, new_only = [], [], []

    for name, param in old_state.items():
        if any(name.startswith(p) for p in prefixes_to_load):
            if name in new_state and new_state[name].shape == param.shape:
                new_state[name] = param
                loaded.append(name)
            else:
                reason = "not in new model" if name not in new_state else (
                    f"shape mismatch {tuple(param.shape)} vs {tuple(new_state[name].shape)}"
                )
                skipped.append(f"{name} ({reason})")
        else:
            skipped.append(f"{name} (prefix not in load list)")

    # Names that exist in the new model but had no counterpart in the
    # old checkpoint at all (e.g. ctc_head.*) -- these stay at their
    # fresh random init, but that fact must be stated explicitly, not
    # inferred from absence.
    for name in new_state:
        if name not in old_state:
            new_only.append(name)

    model.load_state_dict(new_state)

    print(f"[load_checkpoint_partial] Loaded {len(loaded)} params from {path}:")
    for n in loaded:
        print(f"    {n}")
    print(f"[load_checkpoint_partial] Skipped/reinitialized {len(skipped)} params (present in old checkpoint):")
    for n in skipped:
        print(f"    {n}")
    print(f"[load_checkpoint_partial] New params with no old-checkpoint counterpart, randomly initialized ({len(new_only)}):")
    for n in new_only:
        print(f"    {n}")

    return loaded, skipped, new_only


def load_checkpoint(model, optimizer, path, device=None):
    if device is None:
        device = get_device()
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["step"]

def get_device():
    """Single source of truth for device selection across all training scripts."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def move_batch_to_device(batch, device):
    """Batch is a dict of tensors (landmarks, encoder_attention_mask,
    target_tokens, target_padding_mask) -- move each to device in place."""
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}