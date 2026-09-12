"""
train_month2_asllrp.py

Days 34-35: first real Kaggle training kickoff on the corrected,
fully-validated combined CTC + boundary + merged-vocab setup.

Two distinct checkpoint operations, not to be confused:
  - FIRST LAUNCH ONLY: load_checkpoint_partial() transfers the encoder
    from checkpoints/day12_latest.pt (WLASL-trained). No optimizer
    state exists yet for this run.
  - EVERY RESUME AFTER THAT: load_checkpoint() restores this run's own
    checkpoint (full model + optimizer + step), NOT the old WLASL one.
    Re-running load_checkpoint_partial on a resume would silently
    throw away all ASLLRP training progress and optimizer state.

Usage:
    First launch:
        uv run python -m scripts.train_month2_asllrp
    Resume (auto-detects the latest checkpoint under CHECKPOINT_DIR):
        uv run python -m scripts.train_month2_asllrp --resume
"""

import argparse
import json
import re
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.models.sign_transformer import SignTransformer
from src.models.train_utils import (
    load_checkpoint_partial,
    load_checkpoint,
    save_checkpoint,
    get_device,
)
from src.models.dataset import SignLanguageDataset, collate_fn
from src.models.combined_loss import compute_combined_loss

VOCAB_MERGED_PATH = "data/vocab_merged.json"
DECODER_VOCAB_PATH = "data/vocab.json"  # 1850 entries -- old WLASL decoder vocab
ASLLRP_TOKEN_TO_ID_PATH = "data/asllrp_token_to_id.json"
OLD_WLASL_CHECKPOINT = "checkpoints/day12_latest.pt"
ASLLRP_NPZ_DIR = "data/processed/asllrp/train"

CHECKPOINT_DIR = Path("checkpoints/month2_asllrp")
LOG_PATH = Path("logs/month2_asllrp_training.log")

BATCH_SIZE = 4
LEARNING_RATE = 1e-4
BOUNDARY_LOSS_WEIGHT = 1.0  # Day 33 decision: kept as documented default,
                             # explicitly flagged to revisit if boundary_loss
                             # is still near ln(2)=0.693 after real training
GRAD_CLIP_MAX_NORM = 1.0
CHECKPOINT_EVERY_N_STEPS = 2000
LOG_EVERY_N_STEPS = 25


# ONLY matches normal checkpoints:
# step_0058000.pt
#
# Does NOT match:
# step_0058200_final.pt
# step_0058200.pt.tmp
# anything else.
_STEP_RE = re.compile(r"^step_(\d+)\.pt$")


def get_latest_checkpoint(checkpoint_dir):
    """
    Return the newest checkpoint that is actually valid.

    We NEVER trust the newest filename blindly.

    Each candidate is:
      1. matched against the normal checkpoint filename format
      2. loaded with torch.load()
      3. checked for required checkpoint keys
      4. checked that the internal step matches the filename step

    If the newest checkpoint is corrupted, we automatically fall
    back to the next-newest valid checkpoint.
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    candidates = []

    for path in checkpoint_dir.iterdir():
        if not path.is_file():
            continue

        match = _STEP_RE.match(path.name)

        if match:
            step_num = int(match.group(1))
            candidates.append((step_num, path))

    # Newest step first
    candidates.sort(key=lambda x: x[0], reverse=True)

    required_keys = {
        "step",
        "model_state_dict",
        "optimizer_state_dict",
    }

    for step_num, path in candidates:

        try:
            checkpoint = torch.load(
                path,
                map_location="cpu"
            )

        except Exception as e:
            print(
                f"[RESUME] Skipping corrupted checkpoint "
                f"{path.name}: {type(e).__name__}: {e}"
            )
            continue

        if not required_keys.issubset(checkpoint.keys()):
            print(
                f"[RESUME] Skipping {path.name}: "
                f"missing required keys"
            )
            continue

        checkpoint_step = checkpoint["step"]

        if checkpoint_step != step_num:
            print(
                f"[RESUME] Skipping {path.name}: "
                f"filename says step={step_num}, "
                f"but checkpoint contains step={checkpoint_step}"
            )
            continue

        print(
            f"[RESUME] Selected valid checkpoint: "
            f"{path.name} (step {step_num})"
        )

        return path

    print("[RESUME] No valid checkpoint found.")

    return None

KEEP_LAST_N_CHECKPOINTS = 3


def cleanup_old_checkpoints(checkpoint_dir, keep_last_n=KEEP_LAST_N_CHECKPOINTS):
    """Keep only the newest N normal checkpoints."""
    checkpoints = []

    for path in checkpoint_dir.iterdir():
        if not path.is_file():
            continue

        match = _STEP_RE.match(path.name)
        if match:
            step_num = int(match.group(1))
            checkpoints.append((step_num, path))

    checkpoints.sort(key=lambda x: x[0], reverse=True)

    for step_num, path in checkpoints[keep_last_n:]:
        try:
            path.unlink()
            log_line(f"[CHECKPOINT CLEANUP] Deleted old checkpoint: {path.name}")
        except Exception as e:
            log_line(
                f"[CHECKPOINT CLEANUP] Failed to delete {path.name}: "
                f"{type(e).__name__}: {e}"
            )


def log_line(msg):
    """Write to both console and a persistent log FILE -- per Day 34-35's
    plan explicitly: 'a full log file, not just console output, so a
    Day-18/19-style retroactive review is possible later.'"""
    print(msg)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true",
                         help="Resume from the latest checkpoint under CHECKPOINT_DIR")
    parser.add_argument("--max-steps-this-session", type=int, default=None,
                         help="Stop after N steps this session (for the explicit resume test)")
    args = parser.parse_args()

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
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    start_step = 0
    latest_ckpt = get_latest_checkpoint(CHECKPOINT_DIR)

    if args.resume:
        if latest_ckpt is None:
            raise RuntimeError(
                "--resume passed but no checkpoint found under "
                f"{CHECKPOINT_DIR}. First launch should not use --resume."
            )
        log_line(f"[RESUME] Loading full checkpoint: {latest_ckpt}")
        start_step = load_checkpoint(model, optimizer, latest_ckpt, device=device)
        log_line(f"[RESUME] Resumed at step {start_step}")
    else:
        if latest_ckpt is not None:
            raise RuntimeError(
                f"A checkpoint already exists at {latest_ckpt} but --resume "
                "was not passed. Re-running the first-launch path now would "
                "silently discard this run's progress and re-transfer only "
                "the encoder from the old WLASL checkpoint. Pass --resume, "
                "or delete the checkpoint directory if this is intentional."
            )
        log_line(f"[FIRST LAUNCH] Transferring encoder from {OLD_WLASL_CHECKPOINT}")
        loaded, skipped, new_only = load_checkpoint_partial(
            model, OLD_WLASL_CHECKPOINT, prefixes_to_load=("encoder.",)
        )
        log_line(f"[FIRST LAUNCH] Encoder transfer: {len(loaded)} loaded, "
                  f"{len(skipped)} skipped, {len(new_only)} new (fresh init)")

    dataset = SignLanguageDataset(
        npz_dir=ASLLRP_NPZ_DIR,
        vocab_path=VOCAB_MERGED_PATH,
        asllrp_token_to_id_path=ASLLRP_TOKEN_TO_ID_PATH,
    )
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn,
        num_workers=2, persistent_workers=True, shuffle=True,
    )

    log_line(f"=== Training session start: begin_step={start_step}, "
              f"resume={args.resume}, device={device} ===")

    step = start_step
    session_start_time = time.time()
    session_steps = 0
    ctc_losses, boundary_losses = [], []

    model.train()
    try:
        while True:
            for batch in loader:
                if args.max_steps_this_session is not None and session_steps >= args.max_steps_this_session:
                    raise StopIteration  # deliberate early stop, for the resume test

                batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

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
                    log_line(f"[WARNING] Step {step}: non-finite loss, skipping batch "
                              f"(video_ids={batch['video_ids']})")
                    step += 1
                    session_steps += 1
                    continue

                optimizer.zero_grad()
                total_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_MAX_NORM)
                optimizer.step()

                ctc_losses.append(ctc_loss.item())
                boundary_losses.append(boundary_loss.item())

                if step % LOG_EVERY_N_STEPS == 0:
                    elapsed = time.time() - session_start_time
                    steps_per_min = (session_steps + 1) / (elapsed / 60) if elapsed > 0 else 0.0
                    log_line(
                        f"Step {step}: ctc_loss={ctc_loss.item():.4f} "
                        f"boundary_loss={boundary_loss.item():.4f} "
                        f"grad_norm={grad_norm:.4f} "
                        f"steps/min={steps_per_min:.1f}"
                    )

                if step % CHECKPOINT_EVERY_N_STEPS == 0 and step > start_step:
                    ckpt_path = CHECKPOINT_DIR / f"step_{step:07d}.pt"

                    save_checkpoint(model, optimizer, step, ckpt_path)

                    log_line(f"[CHECKPOINT] Saved: {ckpt_path}")

                    cleanup_old_checkpoints(CHECKPOINT_DIR)

                step += 1
                session_steps += 1

    except StopIteration:
        pass
    except KeyboardInterrupt:
        log_line(f"[INTERRUPTED] Stopped at step {step} by user/session limit")
    finally:
        # No final-checkpoint save here on purpose.
        # Periodic CHECKPOINT_EVERY_N_STEPS saves are the
        # recovery mechanism.

        log_line(
            f"[SESSION TERMINATED] Stopped at step {step}. "
            f"Last periodic checkpoint should be at or before "
            f"step {(step // CHECKPOINT_EVERY_N_STEPS) * CHECKPOINT_EVERY_N_STEPS}."
        )

        elapsed = time.time() - session_start_time

        log_line(
            f"=== Session end: step={step}, "
            f"session_steps={session_steps}, "
            f"elapsed={elapsed/60:.1f} min, "
            f"steps/min={session_steps/(elapsed/60):.1f} ==="
        )

        if ctc_losses:
            log_line(
                f"This session: ctc_loss last={ctc_losses[-1]:.4f}, "
                f"boundary_loss last={boundary_losses[-1]:.4f}"
            )

if __name__ == "__main__":
    main()