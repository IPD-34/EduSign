"""
train_month2_asllrp_gated.py

Stage 1 of the C3 real-training task: train the bimanual gate
(BimanualGateHead, use_bimanual_gate=True) for real, jointly with the
already-stable CTC + boundary losses, resuming from best_val.pt (the
checkpoint of record) rather than the old WLASL day12_latest.pt.

Deliberately a SEPARATE script from train_month2_asllrp.py, not a flag
bolted onto it -- this keeps the already-verified ungated training path
completely untouched (zero risk of a gate-related change silently
altering the run that produced best_val.pt), mirroring how Day 44 kept
compute_combined_loss() itself unmodified when adding
compute_combined_loss_with_gate().

--- CHECKPOINT ISOLATION, STATED EXPLICITLY ---
  - BEST_VAL_CHECKPOINT_PATH = checkpoints/month2_asllrp/best_val_gated.pt
    (+ best_val_gated_meta.json) -- as specified: lives alongside the
    original best_val.pt so both a gate-off and gate-on checkpoint exist
    from a comparable point, WITHOUT ever overwriting best_val.pt itself
    (different filename, and save_checkpoint()/get_latest_checkpoint()
    only ever touch the path they're explicitly given).
  - CHECKPOINT_DIR (step-numbered rolling checkpoints, used for --resume)
    = checkpoints/month2_asllrp_gated/ -- a SEPARATE directory from the
    original checkpoints/month2_asllrp/. Reason: get_latest_checkpoint()
    scans an entire directory for step_XXXXXXX.pt files: if this run's
    periodic checkpoints landed in the SAME directory as the original
    run's (step_0010000.pt, step_0012000.pt, step_0014000.pt), a
    --resume on either run could pick up the other run's checkpoint by
    step-number ordering alone. A separate directory makes that
    structurally impossible rather than relying on remembering not to
    mix them up.

--- FIRST-LAUNCH TRANSFER, DIFFERENT FROM THE ORIGINAL SCRIPT ---
The original train_month2_asllrp.py transfers only encoder.* from the
old WLASL checkpoint (day12_latest.pt) on first launch, since CTC/
boundary heads didn't exist there. Here, encoder.*, ctc_head.*, and
boundary_head.* all already exist and are trained, in best_val.pt --
those three prefixes are loaded from best_val.pt; bimanual_gate_head.*
has no counterpart there and stays freshly initialized. This exactly
mirrors run_sign_level_ablation()'s load pattern in
ablation_score_bimanual_gate.py and Day 44's smoke test.

--- TRAINING SCHEME: JOINT, UNSUPERVISED (Day 44 decision, unchanged) ---
gate_reg_weight=0.0 always -- Person D's symmetric/asymmetric split is
for Stage 3's evaluation ONLY. Using it here, even as a soft signal,
would make Stage 3's ablation circular (scoring a gate against the exact
labels it was partly trained to match). The gate's only training signal
is the gradient reaching bimanual_gate_head through how routed_features
change ctc_loss/boundary_loss.

--- boundary_loss_weight: LEFT AT 1.0, EXPLICITLY, NOT SILENTLY ---
Day 39 flagged boundary_loss drifting off the ln(2) baseline as needing
a real decision, never made. This script does not resolve that decision
by omission -- it is left at 1.0 here specifically so the gate is the
ONLY new variable in this run, not a confound alongside a reweighted
boundary term. The open item stays open.

Usage:
    First launch:
        uv run python -m scripts.train_month2_asllrp_gated
    Resume:
        uv run python -m scripts.train_month2_asllrp_gated --resume
"""

import argparse
import json
import re
import signal
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.sign_transformer import SignTransformer
from src.models.train_utils import (
    load_checkpoint_partial,
    load_checkpoint,
    save_checkpoint,
    get_device,
)
from src.models.dataset import SignLanguageDataset, collate_fn
from src.models.combined_loss import compute_combined_loss_with_gate

VOCAB_MERGED_PATH = "data/vocab_merged.json"
DECODER_VOCAB_PATH = "data/vocab.json"
ASLLRP_TOKEN_TO_ID_PATH = "data/asllrp_token_to_id.json"
BASE_CHECKPOINT = "checkpoints/month2_asllrp/best_val.pt"  # checkpoint of record -- NOT day12_latest.pt
ASLLRP_NPZ_DIR = "data/processed/asllrp/train"
ASLLRP_VAL_NPZ_DIR = "data/processed/asllrp/val"

CHECKPOINT_DIR = Path("checkpoints/month2_asllrp_gated")  # ISOLATED from the ungated run -- see module docstring
LOG_PATH = Path("logs/month2_asllrp_gated_training.log")  # separate log file, same reason

BATCH_SIZE = 4
LEARNING_RATE = 1e-4
BOUNDARY_LOSS_WEIGHT = 1.0  # left unchanged, explicitly -- see module docstring
GATE_REG_WEIGHT = 0.0       # joint/unsupervised, Day 44 decision -- see module docstring
GRAD_CLIP_MAX_NORM = 1.0
CHECKPOINT_EVERY_N_STEPS = 2000
LOG_EVERY_N_STEPS = 25

CHECKPOINT_EVERY_N_SECONDS = 15 * 60
TIME_LIMIT_SECONDS = 5 * 60 * 60
SAFETY_BUFFER_SECONDS = 15 * 60
EFFECTIVE_TRAIN_LIMIT_SECONDS = TIME_LIMIT_SECONDS - SAFETY_BUFFER_SECONDS

EVAL_EVERY_N_STEPS = 2000
BEST_VAL_CHECKPOINT_PATH = Path("checkpoints/month2_asllrp/best_val_gated.pt")
BEST_VAL_META_PATH = Path("checkpoints/month2_asllrp/best_val_gated_meta.json")

DEFAULT_EARLY_STOP_PATIENCE = 5

_STEP_RE = re.compile(r"^step_(\d+)\.pt$")


def get_latest_checkpoint(checkpoint_dir):
    """Identical logic to train_month2_asllrp.py's version -- see that
    file for full rationale. Operates only on CHECKPOINT_DIR (the
    isolated gated-run directory), never touches the original run's."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    candidates = []
    for path in checkpoint_dir.iterdir():
        if not path.is_file():
            continue
        match = _STEP_RE.match(path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    candidates.sort(key=lambda x: x[0], reverse=True)

    required_keys = {"step", "model_state_dict", "optimizer_state_dict"}
    for step_num, path in candidates:
        try:
            checkpoint = torch.load(path, map_location="cpu")
        except Exception as e:
            print(f"[RESUME] Skipping corrupted checkpoint {path.name}: {type(e).__name__}: {e}")
            continue
        if not required_keys.issubset(checkpoint.keys()):
            print(f"[RESUME] Skipping {path.name}: missing required keys")
            continue
        if checkpoint["step"] != step_num:
            print(f"[RESUME] Skipping {path.name}: filename says step={step_num}, "
                  f"but checkpoint contains step={checkpoint['step']}")
            continue
        print(f"[RESUME] Selected valid checkpoint: {path.name} (step {step_num})")
        return path
    print("[RESUME] No valid checkpoint found.")
    return None


KEEP_LAST_N_CHECKPOINTS = 3


def cleanup_old_checkpoints(checkpoint_dir, keep_last_n=KEEP_LAST_N_CHECKPOINTS):
    checkpoints = []
    for path in checkpoint_dir.iterdir():
        if not path.is_file():
            continue
        match = _STEP_RE.match(path.name)
        if match:
            checkpoints.append((int(match.group(1)), path))
    checkpoints.sort(key=lambda x: x[0], reverse=True)
    for step_num, path in checkpoints[keep_last_n:]:
        try:
            path.unlink()
            log_line(f"[CHECKPOINT CLEANUP] Deleted old checkpoint: {path.name}")
        except Exception as e:
            log_line(f"[CHECKPOINT CLEANUP] Failed to delete {path.name}: {type(e).__name__}: {e}")


def log_line(msg):
    print(msg)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


@torch.no_grad()
def compute_val_ctc_loss_gated(model, val_loader, blank_id, device):
    """Same metric definition as train_month2_asllrp.py's
    compute_val_ctc_loss (mean CTC loss per gloss token, zero_infinity=True)
    -- kept numerically comparable to best_val.pt's own val_loss (5.5040)
    on purpose, so 'is the gated model at least as good on held-out CTC
    recognition' is a real, apples-to-apples comparison. Uses
    forward_ctc_boundary_and_gate (gate ACTIVE during eval, matching how
    the model is actually used) and discards the boundary/gate outputs --
    only the CTC term is being tracked for early-stopping purposes here."""
    was_training = model.training
    model.eval()

    ctc_loss_fn = nn.CTCLoss(blank=blank_id, zero_infinity=True, reduction="sum")
    total_loss, total_targets = 0.0, 0

    for batch in val_loader:
        landmarks = batch["landmarks"].to(device)
        enc_mask = batch["encoder_attention_mask"].to(device)

        ctc_log_probs, _boundary_logits, _gate_logits = model.forward_ctc_boundary_and_gate(
            landmarks, enc_mask
        )
        log_probs_ctc = ctc_log_probs.transpose(0, 1)
        input_lengths = enc_mask.sum(dim=1).cpu()

        target_tokens = batch["target_tokens"]
        target_mask = batch["target_padding_mask"]

        targets, target_lengths = [], []
        for i in range(target_tokens.shape[0]):
            real = target_tokens[i][target_mask[i]]
            gloss_only = real[1:-1]
            targets.append(gloss_only)
            target_lengths.append(len(gloss_only))

        targets_flat = torch.cat(targets)
        target_lengths_t = torch.tensor(target_lengths, dtype=torch.long)

        loss = ctc_loss_fn(log_probs_ctc.cpu(), targets_flat, input_lengths, target_lengths_t)
        if torch.isfinite(loss):
            total_loss += loss.item()
            total_targets += sum(target_lengths)

    if was_training:
        model.train()
    return total_loss / max(total_targets, 1)


def load_best_val_meta():
    if BEST_VAL_META_PATH.exists():
        try:
            with open(BEST_VAL_META_PATH) as f:
                meta = json.load(f)
            return (
                meta.get("val_loss", float("inf")),
                meta.get("step"),
                meta.get("patience_counter", 0),
            )
        except Exception as e:
            log_line(f"[VAL] Could not read {BEST_VAL_META_PATH}: {type(e).__name__}: {e}. "
                      f"Starting best-val tracking fresh.")
    return float("inf"), None, 0


def save_best_val_meta(step, val_loss, patience_counter):
    BEST_VAL_META_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = BEST_VAL_META_PATH.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump({"step": step, "val_loss": val_loss, "patience_counter": patience_counter}, f)
    tmp.replace(BEST_VAL_META_PATH)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-steps-this-session", type=int, default=None)
    parser.add_argument("--early-stop-patience", type=int, default=DEFAULT_EARLY_STOP_PATIENCE)
    parser.add_argument("--eval-every-n-steps", type=int, default=EVAL_EVERY_N_STEPS)
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

    model = SignTransformer(
        vocab_size=len(decoder_vocab), ctc_vocab_size=ctc_vocab_size, use_bimanual_gate=True,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    start_step = 0
    latest_ckpt = get_latest_checkpoint(CHECKPOINT_DIR)

    if args.resume:
        if latest_ckpt is None:
            raise RuntimeError(
                f"--resume passed but no checkpoint found under {CHECKPOINT_DIR}. "
                "First launch should not use --resume."
            )
        log_line(f"[RESUME] Loading full checkpoint: {latest_ckpt}")
        start_step = load_checkpoint(model, optimizer, latest_ckpt, device=device)
        log_line(f"[RESUME] Resumed at step {start_step}")
    else:
        if latest_ckpt is not None:
            raise RuntimeError(
                f"A checkpoint already exists at {latest_ckpt} but --resume was not passed. "
                "Pass --resume, or clear the gated checkpoint directory if this is intentional."
            )
        log_line(f"[FIRST LAUNCH] Transferring encoder/ctc_head/boundary_head from {BASE_CHECKPOINT} "
                  f"(checkpoint of record) -- bimanual_gate_head stays freshly initialized.")
        loaded, skipped, new_only = load_checkpoint_partial(
            model, BASE_CHECKPOINT, prefixes_to_load=("encoder.", "ctc_head.", "boundary_head.")
        )
        log_line(f"[FIRST LAUNCH] Transfer: {len(loaded)} loaded, {len(skipped)} skipped, "
                  f"{len(new_only)} new (bimanual_gate_head expected here)")

    dataset = SignLanguageDataset(
        npz_dir=ASLLRP_NPZ_DIR, vocab_path=VOCAB_MERGED_PATH,
        asllrp_token_to_id_path=ASLLRP_TOKEN_TO_ID_PATH,
    )
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn,
        num_workers=2, persistent_workers=True, shuffle=True, pin_memory=True,
    )

    val_dataset = SignLanguageDataset(
        npz_dir=ASLLRP_VAL_NPZ_DIR, vocab_path=VOCAB_MERGED_PATH,
        asllrp_token_to_id_path=ASLLRP_TOKEN_TO_ID_PATH,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn,
        shuffle=False, pin_memory=True,
    )
    log_line(f"[VAL] Loaded val set: {len(val_dataset)} utterances, eval every {args.eval_every_n_steps} steps")

    best_val_loss, best_val_step, patience_counter = load_best_val_meta()
    if best_val_step is not None:
        log_line(f"[VAL] Resuming best-val tracking: current best is step {best_val_step} "
                  f"(val_loss={best_val_loss:.4f}), patience_counter={patience_counter}/{args.early_stop_patience}")
    else:
        log_line(f"[VAL] No prior best_val_gated_meta.json -- starting fresh at val_loss=inf, patience_counter=0.")
    if args.early_stop_patience <= 0:
        log_line("[VAL] Early stopping DISABLED (--early-stop-patience <= 0).")

    log_line(f"=== GATED training session start: begin_step={start_step}, resume={args.resume}, "
              f"device={device}, boundary_loss_weight={BOUNDARY_LOSS_WEIGHT}, "
              f"gate_reg_weight={GATE_REG_WEIGHT} ===")
    log_line(f"[TIME BUDGET] hard stop at {EFFECTIVE_TRAIN_LIMIT_SECONDS/60:.0f} min; "
              f"time-based checkpoint floor every {CHECKPOINT_EVERY_N_SECONDS/60:.0f} min")

    step = start_step
    session_start_time = time.time()
    session_steps = 0
    ctc_losses, boundary_losses, gate_entropies = [], [], []
    last_checkpoint_time = session_start_time

    _step_box = {"value": step}

    def _handle_sigterm(signum, frame):
        current_step = _step_box["value"]
        log_line(f"[SIGTERM] Received at step {current_step}. Attempting emergency checkpoint save.")
        try:
            ckpt_path = CHECKPOINT_DIR / f"step_{current_step:07d}.pt"
            save_checkpoint(model, optimizer, current_step, ckpt_path)
            log_line(f"[SIGTERM] Emergency checkpoint saved: {ckpt_path}")
        except Exception as e:
            log_line(f"[SIGTERM] Emergency save FAILED: {type(e).__name__}: {e}.")
        sys.exit(0)

    signal.signal(signal.SIGTERM, _handle_sigterm)

    model.train()
    try:
        while True:
            for batch in loader:
                if args.max_steps_this_session is not None and session_steps >= args.max_steps_this_session:
                    raise StopIteration

                _step_box["value"] = step
                elapsed = time.time() - session_start_time

                if elapsed >= EFFECTIVE_TRAIN_LIMIT_SECONDS:
                    log_line(f"[TIME BUDGET] {elapsed/60:.1f} min elapsed, reached effective limit. "
                              f"Saving final checkpoint and exiting cleanly at step {step}.")
                    ckpt_path = CHECKPOINT_DIR / f"step_{step:07d}.pt"
                    save_checkpoint(model, optimizer, step, ckpt_path)
                    log_line(f"[CHECKPOINT] Time-budget final save: {ckpt_path}")
                    cleanup_old_checkpoints(CHECKPOINT_DIR)
                    last_checkpoint_time = time.time()
                    raise StopIteration

                batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

                ctc_log_probs, boundary_logits, gate_logits = model.forward_ctc_boundary_and_gate(
                    batch["landmarks"], batch["encoder_attention_mask"]
                )
                total_loss, ctc_loss, boundary_loss, gate_entropy = compute_combined_loss_with_gate(
                    ctc_log_probs, boundary_logits, gate_logits,
                    batch["target_tokens"], batch["boundary_target"],
                    batch["encoder_attention_mask"],
                    bos_id, eos_id, pad_id, blank_id,
                    boundary_loss_weight=BOUNDARY_LOSS_WEIGHT,
                    gate_reg_weight=GATE_REG_WEIGHT,
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
                gate_entropies.append(gate_entropy.item())

                if step % LOG_EVERY_N_STEPS == 0:
                    steps_per_min = (session_steps + 1) / (elapsed / 60) if elapsed > 0 else 0.0
                    log_line(
                        f"Step {step}: ctc_loss={ctc_loss.item():.4f} "
                        f"boundary_loss={boundary_loss.item():.4f} "
                        f"gate_entropy={gate_entropy.item():.4f} "
                        f"grad_norm={grad_norm:.4f} "
                        f"steps/min={steps_per_min:.1f}"
                    )

                step_triggered = (step % CHECKPOINT_EVERY_N_STEPS == 0 and step > start_step)
                time_triggered = (time.time() - last_checkpoint_time >= CHECKPOINT_EVERY_N_SECONDS)

                if step_triggered or time_triggered:
                    ckpt_path = CHECKPOINT_DIR / f"step_{step:07d}.pt"
                    save_checkpoint(model, optimizer, step, ckpt_path)
                    trigger_reason = "step-count" if step_triggered else "time-floor"
                    log_line(f"[CHECKPOINT] Saved ({trigger_reason}): {ckpt_path}")
                    cleanup_old_checkpoints(CHECKPOINT_DIR)
                    last_checkpoint_time = time.time()

                if step % args.eval_every_n_steps == 0 and step > start_step:
                    val_loss = compute_val_ctc_loss_gated(model, val_loader, blank_id, device)
                    log_line(
                        f"[VAL] Step {step}: mean_ctc_loss_per_gloss_token={val_loss:.4f} "
                        f"(best so far: {best_val_loss:.4f}"
                        + (f" @ step {best_val_step}" if best_val_step is not None else "")
                        + ")"
                    )
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_val_step = step
                        patience_counter = 0
                        save_checkpoint(model, optimizer, step, BEST_VAL_CHECKPOINT_PATH)
                        save_best_val_meta(step, val_loss, patience_counter)
                        log_line(f"[VAL] NEW BEST -- saved {BEST_VAL_CHECKPOINT_PATH} "
                                  f"(step {step}, val_loss={val_loss:.4f}); patience reset to 0")
                    else:
                        patience_counter += 1
                        save_best_val_meta(best_val_step, best_val_loss, patience_counter)
                        patience_display = str(args.early_stop_patience) if args.early_stop_patience > 0 else "disabled"
                        log_line(f"[VAL] No improvement (patience {patience_counter}/{patience_display})")

                        if 0 < args.early_stop_patience <= patience_counter:
                            log_line(
                                f"[EARLY STOP] {patience_counter} consecutive evals "
                                f"({patience_counter * args.eval_every_n_steps} steps) with no held-out "
                                f"improvement over step {best_val_step} (val_loss={best_val_loss:.4f}). "
                                f"Stopping training. Use {BEST_VAL_CHECKPOINT_PATH} going forward."
                            )
                            ckpt_path = CHECKPOINT_DIR / f"step_{step:07d}.pt"
                            save_checkpoint(model, optimizer, step, ckpt_path)
                            log_line(f"[CHECKPOINT] Final save before early stop: {ckpt_path}")
                            cleanup_old_checkpoints(CHECKPOINT_DIR)
                            raise StopIteration

                step += 1
                session_steps += 1

    except StopIteration:
        pass
    except KeyboardInterrupt:
        log_line(f"[INTERRUPTED] Stopped at step {step} by user/session limit")
    finally:
        log_line(f"[SESSION TERMINATED] Stopped at step {step}. "
                  f"Best-on-val checkpoint remains step {best_val_step} (val_loss={best_val_loss:.4f}).")
        elapsed = time.time() - session_start_time
        log_line(f"=== Session end: step={step}, session_steps={session_steps}, "
                  f"elapsed={elapsed/60:.1f} min, steps/min={session_steps/(elapsed/60) if elapsed > 0 else 0:.1f} ===")
        if ctc_losses:
            log_line(f"This session: ctc_loss last={ctc_losses[-1]:.4f}, "
                      f"boundary_loss last={boundary_losses[-1]:.4f}, "
                      f"gate_entropy last={gate_entropies[-1]:.4f}")


if __name__ == "__main__":
    main()
