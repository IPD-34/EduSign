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

--- SESSION-SAFETY PATCH (post two lost-session incident) ---
Root cause of the two lost sessions: checkpointing was step-count-only
(CHECKPOINT_EVERY_N_STEPS), with no time-based floor and no proactive
stop before Kaggle's own session/commit limit. Three additions:
  1. CHECKPOINT_EVERY_N_SECONDS -- time-based checkpoint floor.
  2. EFFECTIVE_TRAIN_LIMIT_SECONDS -- hard, proactive wall-clock stop,
     so the script always finishes executing ON ITS OWN with margin.
  3. A SIGTERM handler -- last-resort save if Kaggle kills it anyway.
save_checkpoint() itself is unchanged (already atomic).

--- EARLY-STOPPING PATCH (post overfitting confirmation, part 2) ---
Tracking the best-val checkpoint (below) tells you WHICH checkpoint to
use, but doesn't stop the run from spending further GPU-hours making
MORE overfit checkpoints past the point generalization already peaked
(exactly what happened between step 300 and step 428,000: 427,700
further steps, zero of them an improvement on val).

Fix: a standard patience counter. Every EVAL_EVERY_N_STEPS check that
does NOT beat best_val_loss increments the counter; any check that
DOES beat it resets the counter to 0. Once the counter reaches
--early-stop-patience consecutive non-improving evals, training stops
itself -- same clean-exit path as the time-budget stop (save, log,
raise StopIteration), not a hard kill.

The counter is persisted in best_val_meta.json, not just held in a
local variable -- otherwise every Kaggle --resume across a multi-
session chain would silently reset it to 0, and with EVAL_EVERY_N_STEPS
this coarse relative to a single session's step count, early stopping
might never actually get a chance to fire across session boundaries.

Note: worse, later checkpoints don't need a separate deletion step --
cleanup_old_checkpoints() already prunes step_XXXXXXX.pt files down to
the last 3, and best_val.pt is permanently protected by filename. The
real fix is stopping the run from producing more of them, not deleting
ones that already exist.

--- BEST-VAL-CHECKPOINT PATCH (post overfitting confirmation) ---
eval_holdout.py confirmed a real, monotonic held-out generalization
regression: mean_ctc_loss_per_gloss_token on the 122-utterance val
split went 5.89 (step 300) -> 11.69 (step 58,000) -> ~17.1 (step
428,000). Training loss alone was never going to catch this -- ASLLRP
train is only 990 utterances, so step 428,000 is ~1,730 epochs over
the same examples, and CTC/boundary training loss had already
collapsed to near-zero (memorization), long past the point where it
still reflected generalization.

Fix: periodically evaluate on the real held-out val split DURING
training (not reconstructed after the fact), and separately persist
whichever checkpoint is best-on-val so far, distinct from the normal
step-numbered rolling checkpoints (which track recency, not quality,
and get pruned by cleanup_old_checkpoints()).

Design choices, stated explicitly:
  - best_val.pt is saved via the existing, unmodified save_checkpoint()
    -- so it is a fully normal {step, model_state_dict,
    optimizer_state_dict} checkpoint, loadable by every existing tool
    (get_latest_checkpoint()'s validation logic, eval_holdout.py,
    load_checkpoint() for a real resume) with zero special-casing.
  - The val_loss value itself is NOT embedded in the checkpoint payload
    (train_utils.py's save_checkpoint() was deliberately left
    unmodified). Instead it's tracked in a small JSON sidecar file,
    best_val_meta.json, next to the checkpoint. This keeps
    train_utils.py's checkpoint format identical everywhere in the
    project, rather than growing a second, checkpoint-specific format.
  - best_val.pt is filename-distinct from step_XXXXXXX.pt, so it is
    automatically invisible to cleanup_old_checkpoints() (which only
    matches the _STEP_RE pattern) -- no change needed there either.
  - The val metric matches eval_holdout.py's definition exactly (mean
    CTC loss per gloss token, BOS/EOS stripped, zero_infinity=True),
    so numbers logged here are directly comparable to what
    eval_holdout.py already reported for earlier checkpoints.
  - Eval runs in model.eval() / no_grad(), then explicitly restores
    whatever training mode the model was in beforehand -- so it never
    silently leaves dropout disabled for subsequent training steps.

Usage:
    First launch:
        uv run python -m scripts.train_month2_asllrp
    Resume (auto-detects the latest checkpoint under CHECKPOINT_DIR):
        uv run python -m scripts.train_month2_asllrp --resume
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
from src.models.combined_loss import compute_combined_loss

VOCAB_MERGED_PATH = "data/vocab_merged.json"
DECODER_VOCAB_PATH = "data/vocab.json"  # 1850 entries -- old WLASL decoder vocab
ASLLRP_TOKEN_TO_ID_PATH = "data/asllrp_token_to_id.json"
OLD_WLASL_CHECKPOINT = "checkpoints/day12_latest.pt"
ASLLRP_NPZ_DIR = "data/processed/asllrp/train"
ASLLRP_VAL_NPZ_DIR = "data/processed/asllrp/val"  # --- BEST-VAL PATCH ---

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

# --- SESSION-SAFETY PATCH: constants ---
CHECKPOINT_EVERY_N_SECONDS = 15 * 60  # 15 minutes, independent time-based floor
TIME_LIMIT_SECONDS = 5 * 60 * 60       # 5-hour cap, per the restart decision
SAFETY_BUFFER_SECONDS = 15 * 60        # 15 minutes reserved
EFFECTIVE_TRAIN_LIMIT_SECONDS = TIME_LIMIT_SECONDS - SAFETY_BUFFER_SECONDS

# --- BEST-VAL PATCH: constants ---
EVAL_EVERY_N_STEPS = 2000  # same cadence as checkpointing, easy mental model
BEST_VAL_CHECKPOINT_PATH = CHECKPOINT_DIR / "best_val.pt"
BEST_VAL_META_PATH = CHECKPOINT_DIR / "best_val_meta.json"

# --- EARLY-STOPPING PATCH: default, overridable via --early-stop-patience ---
# 5 consecutive non-improving evals * EVAL_EVERY_N_STEPS(2000) = 10,000 steps
# of no held-out improvement before auto-stopping. At ~247 steps/epoch on
# ASLLRP train, that's ~40 epochs of no improvement -- enough buffer to not
# false-trigger on normal eval-to-eval noise from a 122-utterance val set,
# short enough to not repeat the 427,700-step waste that motivated this.
DEFAULT_EARLY_STOP_PATIENCE = 5


# ONLY matches normal checkpoints:
# step_0058000.pt
#
# Does NOT match:
# step_0058200_final.pt
# step_0058200.pt.tmp
# best_val.pt          <-- deliberately excluded, see BEST-VAL PATCH note above
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
    """Keep only the newest N normal checkpoints.

    best_val.pt is untouched by this function: it never matches
    _STEP_RE, so it's simply skipped over below, same as any other
    non-matching file (e.g. best_val_meta.json)."""
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


# --- BEST-VAL PATCH: metric matches eval_holdout.py exactly ---
@torch.no_grad()
def compute_val_ctc_loss(model, val_loader, blank_id, device):
    """
    Mean CTC loss per gloss token on held-out val data -- same metric,
    same BOS/EOS-stripping convention, same zero_infinity=True setting
    as eval_holdout.py, so numbers logged here are directly comparable
    to what that script already reported for earlier checkpoints.

    Explicitly restores whatever training mode (train/eval) the model
    was in before this call, so it never silently leaves dropout
    disabled for the training steps that follow.
    """
    was_training = model.training
    model.eval()

    ctc_loss_fn = nn.CTCLoss(blank=blank_id, zero_infinity=True, reduction="sum")
    total_loss, total_targets = 0.0, 0

    for batch in val_loader:
        landmarks = batch["landmarks"].to(device)
        enc_mask = batch["encoder_attention_mask"].to(device)

        log_probs = model.forward_ctc(landmarks, enc_mask)
        log_probs_ctc = log_probs.transpose(0, 1)
        input_lengths = enc_mask.sum(dim=1).cpu()

        target_tokens = batch["target_tokens"]
        target_mask = batch["target_padding_mask"]

        targets, target_lengths = [], []
        for i in range(target_tokens.shape[0]):
            real = target_tokens[i][target_mask[i]]
            gloss_only = real[1:-1]  # strip BOS, EOS
            targets.append(gloss_only)
            target_lengths.append(len(gloss_only))

        targets_flat = torch.cat(targets)
        target_lengths_t = torch.tensor(target_lengths, dtype=torch.long)

        loss = ctc_loss_fn(
            log_probs_ctc.cpu(), targets_flat, input_lengths, target_lengths_t
        )
        if torch.isfinite(loss):
            total_loss += loss.item()
            total_targets += sum(target_lengths)

    if was_training:
        model.train()

    return total_loss / max(total_targets, 1)


def load_best_val_meta():
    """Returns (best_val_loss, best_val_step, patience_counter), defaulting
    to (inf, None, 0) if no prior best_val_meta.json exists -- e.g. on a
    true first launch, or the first run since this patch was added.
    patience_counter defaults to 0 even for an OLDER meta file that
    predates the early-stopping patch (missing key) -- backward compatible,
    doesn't require regenerating existing meta files."""
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
            log_line(f"[VAL] Could not read {BEST_VAL_META_PATH}: "
                      f"{type(e).__name__}: {e}. Starting best-val tracking fresh.")
    return float("inf"), None, 0


def save_best_val_meta(step, val_loss, patience_counter):
    """Small, atomic-enough sidecar write (temp + rename) -- deliberately
    NOT extending save_checkpoint()'s payload format in train_utils.py,
    per this project's preference to leave that file unmodified."""
    BEST_VAL_META_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = BEST_VAL_META_PATH.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(
            {"step": step, "val_loss": val_loss, "patience_counter": patience_counter},
            f,
        )
    tmp.replace(BEST_VAL_META_PATH)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true",
                         help="Resume from the latest checkpoint under CHECKPOINT_DIR")
    parser.add_argument("--max-steps-this-session", type=int, default=None,
                         help="Stop after N steps this session (for the explicit resume test)")
    parser.add_argument("--early-stop-patience", type=int, default=DEFAULT_EARLY_STOP_PATIENCE,
                         help="Stop training after this many consecutive non-improving "
                              "held-out val evals (0 = disabled). Persists across --resume "
                              f"via best_val_meta.json. Default: {DEFAULT_EARLY_STOP_PATIENCE}.")
    parser.add_argument("--eval-every-n-steps", type=int, default=EVAL_EVERY_N_STEPS,
                         help=f"Override the held-out eval cadence (default {EVAL_EVERY_N_STEPS}). "
                              "Useful for a short local test to actually cross an eval boundary "
                              "without waiting for real training-scale step counts.")
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
        pin_memory=True,
    )

    # --- BEST-VAL PATCH: val loader, built once, reused every eval ---
    val_dataset = SignLanguageDataset(
        npz_dir=ASLLRP_VAL_NPZ_DIR,
        vocab_path=VOCAB_MERGED_PATH,
        asllrp_token_to_id_path=ASLLRP_TOKEN_TO_ID_PATH,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn,
        shuffle=False,  # no need to shuffle for eval
        pin_memory=True, 
    )
    log_line(f"[VAL] Loaded val set: {len(val_dataset)} utterances, "
              f"eval every {args.eval_every_n_steps} steps")

    best_val_loss, best_val_step, patience_counter = load_best_val_meta()
    if best_val_step is not None:
        log_line(f"[VAL] Resuming best-val tracking: current best is step "
                  f"{best_val_step} (val_loss={best_val_loss:.4f}), "
                  f"patience_counter={patience_counter}/{args.early_stop_patience} -- "
                  f"a new checkpoint must beat this to become the new best "
                  f"and reset patience.")
    else:
        log_line(f"[VAL] No prior best_val_meta.json -- starting best-val "
                  f"tracking fresh at val_loss=inf, patience_counter=0.")
    if args.early_stop_patience <= 0:
        log_line("[VAL] Early stopping DISABLED (--early-stop-patience <= 0).")
    # --- end best-val setup ---

    log_line(f"=== Training session start: begin_step={start_step}, "
              f"resume={args.resume}, device={device} ===")
    log_line(
        f"[TIME BUDGET] hard stop at {EFFECTIVE_TRAIN_LIMIT_SECONDS/60:.0f} min "
        f"(of {TIME_LIMIT_SECONDS/60:.0f} min session cap, "
        f"{SAFETY_BUFFER_SECONDS/60:.0f} min buffer reserved); "
        f"time-based checkpoint floor every {CHECKPOINT_EVERY_N_SECONDS/60:.0f} min"
    )

    step = start_step
    session_start_time = time.time()
    session_steps = 0
    ctc_losses, boundary_losses = [], []

    last_checkpoint_time = session_start_time

    _step_box = {"value": step}

    def _handle_sigterm(signum, frame):
        current_step = _step_box["value"]
        log_line(f"[SIGTERM] Received at step {current_step}. "
                  f"Attempting emergency checkpoint save.")
        try:
            ckpt_path = CHECKPOINT_DIR / f"step_{current_step:07d}.pt"
            save_checkpoint(model, optimizer, current_step, ckpt_path)
            log_line(f"[SIGTERM] Emergency checkpoint saved: {ckpt_path}")
        except Exception as e:
            log_line(f"[SIGTERM] Emergency save FAILED: {type(e).__name__}: {e}. "
                      f"Falling back to whatever the last periodic/time-based "
                      f"checkpoint already on disk was.")
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
                    log_line(
                        f"[TIME BUDGET] {elapsed/60:.1f} min elapsed, reached "
                        f"effective limit ({EFFECTIVE_TRAIN_LIMIT_SECONDS/60:.0f} min). "
                        f"Saving final checkpoint and exiting cleanly at step {step}."
                    )
                    ckpt_path = CHECKPOINT_DIR / f"step_{step:07d}.pt"
                    save_checkpoint(model, optimizer, step, ckpt_path)
                    log_line(f"[CHECKPOINT] Time-budget final save: {ckpt_path}")
                    cleanup_old_checkpoints(CHECKPOINT_DIR)
                    last_checkpoint_time = time.time()
                    raise StopIteration

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
                    steps_per_min = (session_steps + 1) / (elapsed / 60) if elapsed > 0 else 0.0
                    log_line(
                        f"Step {step}: ctc_loss={ctc_loss.item():.4f} "
                        f"boundary_loss={boundary_loss.item():.4f} "
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

                # --- BEST-VAL + EARLY-STOPPING PATCH: periodic held-out eval ---
                if step % args.eval_every_n_steps == 0 and step > start_step:
                    val_loss = compute_val_ctc_loss(model, val_loader, blank_id, device)
                    log_line(
                        f"[VAL] Step {step}: mean_ctc_loss_per_gloss_token={val_loss:.4f} "
                        f"(best so far: {best_val_loss:.4f}"
                        + (f" @ step {best_val_step}" if best_val_step is not None else "")
                        + ")"
                    )
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_val_step = step
                        patience_counter = 0  # reset -- this is a real improvement
                        save_checkpoint(model, optimizer, step, BEST_VAL_CHECKPOINT_PATH)
                        save_best_val_meta(step, val_loss, patience_counter)
                        log_line(
                            f"[VAL] NEW BEST -- saved {BEST_VAL_CHECKPOINT_PATH} "
                            f"(step {step}, val_loss={val_loss:.4f}); patience reset to 0"
                        )
                    else:
                        patience_counter += 1
                        save_best_val_meta(best_val_step, best_val_loss, patience_counter)
                        patience_display = str(args.early_stop_patience) if args.early_stop_patience > 0 else "disabled"
                        log_line(
                            f"[VAL] No improvement (patience "
                            f"{patience_counter}/{patience_display})"
                        )

                        if 0 < args.early_stop_patience <= patience_counter:
                            log_line(
                                f"[EARLY STOP] {patience_counter} consecutive evals "
                                f"({patience_counter * args.eval_every_n_steps} steps) with no "
                                f"held-out improvement over step {best_val_step} "
                                f"(val_loss={best_val_loss:.4f}). Stopping training. "
                                f"Use checkpoints/month2_asllrp/best_val.pt going forward, "
                                f"not the latest step-numbered checkpoint."
                            )
                            ckpt_path = CHECKPOINT_DIR / f"step_{step:07d}.pt"
                            save_checkpoint(model, optimizer, step, ckpt_path)
                            log_line(f"[CHECKPOINT] Final save before early stop: {ckpt_path}")
                            cleanup_old_checkpoints(CHECKPOINT_DIR)
                            raise StopIteration
                # --- end best-val + early-stopping patch ---

                step += 1
                session_steps += 1

    except StopIteration:
        pass
    except KeyboardInterrupt:
        log_line(f"[INTERRUPTED] Stopped at step {step} by user/session limit")
    finally:
        log_line(
            f"[SESSION TERMINATED] Stopped at step {step}. "
            f"Last checkpoint should be at or within "
            f"{CHECKPOINT_EVERY_N_SECONDS/60:.0f} min of this point. "
            f"Best-on-val checkpoint remains step {best_val_step} "
            f"(val_loss={best_val_loss:.4f})."
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