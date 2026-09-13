"""
scripts/stageB1_real_training_run.py

The first REAL (not smoke-scale) training commitment of the improvement
campaign. Tests whether Stage B1's spatial augmentation, given a REAL
training horizon (natural early-stopping, not 500 steps), narrows the
train/held-out gap that the 500-step smoke tests couldn't resolve
(inconclusive: direction flipped across measurement points, noise-
dominated at that scale).

ONE variable only: --augment on/off. Everything else (architecture,
boundary_loss_weight=1.0, blank_logit_bias=0.0 -- both of Stage A's
ruled-out levers left at plain defaults, gate active, learning rate,
batch size, early-stopping patience/cadence) is held fixed and IDENTICAL
between the two runs this script produces, so any difference found is
attributable to augmentation alone -- same discipline as every stage of
this campaign.

--- INFRASTRUCTURE: mirrors train_month2_asllrp_gated.py exactly ---
Same best-val-by-loss tracking + patience-based early stopping
(EVAL_EVERY_N_STEPS=2000, DEFAULT_EARLY_STOP_PATIENCE=5 -- unchanged
from every other real run this project has done), same checkpoint
cadence/cleanup, same SIGTERM emergency-save handling. Isolated per run
name (--run-name control|augmented) under checkpoints/stageB1_real/
<run-name>/, exactly the same "separate directory so two runs' step-
numbered checkpoints can never collide" reasoning
train_month2_asllrp_gated.py already documented for its own isolation
from the ungated run.

Both runs resume MODEL WEIGHTS ONLY from checkpoints/month2_asllrp_gated/
step_0008000.pt (load_checkpoint_partial, fresh optimizer) -- this is a
first launch of two NEW branches from that point, not a --resume of a
prior session's optimizer trajectory, same convention every Stage A/B1
experiment has used.

--- NEW: WER/boundary-F1 PROBE ALONGSIDE VAL-CTC-LOSS TRACKING ---
Day 52 already found val CTC loss doesn't reliably track real decode
accuracy (step_0008000.pt itself has WORSE val loss than best_val_gated.pt
but MEANINGFULLY BETTER real WER). Selecting this run's "best" checkpoint
by loss alone would risk repeating exactly that mistake. So every
WER_PROBE_EVERY_N_STEPS, this script ALSO runs the real, hand-validated
Day 50/51 machinery (ctc_prefix_beam_search + edit_distance for WER;
extract_intervals-based IoU matching for boundary-F1 -- both imported,
not reimplemented) on a FIXED 30-sample val subset (cheaper than the
full 122-sample split, since this runs many times during training, but
still real beam-search decoding, not a loss proxy) and tracks its own,
SEPARATE best-by-WER checkpoint (best_by_wer.pt) alongside the
best-by-loss one. Which of the two actually gets used for the final
full-set evaluation is decided AFTER training, by comparing which
metric's "best" checkpoint reports better real WER on that probe --
not assumed in advance.

--- TIME BUDGET, STATED EXPLICITLY ---
Hard backstop of 40 minutes wall-clock per run (TIME_LIMIT_SECONDS
below) -- NOT the primary stopping mechanism (that's natural early-
stopping via patience, as requested), just a safety ceiling in case
early-stopping doesn't trigger promptly. Chosen from this project's own
observed throughput (the original gated run did step 300->14000, a full
early-stop cycle, in 19.4 minutes at ~707 steps/min) -- 40 minutes at
similar throughput covers ~28,000 steps, several times the ~10,000-step
minimum a patience-5-at-2000-step-intervals early stop requires, so it
should not bind unless something is behaving very differently than
every prior real run here has.
"""

import argparse
import json
import re
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader

from src.models.sign_transformer import SignTransformer
from src.models.train_utils import (
    load_checkpoint_partial, load_checkpoint, save_checkpoint, get_device,
)
from src.models.dataset import SignLanguageDataset, collate_fn
from src.models.combined_loss import compute_combined_loss_with_gate
from scripts.compute_wer_beam_search import ctc_prefix_beam_search, edit_distance
from scripts.compute_boundary_f1 import (
    compute_sample_boundary_arrays, boundary_f1_for_sample,
)

VOCAB_MERGED_PATH = "data/vocab_merged.json"
DECODER_VOCAB_PATH = "data/vocab.json"
ASLLRP_TOKEN_TO_ID_PATH = "data/asllrp_token_to_id.json"
RESUME_CHECKPOINT = "checkpoints/month2_asllrp_gated/step_0008000.pt"  # current checkpoint of record
TRAIN_NPZ_DIR = "data/processed/asllrp/train"
VAL_NPZ_DIR = "data/processed/asllrp/val"

BATCH_SIZE = 4
LEARNING_RATE = 1e-4
BOUNDARY_LOSS_WEIGHT = 1.0   # Stage A's ruled-out lever, plain default -- isolation
GATE_REG_WEIGHT = 0.0
BLANK_LOGIT_BIAS = 0.0       # Stage A's ruled-out lever, plain default -- isolation
GRAD_CLIP_MAX_NORM = 1.0
CHECKPOINT_EVERY_N_STEPS = 2000
LOG_EVERY_N_STEPS = 25
KEEP_LAST_N_CHECKPOINTS = 3

EVAL_EVERY_N_STEPS = 2000            # unchanged from every other real run
DEFAULT_EARLY_STOP_PATIENCE = 5      # unchanged from every other real run

WER_PROBE_EVERY_N_STEPS = 4000
WER_PROBE_SEED = 157
WER_PROBE_SIZE = 30
BEAM_WIDTH = 10
EXPAND_TOP_K = 20
IOU_THRESHOLD = 0.5
BOUNDARY_PROB_THRESHOLD = 0.5

CHECKPOINT_EVERY_N_SECONDS = 10 * 60
TIME_LIMIT_SECONDS = 40 * 60  # hard backstop, NOT the primary stop -- see module docstring
SAFETY_BUFFER_SECONDS = 3 * 60
EFFECTIVE_TRAIN_LIMIT_SECONDS = TIME_LIMIT_SECONDS - SAFETY_BUFFER_SECONDS

_STEP_RE = re.compile(r"^step_(\d+)\.pt$")


def get_latest_checkpoint(checkpoint_dir):
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
        except Exception:
            continue
        if not required_keys.issubset(checkpoint.keys()) or checkpoint["step"] != step_num:
            continue
        return path
    return None


def cleanup_old_checkpoints(checkpoint_dir, log_line, keep_last_n=KEEP_LAST_N_CHECKPOINTS):
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


@torch.no_grad()
def compute_val_ctc_loss(model, val_loader, blank_id, device):
    import torch.nn as nn
    was_training = model.training
    model.eval()
    ctc_loss_fn = nn.CTCLoss(blank=blank_id, zero_infinity=True, reduction="sum")
    total_loss, total_targets = 0.0, 0
    for batch in val_loader:
        landmarks = batch["landmarks"].to(device)
        enc_mask = batch["encoder_attention_mask"].to(device)
        ctc_log_probs, _bl, _gl = model.forward_ctc_boundary_and_gate(landmarks, enc_mask)
        log_probs_ctc = ctc_log_probs.transpose(0, 1)
        input_lengths = enc_mask.sum(dim=1).cpu()
        target_tokens = batch["target_tokens"]
        target_mask = batch["target_padding_mask"]
        targets, target_lengths = [], []
        for i in range(target_tokens.shape[0]):
            real = target_tokens[i][target_mask[i]]
            targets.append(real[1:-1])
            target_lengths.append(len(real[1:-1]))
        targets_flat = torch.cat(targets)
        target_lengths_t = torch.tensor(target_lengths, dtype=torch.long)
        loss = ctc_loss_fn(log_probs_ctc.cpu(), targets_flat, input_lengths, target_lengths_t)
        if torch.isfinite(loss):
            total_loss += loss.item()
            total_targets += sum(target_lengths)
    if was_training:
        model.train()
    return total_loss / max(total_targets, 1)


@torch.no_grad()
def wer_and_boundary_f1_probe(model, probe_dataset, probe_indices, device, blank_id):
    """Real beam-search WER + IoU-matched boundary-F1 on a fixed subset --
    NOT a loss proxy. Reuses Day 50/51's hand-validated functions unchanged."""
    from src.preprocessing.sample_writer import load_sample
    was_training = model.training
    model.eval()

    total_edits, total_ref = 0, 0
    total_tp = total_fp = total_fn = 0
    for idx in probe_indices:
        item = probe_dataset[idx]
        meta = load_sample(probe_dataset.npz_paths[idx]).metadata
        landmarks = item["landmarks"].unsqueeze(0).to(device)
        T = landmarks.shape[1]
        mask = torch.ones(1, T, dtype=torch.bool, device=device)
        true_ids = item["target_tokens"].tolist()[1:-1]

        ctc_log_probs, _bl, _gl = model.forward_ctc_boundary_and_gate(landmarks, mask)
        log_probs = ctc_log_probs[0].cpu()
        decoded_ids = ctc_prefix_beam_search(log_probs, blank_id, BEAM_WIDTH, EXPAND_TOP_K)
        total_edits += edit_distance(true_ids, decoded_ids)
        total_ref += len(true_ids)

        gt_binary, pred_binary, _stats = compute_sample_boundary_arrays(
            model, item, meta, device, BOUNDARY_PROB_THRESHOLD
        )
        bf1 = boundary_f1_for_sample(gt_binary, pred_binary, IOU_THRESHOLD)
        total_tp += bf1["tp"]; total_fp += bf1["fp"]; total_fn += bf1["fn"]

    if was_training:
        model.train()
    corpus_wer = total_edits / max(total_ref, 1)
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"wer": corpus_wer, "boundary_precision": precision, "boundary_recall": recall, "boundary_f1": f1}


def load_best_meta(path):
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return None


def save_best_meta(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--augment", action="store_true", help="Enable Stage B1 spatial augmentation on the train split.")
    parser.add_argument("--run-name", required=True, choices=["control", "augmented"])
    parser.add_argument("--resume", action="store_true", help="Resume THIS run's own prior session (not first-launch from step_0008000.pt).")
    parser.add_argument("--early-stop-patience", type=int, default=DEFAULT_EARLY_STOP_PATIENCE)
    args = parser.parse_args()

    run_dir = Path("checkpoints/stageB1_real") / args.run_name
    log_path = Path("logs") / f"stageB1_real_{args.run_name}.log"
    best_val_ckpt_path = run_dir / "best_val.pt"
    best_val_meta_path = run_dir / "best_val_meta.json"
    best_wer_ckpt_path = run_dir / "best_by_wer.pt"
    best_wer_meta_path = run_dir / "best_by_wer_meta.json"

    def log_line(msg):
        print(msg)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    device = get_device()
    with open(VOCAB_MERGED_PATH) as f:
        vocab_merged = json.load(f)
    with open(DECODER_VOCAB_PATH) as f:
        decoder_vocab = json.load(f)
    ctc_vocab_size = len(vocab_merged)
    blank_id = ctc_vocab_size
    bos_id, eos_id, pad_id = vocab_merged["<BOS>"], vocab_merged["<EOS>"], vocab_merged["<PAD>"]

    model = SignTransformer(
        vocab_size=len(decoder_vocab), ctc_vocab_size=ctc_vocab_size, use_bimanual_gate=True,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    start_step = 0
    latest_ckpt = get_latest_checkpoint(run_dir)
    if args.resume:
        if latest_ckpt is None:
            raise RuntimeError(f"--resume passed but no checkpoint found under {run_dir}.")
        log_line(f"[RESUME] Loading full checkpoint: {latest_ckpt}")
        start_step = load_checkpoint(model, optimizer, latest_ckpt, device=device)
        log_line(f"[RESUME] Resumed at step {start_step}")
    else:
        if latest_ckpt is not None:
            raise RuntimeError(f"Checkpoint already exists at {latest_ckpt} but --resume not passed.")
        log_line(f"[FIRST LAUNCH] run_name={args.run_name} augment={args.augment} -- "
                 f"loading model weights ONLY from {RESUME_CHECKPOINT} (fresh optimizer, "
                 f"new experimental branch, not a literal continuation)")
        loaded, skipped, new_only = load_checkpoint_partial(
            model, RESUME_CHECKPOINT,
            prefixes_to_load=("encoder.", "ctc_head.", "boundary_head.", "bimanual_gate_head."),
        )
        log_line(f"[FIRST LAUNCH] Transfer: {len(loaded)} loaded, {len(skipped)} skipped, {len(new_only)} new-only")

    train_dataset = SignLanguageDataset(
        npz_dir=TRAIN_NPZ_DIR, vocab_path=VOCAB_MERGED_PATH, asllrp_token_to_id_path=ASLLRP_TOKEN_TO_ID_PATH,
        augment=args.augment,
    )
    loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn,
        shuffle=True, num_workers=0,
    )
    val_dataset = SignLanguageDataset(
        npz_dir=VAL_NPZ_DIR, vocab_path=VOCAB_MERGED_PATH, asllrp_token_to_id_path=ASLLRP_TOKEN_TO_ID_PATH,
        augment=False,  # val is NEVER augmented, either config
    )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=False)
    log_line(f"[VAL] Loaded val set: {len(val_dataset)} utterances (train augment={args.augment}, val augment=False always)")

    import random
    wer_probe_dataset = SignLanguageDataset(
        npz_dir=VAL_NPZ_DIR, vocab_path=VOCAB_MERGED_PATH, asllrp_token_to_id_path=ASLLRP_TOKEN_TO_ID_PATH,
        augment=False,
    )
    rng = random.Random(WER_PROBE_SEED)
    wer_probe_indices = sorted(rng.sample(range(len(wer_probe_dataset)), min(WER_PROBE_SIZE, len(wer_probe_dataset))))
    log_line(f"[WER PROBE] {len(wer_probe_indices)} fixed val samples (seed={WER_PROBE_SEED}), every {WER_PROBE_EVERY_N_STEPS} steps")

    best_val = load_best_meta(best_val_meta_path)
    best_val_loss = best_val["val_loss"] if best_val else float("inf")
    best_val_step = best_val["step"] if best_val else None
    patience_counter = best_val["patience_counter"] if best_val else 0

    best_wer = load_best_meta(best_wer_meta_path)
    best_wer_value = best_wer["wer"] if best_wer else float("inf")
    best_wer_step = best_wer["step"] if best_wer else None

    log_line(f"=== Stage B1 REAL run start: run_name={args.run_name}, augment={args.augment}, "
              f"begin_step={start_step}, device={device}, boundary_loss_weight={BOUNDARY_LOSS_WEIGHT}, "
              f"blank_logit_bias={BLANK_LOGIT_BIAS} ===")
    log_line(f"[TIME BUDGET] hard backstop {EFFECTIVE_TRAIN_LIMIT_SECONDS/60:.0f} min "
              f"(NOT the primary stop mechanism -- see module docstring)")

    step = start_step
    session_start_time = time.time()
    session_steps = 0
    ctc_losses, boundary_losses = [], []
    last_checkpoint_time = session_start_time

    _step_box = {"value": step}

    def _handle_sigterm(signum, frame):
        current_step = _step_box["value"]
        log_line(f"[SIGTERM] Received at step {current_step}. Emergency checkpoint save.")
        try:
            save_checkpoint(model, optimizer, current_step, run_dir / f"step_{current_step:07d}.pt")
        except Exception as e:
            log_line(f"[SIGTERM] Emergency save FAILED: {type(e).__name__}: {e}")
        sys.exit(0)

    signal.signal(signal.SIGTERM, _handle_sigterm)

    model.train()
    try:
        while True:
            for batch in loader:
                _step_box["value"] = step
                elapsed = time.time() - session_start_time

                if elapsed >= EFFECTIVE_TRAIN_LIMIT_SECONDS:
                    log_line(f"[TIME BUDGET] {elapsed/60:.1f} min elapsed, hit backstop. Saving and exiting.")
                    save_checkpoint(model, optimizer, step, run_dir / f"step_{step:07d}.pt")
                    cleanup_old_checkpoints(run_dir, log_line)
                    raise StopIteration

                batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                ctc_log_probs, boundary_logits, gate_logits = model.forward_ctc_boundary_and_gate(
                    batch["landmarks"], batch["encoder_attention_mask"]
                )
                total_loss, ctc_loss, boundary_loss, _ge = compute_combined_loss_with_gate(
                    ctc_log_probs, boundary_logits, gate_logits,
                    batch["target_tokens"], batch["boundary_target"], batch["encoder_attention_mask"],
                    bos_id, eos_id, pad_id, blank_id,
                    boundary_loss_weight=BOUNDARY_LOSS_WEIGHT, gate_reg_weight=GATE_REG_WEIGHT,
                    blank_logit_bias=BLANK_LOGIT_BIAS,
                )

                if not torch.isfinite(total_loss):
                    log_line(f"[WARNING] Step {step}: non-finite loss, skipping batch")
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
                    log_line(f"Step {step}: ctc_loss={ctc_loss.item():.4f} boundary_loss={boundary_loss.item():.4f} "
                              f"grad_norm={grad_norm:.4f} steps/min={steps_per_min:.1f}")

                if step % CHECKPOINT_EVERY_N_STEPS == 0 and step > start_step:
                    save_checkpoint(model, optimizer, step, run_dir / f"step_{step:07d}.pt")
                    log_line(f"[CHECKPOINT] Saved (step-count): step_{step:07d}.pt")
                    cleanup_old_checkpoints(run_dir, log_line)
                    last_checkpoint_time = time.time()
                elif time.time() - last_checkpoint_time >= CHECKPOINT_EVERY_N_SECONDS:
                    save_checkpoint(model, optimizer, step, run_dir / f"step_{step:07d}.pt")
                    log_line(f"[CHECKPOINT] Saved (time-floor): step_{step:07d}.pt")
                    cleanup_old_checkpoints(run_dir, log_line)
                    last_checkpoint_time = time.time()

                if step % EVAL_EVERY_N_STEPS == 0 and step > start_step:
                    val_loss = compute_val_ctc_loss(model, val_loader, blank_id, device)
                    log_line(f"[VAL] Step {step}: mean_ctc_loss_per_gloss_token={val_loss:.4f} "
                              f"(best so far: {best_val_loss:.4f}" +
                              (f" @ step {best_val_step}" if best_val_step is not None else "") + ")")
                    if val_loss < best_val_loss:
                        best_val_loss, best_val_step, patience_counter = val_loss, step, 0
                        save_checkpoint(model, optimizer, step, best_val_ckpt_path)
                        save_best_meta(best_val_meta_path, {"step": step, "val_loss": val_loss, "patience_counter": 0})
                        log_line(f"[VAL] NEW BEST-BY-LOSS -- saved {best_val_ckpt_path} (step {step}, val_loss={val_loss:.4f})")
                    else:
                        patience_counter += 1
                        save_best_meta(best_val_meta_path, {"step": best_val_step, "val_loss": best_val_loss, "patience_counter": patience_counter})
                        log_line(f"[VAL] No improvement (patience {patience_counter}/{args.early_stop_patience})")
                        if 0 < args.early_stop_patience <= patience_counter:
                            log_line(f"[EARLY STOP] {patience_counter} consecutive evals with no improvement "
                                      f"over step {best_val_step} (val_loss={best_val_loss:.4f}). Stopping.")
                            save_checkpoint(model, optimizer, step, run_dir / f"step_{step:07d}.pt")
                            cleanup_old_checkpoints(run_dir, log_line)
                            raise StopIteration

                if step % WER_PROBE_EVERY_N_STEPS == 0 and step > start_step:
                    probe = wer_and_boundary_f1_probe(model, wer_probe_dataset, wer_probe_indices, device, blank_id)
                    log_line(f"[WER PROBE] Step {step}: wer={probe['wer']:.4f} "
                              f"boundary_f1={probe['boundary_f1']:.4f} "
                              f"(P={probe['boundary_precision']:.4f} R={probe['boundary_recall']:.4f}) "
                              f"(best wer so far: {best_wer_value:.4f}" +
                              (f" @ step {best_wer_step}" if best_wer_step is not None else "") + ")")
                    if probe["wer"] < best_wer_value:
                        best_wer_value, best_wer_step = probe["wer"], step
                        save_checkpoint(model, optimizer, step, best_wer_ckpt_path)
                        save_best_meta(best_wer_meta_path, {"step": step, **probe})
                        log_line(f"[WER PROBE] NEW BEST-BY-WER -- saved {best_wer_ckpt_path} (step {step}, wer={probe['wer']:.4f})")

                step += 1
                session_steps += 1

    except StopIteration:
        pass
    except KeyboardInterrupt:
        log_line(f"[INTERRUPTED] Stopped at step {step}")
    finally:
        log_line(f"[SESSION TERMINATED] Stopped at step {step}. "
                  f"best_val: step={best_val_step} loss={best_val_loss:.4f}. "
                  f"best_by_wer: step={best_wer_step} wer={best_wer_value:.4f}")
        elapsed = time.time() - session_start_time
        log_line(f"=== Session end: step={step}, session_steps={session_steps}, "
                  f"elapsed={elapsed/60:.2f} min, steps/min={session_steps/(elapsed/60) if elapsed > 0 else 0:.1f} ===")


if __name__ == "__main__":
    main()
