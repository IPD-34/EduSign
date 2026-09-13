"""
scripts/stageC_transition_loss_real_run.py

Stage C: the first REAL (not smoke-scale) training commitment for the
transition-loss mechanism (src/preprocessing/boundary_labels.py's
build_transition_targets() + combined_loss.py's
compute_combined_loss_with_gate_and_transition()) -- Stage C's own
1000-step smoke test found real, consistent, mechanism-specific
improvement (both boundary-F1 criteria up, predicted blob count moving
toward the true count, especially on dense multi-sign utterances) and
earned a go recommendation. Regularization and oversampling were both
ruled out at smoke scale and are NOT part of this run -- ONE variable
only (transition_loss_weight=1.0, sigma=1.0, EXACTLY the smoke test's
config, unchanged), same isolation discipline as every stage of this
campaign.

--- INFRASTRUCTURE: mirrors stageB1_real_training_run.py exactly ---
Same resume point (checkpoints/month2_asllrp_gated/step_0008000.pt,
model-weights-only via load_checkpoint_partial, fresh optimizer) as
BOTH the control and augmented real runs -- so this run is directly,
cleanly comparable to both as a three-way comparison, not just to
control. Same best-val-by-loss tracking + patience-based early
stopping (EVAL_EVERY_N_STEPS=2000, DEFAULT_EARLY_STOP_PATIENCE=5), same
WER/boundary-F1 probe cadence and machinery, same SIGTERM emergency-
save handling, same 40-minute hard backstop (both prior real runs
early-stopped at step 12000 in 24.5-36.0 minutes at 278-490 steps/min,
so 40 minutes leaves comfortable margin at similar or slightly lower
throughput from the one extra cheap loss term).

--- DELIBERATE DEVIATION FROM stageB1_real_training_run.py: NO CHECKPOINT
CLEANUP ---
stageB1_real_training_run.py deletes all but the last
KEEP_LAST_N_CHECKPOINTS=3 step-numbered checkpoints as it goes. This
run's own task explicitly requires a FULL sweep across every saved
checkpoint at the end (per the project's now-standing lesson: every
cheap proxy tried so far -- val CTC loss, 30-sample WER probe, 55-
sample WER probe -- has mis-ranked checkpoints at least once, so no
single proxy is trusted to pick the final "best" checkpoint alone).
Deleting older checkpoints during the run would make that sweep
impossible for whatever got cleaned up. cleanup_old_checkpoints() is
therefore never called here -- every step-numbered checkpoint this run
saves stays on disk (at ~84MB each and a ~12000-step run producing
~6-7 of them per the prior runs' own early-stop point, this is a small,
deliberate, and bounded amount of extra disk, not an oversight).

--- WER/boundary-F1 PROBE ALONGSIDE VAL-CTC-LOSS TRACKING, EXTENDED ---
Same Day-52-motivated dual tracking as stageB1_real_training_run.py
(val CTC loss doesn't reliably track real decode accuracy), PLUS this
run's probe additionally reports the loose center-tolerance boundary-
F1 criterion and mean predicted-vs-true blob count (extract_intervals-
based), for DIRECTIONAL MONITORING ONLY during training -- explicitly
NOT used to select the final checkpoint (that's the full 122-sample
post-hoc sweep, done by a separate script after this run ends). Both
best_val.pt (by loss) and best_by_wer.pt (by the 30-sample WER probe)
are still tracked and saved as before, purely as convenient reference
points to inspect while the sweep script runs -- the task's own
"do not select the best checkpoint via any proxy alone" instruction
governs which checkpoint actually gets reported as best.
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
from src.models.combined_loss import compute_combined_loss_with_gate_and_transition
from scripts.compute_wer_beam_search import ctc_prefix_beam_search, edit_distance
from scripts.compute_boundary_f1 import (
    compute_sample_boundary_arrays, boundary_f1_for_sample, boundary_f1_for_sample_center_tolerance,
    extract_intervals, IOU_THRESHOLD, CENTER_TOLERANCE_TIMESTEPS,
)

VOCAB_MERGED_PATH = "data/vocab_merged.json"
DECODER_VOCAB_PATH = "data/vocab.json"
ASLLRP_TOKEN_TO_ID_PATH = "data/asllrp_token_to_id.json"
RESUME_CHECKPOINT = "checkpoints/month2_asllrp_gated/step_0008000.pt"  # same resume point as control/augmented
TRAIN_NPZ_DIR = "data/processed/asllrp/train"
VAL_NPZ_DIR = "data/processed/asllrp/val"

BATCH_SIZE = 4
LEARNING_RATE = 1e-4
BOUNDARY_LOSS_WEIGHT = 1.0   # held fixed, plain default -- isolation
GATE_REG_WEIGHT = 0.0
BLANK_LOGIT_BIAS = 0.0       # held fixed, plain default -- isolation
TRANSITION_LOSS_WEIGHT = 1.0  # Stage C smoke test's exact config, unchanged
GRAD_CLIP_MAX_NORM = 1.0
CHECKPOINT_EVERY_N_STEPS = 2000
LOG_EVERY_N_STEPS = 25

EVAL_EVERY_N_STEPS = 2000
DEFAULT_EARLY_STOP_PATIENCE = 5

WER_PROBE_EVERY_N_STEPS = 4000
WER_PROBE_SEED = 157
WER_PROBE_SIZE = 30
BEAM_WIDTH = 10
EXPAND_TOP_K = 20
IOU_THRESH = IOU_THRESHOLD
BOUNDARY_PROB_THRESHOLD = 0.5

CHECKPOINT_EVERY_N_SECONDS = 10 * 60
TIME_LIMIT_SECONDS = 40 * 60  # same backstop as control/augmented -- both stopped at step 12000 well inside this
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
    """Real beam-search WER + BOTH boundary-F1 criteria + blob counts,
    on a fixed subset -- DIRECTIONAL MONITORING ONLY during training,
    see module docstring. Reuses Day 50/51/Stage-C's hand-validated
    functions unchanged."""
    from src.preprocessing.sample_writer import load_sample
    was_training = model.training
    model.eval()

    total_edits, total_ref = 0, 0
    tp_iou = fp_iou = fn_iou = 0
    tp_ct = fp_ct = fn_ct = 0
    pred_blob_counts, true_blob_counts = [], []
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
        r_iou = boundary_f1_for_sample(gt_binary, pred_binary, IOU_THRESH)
        tp_iou += r_iou["tp"]; fp_iou += r_iou["fp"]; fn_iou += r_iou["fn"]
        r_ct = boundary_f1_for_sample_center_tolerance(gt_binary, pred_binary, CENTER_TOLERANCE_TIMESTEPS)
        tp_ct += r_ct["tp"]; fp_ct += r_ct["fp"]; fn_ct += r_ct["fn"]
        true_blob_counts.append(len(extract_intervals(gt_binary)))
        pred_blob_counts.append(len(extract_intervals(pred_binary)))

    if was_training:
        model.train()

    def _prf1(tp, fp, fn):
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return p, r, f1

    corpus_wer = total_edits / max(total_ref, 1)
    p_iou, r_iou, f1_iou = _prf1(tp_iou, fp_iou, fn_iou)
    p_ct, r_ct, f1_ct = _prf1(tp_ct, fp_ct, fn_ct)
    return {
        "wer": corpus_wer,
        "boundary_precision": p_iou, "boundary_recall": r_iou, "boundary_f1": f1_iou,
        "ct_precision": p_ct, "ct_recall": r_ct, "ct_f1": f1_ct,
        "mean_pred_blob_count": sum(pred_blob_counts) / len(pred_blob_counts),
        "mean_true_blob_count": sum(true_blob_counts) / len(true_blob_counts),
    }


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
    parser.add_argument("--resume", action="store_true", help="Resume THIS run's own prior session (not first-launch from step_0008000.pt).")
    parser.add_argument("--early-stop-patience", type=int, default=DEFAULT_EARLY_STOP_PATIENCE)
    args = parser.parse_args()

    run_dir = Path("checkpoints/stageC_transition_loss_real")
    log_path = Path("logs") / "stageC_transition_loss_real.log"
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
        log_line(f"[FIRST LAUNCH] Stage C transition-loss real run -- "
                 f"loading model weights ONLY from {RESUME_CHECKPOINT} (fresh optimizer, "
                 f"new experimental branch, not a literal continuation)")
        loaded, skipped, new_only = load_checkpoint_partial(
            model, RESUME_CHECKPOINT,
            prefixes_to_load=("encoder.", "ctc_head.", "boundary_head.", "bimanual_gate_head."),
        )
        log_line(f"[FIRST LAUNCH] Transfer: {len(loaded)} loaded, {len(skipped)} skipped, {len(new_only)} new-only")

    train_dataset = SignLanguageDataset(
        npz_dir=TRAIN_NPZ_DIR, vocab_path=VOCAB_MERGED_PATH, asllrp_token_to_id_path=ASLLRP_TOKEN_TO_ID_PATH,
        augment=False, include_transition_target=True,
    )
    loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn,
        shuffle=True, num_workers=0,
    )
    val_dataset = SignLanguageDataset(
        npz_dir=VAL_NPZ_DIR, vocab_path=VOCAB_MERGED_PATH, asllrp_token_to_id_path=ASLLRP_TOKEN_TO_ID_PATH,
    )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=False)
    log_line(f"[VAL] Loaded val set: {len(val_dataset)} utterances")

    import random
    wer_probe_dataset = SignLanguageDataset(
        npz_dir=VAL_NPZ_DIR, vocab_path=VOCAB_MERGED_PATH, asllrp_token_to_id_path=ASLLRP_TOKEN_TO_ID_PATH,
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

    log_line(f"=== Stage C transition-loss REAL run start: begin_step={start_step}, device={device}, "
              f"boundary_loss_weight={BOUNDARY_LOSS_WEIGHT}, transition_loss_weight={TRANSITION_LOSS_WEIGHT}, "
              f"blank_logit_bias={BLANK_LOGIT_BIAS} ===")
    log_line(f"[TIME BUDGET] hard backstop {EFFECTIVE_TRAIN_LIMIT_SECONDS/60:.0f} min "
              f"(NOT the primary stop mechanism -- see module docstring)")
    log_line(f"[CHECKPOINT POLICY] No cleanup -- every step-numbered checkpoint is kept for the post-hoc "
              f"full-set sweep, deliberate deviation from stageB1_real_training_run.py -- see module docstring")

    step = start_step
    session_start_time = time.time()
    session_steps = 0
    ctc_losses, boundary_losses, transition_losses = [], [], []
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
                    raise StopIteration

                batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                ctc_log_probs, boundary_logits, gate_logits = model.forward_ctc_boundary_and_gate(
                    batch["landmarks"], batch["encoder_attention_mask"]
                )
                total_loss, ctc_loss, boundary_loss, _ge, transition_loss = compute_combined_loss_with_gate_and_transition(
                    ctc_log_probs, boundary_logits, gate_logits,
                    batch["target_tokens"], batch["boundary_target"], batch["encoder_attention_mask"],
                    bos_id, eos_id, pad_id, blank_id,
                    boundary_loss_weight=BOUNDARY_LOSS_WEIGHT, gate_reg_weight=GATE_REG_WEIGHT,
                    blank_logit_bias=BLANK_LOGIT_BIAS,
                    transition_target=batch["transition_target"], transition_loss_weight=TRANSITION_LOSS_WEIGHT,
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
                transition_losses.append(transition_loss.item())

                if step % LOG_EVERY_N_STEPS == 0:
                    steps_per_min = (session_steps + 1) / (elapsed / 60) if elapsed > 0 else 0.0
                    log_line(f"Step {step}: ctc_loss={ctc_loss.item():.4f} boundary_loss={boundary_loss.item():.4f} "
                              f"transition_loss={transition_loss.item():.4f} grad_norm={grad_norm:.4f} steps/min={steps_per_min:.1f}")

                if step % CHECKPOINT_EVERY_N_STEPS == 0 and step > start_step:
                    save_checkpoint(model, optimizer, step, run_dir / f"step_{step:07d}.pt")
                    log_line(f"[CHECKPOINT] Saved (step-count, KEPT -- no cleanup this run): step_{step:07d}.pt")
                    last_checkpoint_time = time.time()
                elif time.time() - last_checkpoint_time >= CHECKPOINT_EVERY_N_SECONDS:
                    save_checkpoint(model, optimizer, step, run_dir / f"step_{step:07d}.pt")
                    log_line(f"[CHECKPOINT] Saved (time-floor, KEPT -- no cleanup this run): step_{step:07d}.pt")
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
                            raise StopIteration

                if step % WER_PROBE_EVERY_N_STEPS == 0 and step > start_step:
                    probe = wer_and_boundary_f1_probe(model, wer_probe_dataset, wer_probe_indices, device, blank_id)
                    log_line(f"[WER PROBE] Step {step}: wer={probe['wer']:.4f} "
                              f"iouF1={probe['boundary_f1']:.4f} (P={probe['boundary_precision']:.4f} R={probe['boundary_recall']:.4f}) "
                              f"ctF1={probe['ct_f1']:.4f} (P={probe['ct_precision']:.4f} R={probe['ct_recall']:.4f}) "
                              f"pred_blobs={probe['mean_pred_blob_count']:.2f} true_blobs={probe['mean_true_blob_count']:.2f} "
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
                  f"best_by_wer (30-sample probe, DIRECTIONAL ONLY): step={best_wer_step} wer={best_wer_value:.4f}")
        elapsed = time.time() - session_start_time
        log_line(f"=== Session end: step={step}, session_steps={session_steps}, "
                  f"elapsed={elapsed/60:.2f} min, steps/min={session_steps/(elapsed/60) if elapsed > 0 else 0:.1f} ===")


if __name__ == "__main__":
    main()
