"""
scripts/stageA_blank_and_boundary_smoke_test.py

Stage A of the improvement campaign: cheap, isolated smoke tests only.
NOT a real training run -- no checkpoints saved, no WER/boundary-F1
scoring (explicitly out of scope: a few hundred steps is too early for
decode-level metrics to mean anything, per Day 33's own warning about
reading too much into short-run numbers). This checks ONLY whether two
specific, already-diagnosed symptoms show early directional movement:

  1. CTC blank-dominance: 96.7% of timesteps predict blank even on
     step_0008000.pt (the current checkpoint of record).
  2. Boundary under-segmentation: the boundary head merges multiple
     real sign spans into one blob (FN >> FP, Day 51).

Two independent levers, tested in isolation, then combined -- never
mixed with data augmentation or any other change, so any movement found
is attributable to a specific cause:
  - blank_logit_bias (src/models/combined_loss.py, Stage A addition,
    OFF by default): downweights the CTC blank channel directly in the
    loss computation.
  - boundary_loss_weight (already existed, Day 39's open question):
    how much the boundary BCE term counts in the combined loss.

--- COMBINATIONS TESTED, AND WHY (judgment call, not a full grid) ---
  1. control        (blank_logit_bias=0.0, boundary_loss_weight=1.0)
     Reproduces current defaults exactly. Needed as a same-scale
     baseline: even "nothing changed" may drift a little over 500 more
     steps of continued training, so every other run is judged against
     THIS run's movement, not against step_0008000's frozen starting
     numbers.
  2. blank_alone     (blank_logit_bias=3.0, boundary_loss_weight=1.0)
     Isolates the blank-downweighting lever.
  3. boundary_2      (blank_logit_bias=0.0, boundary_loss_weight=2.0)
  4. boundary_4      (blank_logit_bias=0.0, boundary_loss_weight=4.0)
     Isolates the boundary-weight lever, at 2 candidate values (the
     3-value sweep the task asked for is {1.0 (=control), 2.0, 4.0}).
  5. combined        (blank_logit_bias=3.0, boundary_loss_weight=<best
     of run 3 vs run 4>) -- run only after inspecting 3 vs 4's results,
     combining blank-downweighting with whichever boundary weight
     looked more promising alone. This is NOT a full 2x3 grid (6 runs);
     it's the specific "alone, alone, then best-of-each-combined" path
     the task asked for, stated explicitly rather than guessed at
     silently.

blank_logit_bias=3.0 chosen as a single representative magnitude (not
swept) because the task's explicit >=3-value sweep requirement was for
boundary_loss_weight specifically; demonstrating the new mechanism's
directional effect at one reasonable magnitude is what Stage A needs to
decide whether it's worth a real sweep of ITS OWN in a later stage.

--- SCALE ---
500 steps per run (task's allowed range: "a few hundred to ~1000,
same scale discipline as Days 32-33" -- Day 33's own extended smoke
test used 150; 500 sits inside the allowed range and gives more room to
see a trend than 150 would, without approaching a real training run).
Resumes from step_0008000.pt via load_checkpoint_partial (model weights
only, fresh optimizer -- this is a new experimental branch, not a
literal continuation of that run, same convention every other
downstream script in this project already uses).

--- SIGNAL PROBE: same fixed 25-sample subset as Day 54's screening ---
Reused verbatim (not re-sampled) for continuity/transparency. Probed at
step 0 (baseline, before any of this run's training) and every 100
steps: mean blank-argmax-rate, and mean predicted positive-region
("blob") count from the boundary head (via extract_intervals, reused
from compute_boundary_f1.py) alongside each sample's REAL ground-truth
blob count (its actual number of real sign spans) for direct
comparison. This is cheap (one forward pass per sample, no beam
search, no IoU matching) and deliberately NOT WER/boundary-F1 scoring.
"""
import sys
import json
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader

from src.models.sign_transformer import SignTransformer
from src.models.train_utils import load_checkpoint_partial, save_checkpoint, get_device
from src.models.dataset import SignLanguageDataset, collate_fn
from src.models.combined_loss import compute_combined_loss_with_gate
from src.preprocessing.sample_writer import load_sample
from scripts.compute_boundary_f1 import extract_intervals

VOCAB_MERGED_PATH = "data/vocab_merged.json"
DECODER_VOCAB_PATH = "data/vocab.json"
ASLLRP_TOKEN_TO_ID_PATH = "data/asllrp_token_to_id.json"
TRAIN_NPZ_DIR = "data/processed/asllrp/train"
TEST_NPZ_DIR = "data/processed/asllrp/test"
RESUME_CHECKPOINT = "checkpoints/month2_asllrp_gated/step_0008000.pt"  # current checkpoint of record

BATCH_SIZE = 4
LEARNING_RATE = 1e-4
GRAD_CLIP_MAX_NORM = 1.0
GATE_REG_WEIGHT = 0.0  # unchanged, Day 44 decision -- not what Stage A tests
NUM_STEPS = 500
PROBE_EVERY_N_STEPS = 100
BOUNDARY_PROB_THRESHOLD = 0.5

# Reproducibility + checkpoint persistence -- added after the FIRST Stage A
# run turned out to be undiagnosable after the fact: no seed meant a rerun
# couldn't recover the same trajectory, and no saved checkpoints meant
# there was nothing to load for a follow-up diagnostic (the blank_alone-
# moved-the-wrong-way finding). Both are now standard for every future
# Stage-A-style smoke test, not a one-off fix: TRAIN_SEED fixes model-
# resume-state randomness (none here, since weights come from a
# checkpoint) AND, via a seeded DataLoader generator, the batch/shuffle
# order -- the SAME seed for every config, so configs differ ONLY in
# their loss hyperparameters, never in which batches they happened to see.
TRAIN_SEED = 55
CHECKPOINT_SAVE_DIR = Path("checkpoints/stageA_smoke")  # <config_name>/step_XXXX.pt

# Same 25-sample subset Day 54's screening used (seed=54), reused verbatim.
SCREEN_SEED = 54
SCREEN_SIZE = 25

CONFIGS = [
    ("control",      0.0, 1.0),
    ("blank_alone",  3.0, 1.0),
    ("boundary_2",   0.0, 2.0),
    ("boundary_4",   0.0, 4.0),
    # ("combined", 3.0, <filled in after inspecting boundary_2 vs boundary_4>)
]


def pick_probe_samples():
    dataset = SignLanguageDataset(
        npz_dir=TEST_NPZ_DIR, vocab_path=VOCAB_MERGED_PATH,
        asllrp_token_to_id_path=ASLLRP_TOKEN_TO_ID_PATH,
    )
    n = len(dataset)
    rng = random.Random(SCREEN_SEED)
    indices = sorted(rng.sample(range(n), min(SCREEN_SIZE, n)))
    return dataset, indices


@torch.no_grad()
def probe_signals(model, probe_dataset, probe_indices, device, blank_id):
    """Mean blank-argmax-rate and mean predicted/true blob count over the
    fixed probe subset. No beam search, no IoU matching -- deliberately
    cheaper and different from WER/boundary-F1 scoring."""
    model.eval()
    blank_rates = []
    pred_blob_counts = []
    true_blob_counts = []

    for idx in probe_indices:
        item = probe_dataset[idx]
        meta = load_sample(probe_dataset.npz_paths[idx]).metadata
        landmarks = item["landmarks"].unsqueeze(0).to(device)
        T = landmarks.shape[1]
        mask = torch.ones(1, T, dtype=torch.bool, device=device)

        ctc_log_probs, boundary_logits, _gate_logits = model.forward_ctc_boundary_and_gate(landmarks, mask)
        argmax_ids = ctc_log_probs[0].argmax(dim=-1)
        blank_rates.append((argmax_ids == blank_id).float().mean().item())

        pred_binary = (torch.sigmoid(boundary_logits[0]) >= BOUNDARY_PROB_THRESHOLD).cpu().numpy().astype(int)
        pred_blob_counts.append(len(extract_intervals(pred_binary)))
        # true blob count = number of real sign spans for this utterance
        true_blob_counts.append(sum(1 for s in meta.sign_spans if s is not None))

    model.train()
    return {
        "mean_blank_rate": sum(blank_rates) / len(blank_rates),
        "mean_pred_blob_count": sum(pred_blob_counts) / len(pred_blob_counts),
        "mean_true_blob_count": sum(true_blob_counts) / len(true_blob_counts),
    }


def run_one_config(name, blank_logit_bias, boundary_loss_weight, probe_dataset, probe_indices, device,
                    ctc_vocab_size, decoder_vocab_size, blank_id, bos_id, eos_id, pad_id):
    print(f"\n{'='*90}\nCONFIG: {name}  (blank_logit_bias={blank_logit_bias}, boundary_loss_weight={boundary_loss_weight})\n{'='*90}")

    # Reproducibility: same seed for every config, set right before the
    # DataLoader is built (and its generator is separately seeded too),
    # so every config sees the IDENTICAL batch/shuffle order -- configs
    # differ only in loss hyperparameters, never in which data they saw.
    torch.manual_seed(TRAIN_SEED)

    model = SignTransformer(
        vocab_size=decoder_vocab_size, ctc_vocab_size=ctc_vocab_size, use_bimanual_gate=True,
    ).to(device)
    loaded, skipped, new_only = load_checkpoint_partial(
        model, RESUME_CHECKPOINT,
        prefixes_to_load=("encoder.", "ctc_head.", "boundary_head.", "bimanual_gate_head."),
    )
    print(f"  Checkpoint load: {len(loaded)} loaded, {len(skipped)} skipped, {len(new_only)} new-only")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    train_dataset = SignLanguageDataset(
        npz_dir=TRAIN_NPZ_DIR, vocab_path=VOCAB_MERGED_PATH, asllrp_token_to_id_path=ASLLRP_TOKEN_TO_ID_PATH,
    )
    shuffle_generator = torch.Generator().manual_seed(TRAIN_SEED)
    loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn,
        shuffle=True, num_workers=0, generator=shuffle_generator,
    )

    ckpt_dir = CHECKPOINT_SAVE_DIR / name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    probe_log = []
    baseline = probe_signals(model, probe_dataset, probe_indices, device, blank_id)
    probe_log.append({"step": 0, **baseline})
    print(f"  [PROBE step=0 baseline] blank_rate={baseline['mean_blank_rate']:.4f}  "
          f"pred_blob_count={baseline['mean_pred_blob_count']:.2f}  "
          f"true_blob_count={baseline['mean_true_blob_count']:.2f}")

    ctc_losses, boundary_losses = [], []
    step = 0
    model.train()
    while step < NUM_STEPS:
        for batch in loader:
            if step >= NUM_STEPS:
                break
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

            ctc_log_probs, boundary_logits, gate_logits = model.forward_ctc_boundary_and_gate(
                batch["landmarks"], batch["encoder_attention_mask"]
            )
            total_loss, ctc_loss, boundary_loss, _gate_entropy = compute_combined_loss_with_gate(
                ctc_log_probs, boundary_logits, gate_logits,
                batch["target_tokens"], batch["boundary_target"],
                batch["encoder_attention_mask"],
                bos_id, eos_id, pad_id, blank_id,
                boundary_loss_weight=boundary_loss_weight,
                gate_reg_weight=GATE_REG_WEIGHT,
                blank_logit_bias=blank_logit_bias,
            )

            if not torch.isfinite(total_loss):
                print(f"  [WARNING] step {step}: non-finite loss, skipping batch")
                step += 1
                continue

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_MAX_NORM)
            optimizer.step()

            ctc_losses.append(ctc_loss.item())
            boundary_losses.append(boundary_loss.item())
            step += 1

            if step % PROBE_EVERY_N_STEPS == 0 or step == NUM_STEPS:
                recent_ctc = sum(ctc_losses[-PROBE_EVERY_N_STEPS:]) / len(ctc_losses[-PROBE_EVERY_N_STEPS:])
                recent_boundary = sum(boundary_losses[-PROBE_EVERY_N_STEPS:]) / len(boundary_losses[-PROBE_EVERY_N_STEPS:])
                sig = probe_signals(model, probe_dataset, probe_indices, device, blank_id)
                probe_log.append({"step": step, **sig})
                print(f"  [step={step:4d}] mean_ctc_loss={recent_ctc:.4f}  mean_boundary_loss={recent_boundary:.4f}  |  "
                      f"blank_rate={sig['mean_blank_rate']:.4f}  pred_blob_count={sig['mean_pred_blob_count']:.2f}  "
                      f"true_blob_count={sig['mean_true_blob_count']:.2f}")

                # Standard practice now (was the gap that made the first Stage A
                # run undiagnosable after the fact): persist a real checkpoint
                # at every probe interval, not just log the aggregate numbers.
                ckpt_path = ckpt_dir / f"step_{step:04d}.pt"
                save_checkpoint(model, optimizer, step, ckpt_path)
                print(f"  [CHECKPOINT] saved {ckpt_path}")

    return {
        "name": name, "blank_logit_bias": blank_logit_bias, "boundary_loss_weight": boundary_loss_weight,
        "probe_log": probe_log,
        "final_mean_ctc_loss_last100": sum(ctc_losses[-100:]) / len(ctc_losses[-100:]),
        "final_mean_boundary_loss_last100": sum(boundary_losses[-100:]) / len(boundary_losses[-100:]),
        "n_steps": step,
        "checkpoint_dir": str(ckpt_dir),
        "train_seed": TRAIN_SEED,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", type=str, default=None,
                         help="Comma-separated subset of config names to run (default: all in CONFIGS). "
                              "E.g. --configs control,blank_alone for a targeted rerun/diagnostic.")
    parser.add_argument("--out", type=str, default="logs/stageA_smoke_test_results.json",
                         help="Output path -- override for a rerun so the original file isn't overwritten.")
    args = parser.parse_args()

    configs_to_run = CONFIGS
    if args.configs:
        wanted = set(args.configs.split(","))
        configs_to_run = [c for c in CONFIGS if c[0] in wanted]
        missing = wanted - {c[0] for c in configs_to_run}
        if missing:
            raise SystemExit(f"Unknown config name(s): {missing}. Known: {[c[0] for c in CONFIGS]}")

    device = get_device()
    with open(VOCAB_MERGED_PATH) as f:
        vocab_merged = json.load(f)
    with open(DECODER_VOCAB_PATH) as f:
        decoder_vocab = json.load(f)
    ctc_vocab_size = len(vocab_merged)
    blank_id = ctc_vocab_size
    bos_id, eos_id, pad_id = vocab_merged["<BOS>"], vocab_merged["<EOS>"], vocab_merged["<PAD>"]

    probe_dataset, probe_indices = pick_probe_samples()
    print(f"Probe subset: {len(probe_indices)} fixed test-split samples (seed={SCREEN_SEED}, same as Day 54 screening)")
    print(f"Running configs: {[c[0] for c in configs_to_run]}  (train_seed={TRAIN_SEED})")

    results = []
    for name, bias, bw in configs_to_run:
        r = run_one_config(name, bias, bw, probe_dataset, probe_indices, device,
                            ctc_vocab_size, len(decoder_vocab), blank_id, bos_id, eos_id, pad_id)
        results.append(r)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\n\nDone with configs: {[c[0] for c in configs_to_run]}. Results in {args.out}, checkpoints under {CHECKPOINT_SAVE_DIR}/<config_name>/")


if __name__ == "__main__":
    main()
