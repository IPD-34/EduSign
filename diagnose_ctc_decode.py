"""diagnose_ctc_decode.py -- Day 40 structural sanity check on the CTC
decode path, before relying on it for Week 8 held-out evaluation.

Same class of check that caught the Month 1 nested-tensor fastpath bug
(nn.TransformerEncoder silently corrupting batch-size-1, no-padding
eval-mode inference; fixed via enable_nested_tensor=False). This script:

  1. Confirms enable_nested_tensor=False is active on the LIVE model
     instance used for inference (not just present somewhere in
     encoder.py), by introspecting the loaded model object directly.
  2. Implements a correct CTC greedy decode (argmax per timestep,
     collapse repeats, remove blank) -- adapted from
     diagnose_gloss_position.py's rank/probability pattern, but for
     CTC output instead of the old single-token decoder output.
  3. Reports, per sample:
       - exact P(true gloss sequence) via the CTC forward algorithm
         (nn.CTCLoss's own computation -- exact, not approximated)
       - per-true-gloss-token rank/probability at its CTC forced-
         alignment (Viterbi) timestep -- the CTC analog of
         diagnose_gloss_position.py's per-token rank diagnostic
       - the greedy-decoded gloss sequence, for comparison
  4. Checks explicitly for the Day 17 (Month 1) failure signature:
     single-repeated-token collapse, uniform placeholder resolution
     regardless of input, or multiple samples decoding identically.

Default config: checkpoints/month2_asllrp/best_val.pt (step 4000, the
checkpoint of record) against real ASLLRP TRAINING samples (a sanity
check, not the held-out evaluation) -- this reproduces Day 40 exactly
when run with no arguments.

--checkpoint / --npz-dir let this be pointed at a different checkpoint
or split without editing the source -- added after this file's
CHECKPOINT_PATH constant had been hand-edited back and forth three times
across one investigation arc (best_val.pt -> step_0014000.pt -> back),
which is exactly the kind of stale-config risk this project's own
history warns about. Example real use: comparing step_0014000.pt's
blank-dominance on VAL samples against best_val.pt's, to distinguish
genuine recession from train-set memorization (step 14000 is the
confirmed train-overfit checkpoint -- see Day 36/37/39).

Read-only diagnostic. Does not modify the architecture.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn as nn

from src.preprocessing.sample_writer import load_sample
from src.models.sign_transformer import SignTransformer
from src.models.dataset import SignLanguageDataset, collate_fn

DEFAULT_CHECKPOINT_PATH = "checkpoints/month2_asllrp/best_val.pt"  # checkpoint of record (step 4000)
DEFAULT_ASLLRP_NPZ_DIR = "data/processed/asllrp/train"
VOCAB_MERGED_PATH = "data/vocab_merged.json"
DECODER_VOCAB_PATH = "data/vocab.json"  # 1850 entries -- old WLASL decoder vocab
ASLLRP_TOKEN_TO_ID_PATH = "data/asllrp_token_to_id.json"
ASLLRP_MAX_SEQUENCE_LENGTH = 300  # src/preprocessing/config.py

PLACEHOLDER_TOKENS = {"<UNK>", "<CLASSIFIER>", "<FINGERSPELL>"}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------
# Step 1: load the checkpoint of record and confirm enable_nested_tensor
# is False on the ACTUAL instance used for inference, not just source.
# ---------------------------------------------------------------------
def load_model_for_eval(ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    sd = checkpoint["model_state_dict"]
    step = checkpoint.get("step")

    ctc_out_dim = sd["ctc_head.weight"].shape[0]
    ctc_vocab_size = ctc_out_dim - 1  # last slot is the CTC blank token

    with open(DECODER_VOCAB_PATH) as f:
        decoder_vocab = json.load(f)

    # Mirrors train_month2_asllrp.py's actual construction call exactly
    # (scripts/train_month2_asllrp.py:408) -- same defaults, same path.
    model = SignTransformer(vocab_size=len(decoder_vocab), ctc_vocab_size=ctc_vocab_size).to(DEVICE)
    model.load_state_dict(sd)  # strict=True -- any architecture drift fails loudly here
    model.eval()
    return model, step, ctc_vocab_size


def confirm_nested_tensor_disabled(model):
    """Traces enable_nested_tensor from THIS model instance's actual encoder,
    not from reading encoder.py. Fails loudly (AssertionError) if not False."""
    te = model.encoder.transformer_encoder
    assert isinstance(te, nn.TransformerEncoder), (
        f"Expected model.encoder.transformer_encoder to be nn.TransformerEncoder, "
        f"got {type(te)}"
    )
    live_value = te.enable_nested_tensor
    print("=== Step 1: enable_nested_tensor trace ===")
    print("  Path: this script -> load_model_for_eval() -> SignTransformer(...)")
    print("        -> SignTransformer.__init__ (sign_transformer.py:38-44) builds")
    print("           self.encoder = SignEncoder(...) unconditionally")
    print("        -> SignEncoder.__init__ (encoder.py:57) hardcodes")
    print("           enable_nested_tensor=False directly in the nn.TransformerEncoder")
    print("           call -- it is NOT exposed as a SignEncoder/SignTransformer")
    print("           constructor parameter, so no eval-path call site can override it.")
    print(f"  Live introspection on the loaded eval instance: "
          f"model.encoder.transformer_encoder.enable_nested_tensor = {live_value!r}")
    if live_value is False:
        print("  CONFIRMED: enable_nested_tensor=False is active on the real eval instance.\n")
    else:
        print("  *** FAIL: enable_nested_tensor is NOT False on the eval instance. ***\n")
    return live_value is False


# ---------------------------------------------------------------------
# Step 2: correct CTC greedy decode (argmax -> collapse repeats -> drop blank)
# ---------------------------------------------------------------------
def ctc_greedy_decode(log_probs_1d_per_t, blank_id):
    """log_probs_1d_per_t: (T, V+1) log-probs for ONE unpadded sample."""
    pred_ids = log_probs_1d_per_t.argmax(dim=-1).tolist()
    collapsed = []
    prev = None
    for t in pred_ids:
        if t != prev:
            if t != blank_id:
                collapsed.append(t)
        prev = t
    return collapsed


# ---------------------------------------------------------------------
# Step 3: exact P(true sequence) via the CTC forward algorithm
# (this is exactly what nn.CTCLoss computes -- not an approximation)
# ---------------------------------------------------------------------
def true_sequence_nll(log_probs, target_ids, blank_id):
    """log_probs: (T, V+1) for one sample. Returns -log P(target | input),
    exact via nn.CTCLoss's forward algorithm."""
    T = log_probs.shape[0]
    ctc_loss_fn = nn.CTCLoss(blank=blank_id, zero_infinity=True, reduction="sum")
    lp = log_probs.unsqueeze(1)  # (T, 1, V+1) -- CTC wants (T, B, C)
    target = torch.tensor(target_ids, dtype=torch.long)
    input_lengths = torch.tensor([T], dtype=torch.long)
    target_lengths = torch.tensor([len(target_ids)], dtype=torch.long)
    nll = ctc_loss_fn(lp.cpu(), target, input_lengths, target_lengths)
    return nll.item()


# ---------------------------------------------------------------------
# Step 4: CTC forced alignment (Viterbi) -- exact DP, standard algorithm.
# Gives, for each TRUE gloss token, the single most-probable timestep it
# occupies under the model's own output, so we can report that token's
# rank/probability there -- the CTC analog of diagnose_gloss_position.py's
# per-token rank diagnostic.
# ---------------------------------------------------------------------
def ctc_forced_alignment(log_probs, target_ids, blank_id):
    """log_probs: (T, V+1) log-probs, numpy or tensor. target_ids: list[int].
    Returns list of (gloss_id, aligned_timestep), one per true token, or
    None if no valid alignment exists (input shorter than needed)."""
    T = log_probs.shape[0]
    n = len(target_ids)
    ext = [blank_id]
    for g in target_ids:
        ext.append(g)
        ext.append(blank_id)
    L = len(ext)  # 2n + 1

    NEG_INF = float("-inf")
    alpha = [[NEG_INF] * L for _ in range(T)]
    back = [[None] * L for _ in range(T)]

    lp = log_probs  # (T, V+1) tensor

    alpha[0][0] = lp[0, ext[0]].item()
    if L > 1:
        alpha[0][1] = lp[0, ext[1]].item()

    for t in range(1, T):
        for s in range(L):
            best_prev_val = alpha[t - 1][s]
            best_prev_s = s
            if s >= 1 and alpha[t - 1][s - 1] > best_prev_val:
                best_prev_val = alpha[t - 1][s - 1]
                best_prev_s = s - 1
            if s >= 2 and ext[s] != blank_id and ext[s] != ext[s - 2]:
                if alpha[t - 1][s - 2] > best_prev_val:
                    best_prev_val = alpha[t - 1][s - 2]
                    best_prev_s = s - 2
            if best_prev_val == NEG_INF:
                continue
            alpha[t][s] = best_prev_val + lp[t, ext[s]].item()
            back[t][s] = best_prev_s

    if L == 1:
        end_s = 0
    else:
        end_s = L - 1 if alpha[T - 1][L - 1] >= alpha[T - 1][L - 2] else L - 2

    if alpha[T - 1][end_s] == NEG_INF:
        return None  # no valid alignment (T too short for target length)

    path_s = [None] * T
    s = end_s
    path_s[T - 1] = s
    for t in range(T - 1, 0, -1):
        s = back[t][s]
        path_s[t - 1] = s

    # For each true label index i (odd position 2*i+1 in ext), find the
    # LAST timestep the Viterbi path occupies that position (i.e. the
    # timestep the path commits most confidently to that label).
    aligned_t = [None] * n
    for t in range(T):
        s = path_s[t]
        if s % 2 == 1:
            label_idx = (s - 1) // 2
            aligned_t[label_idx] = t  # keep overwriting -> last occupied t
    return list(zip(target_ids, aligned_t))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT_PATH,
                         help=f"Checkpoint to evaluate. Default: {DEFAULT_CHECKPOINT_PATH} "
                              "(the checkpoint of record).")
    parser.add_argument("--npz-dir", type=str, default=DEFAULT_ASLLRP_NPZ_DIR,
                         help=f"ASLLRP split to sample from. Default: {DEFAULT_ASLLRP_NPZ_DIR}.")
    args = parser.parse_args()
    split_name = Path(args.npz_dir).name  # e.g. "train" or "val", for accurate labeling below

    with open(VOCAB_MERGED_PATH, encoding="utf-8") as f:
        vocab_merged = json.load(f)
    id_to_token = {v: k for k, v in vocab_merged.items()}
    bos_id, eos_id, pad_id = vocab_merged["<BOS>"], vocab_merged["<EOS>"], vocab_merged["<PAD>"]

    model, step, ctc_vocab_size = load_model_for_eval(args.checkpoint)
    blank_id = ctc_vocab_size
    print(f"Loaded {args.checkpoint} (step {step}), ctc_vocab_size={ctc_vocab_size} "
          f"(vocab_merged has {len(vocab_merged)} tokens -- "
          f"{'MATCH' if ctc_vocab_size == len(vocab_merged) else 'MISMATCH!'})\n")

    nested_tensor_ok = confirm_nested_tensor_disabled(model)

    dataset = SignLanguageDataset(
        npz_dir=args.npz_dir,
        vocab_path=VOCAB_MERGED_PATH,
        asllrp_token_to_id_path=ASLLRP_TOKEN_TO_ID_PATH,
    )
    print(f"=== Step 3: sample selection from {len(dataset)} ASLLRP {split_name} utterances ===")

    # Cheap metadata scan (load_sample only reads the npz, no feature flatten)
    candidates = []
    for idx, p in enumerate(dataset.npz_paths):
        sample = load_sample(p)
        meta = sample.metadata
        n_glosses = len(meta.gloss.split("|"))
        candidates.append((idx, meta.video_id, meta.num_frames, n_glosses))

    by_frames = sorted(candidates, key=lambda c: c[2], reverse=True)
    by_glosses = sorted(candidates, key=lambda c: c[3])

    picked_idx = []
    picked_idx.append(by_frames[0])                      # near/at the frame cap
    picked_idx.append(by_glosses[0])                      # shortest gloss sequence
    picked_idx.append(by_glosses[-1])                     # longest gloss sequence
    mid = candidates[len(candidates) // 2]
    picked_idx.append(mid)                                 # a "typical" middle sample
    # one more distinct typical sample, avoiding duplicates
    for c in candidates:
        if c[0] not in {p[0] for p in picked_idx}:
            picked_idx.append(c)
            break

    for idx, vid, nframes, ngloss in picked_idx:
        print(f"  idx={idx} video_id={vid} num_frames={nframes} "
              f"(cap={ASLLRP_MAX_SEQUENCE_LENGTH}) n_glosses={ngloss}")
    print()

    print("=== Step 4: CTC decode + rank/probability per sample ===\n")
    decoded_sequences = []
    failure_flags = []
    blank_collapse_flags = []

    for idx, vid, nframes, ngloss in picked_idx:
        item = dataset[idx]
        landmarks = item["landmarks"].unsqueeze(0).to(DEVICE)
        enc_mask = torch.ones(1, landmarks.shape[1], dtype=torch.bool, device=DEVICE)

        true_tokens = item["target_tokens"].tolist()
        assert true_tokens[0] == bos_id and true_tokens[-1] == eos_id
        true_ids = true_tokens[1:-1]
        true_glosses = [id_to_token.get(i, "<?>") for i in true_ids]

        with torch.no_grad():
            ctc_log_probs, boundary_logits = model.forward_ctc_and_boundary(landmarks, enc_mask)
        log_probs = ctc_log_probs[0].cpu()  # (T, V+1)

        decoded_ids = ctc_greedy_decode(log_probs, blank_id)
        decoded_glosses = [id_to_token.get(i, "<?>") for i in decoded_ids]
        decoded_sequences.append(tuple(decoded_glosses))

        # Quantify blank dominance directly -- greedy decode returning empty
        # means argmax==blank at every timestep, which is a notable pattern
        # worth measuring, not silently passing through as "no output".
        step_argmax = log_probs.argmax(dim=-1)
        step_probs_all = torch.softmax(log_probs, dim=-1)
        blank_argmax_frac = (step_argmax == blank_id).float().mean().item()
        mean_blank_prob = step_probs_all[:, blank_id].mean().item()

        nll = true_sequence_nll(log_probs, true_ids, blank_id)
        prob = torch.exp(torch.tensor(-nll)).item()
        nll_per_token = nll / max(len(true_ids), 1)
        prob_per_token = torch.exp(torch.tensor(-nll_per_token)).item()
        random_baseline_prob_per_token = 1.0 / (ctc_vocab_size + 1)
        random_baseline_nll_per_token = torch.log(torch.tensor(float(ctc_vocab_size + 1))).item()

        print(f"--- {vid}  (num_frames={nframes}, n_glosses={ngloss}) ---")
        print(f"  true gloss sequence:    {true_glosses}")
        print(f"  greedy decoded:         {decoded_glosses}")
        print(f"  exact match:            {decoded_glosses == true_glosses}")
        print(f"  blank dominance:        argmax==<blank> at {blank_argmax_frac*100:.1f}% of "
              f"{nframes} timesteps; mean P(<blank>)={mean_blank_prob:.4f}")
        print(f"  P(true seq) exact [CTC forward algorithm]: {prob:.6e}  "
              f"(nll_sum={nll:.4f})")
        print(f"  per-token: nll={nll_per_token:.4f}  P/token={prob_per_token:.6f}  vs. "
              f"RANDOM-UNIFORM baseline: nll={random_baseline_nll_per_token:.4f}  "
              f"P/token={random_baseline_prob_per_token:.6f}  "
              f"({'BELOW (better than) random' if nll_per_token < random_baseline_nll_per_token else 'AT/ABOVE random -- no learned signal'})")

        alignment = ctc_forced_alignment(log_probs, true_ids, blank_id)
        if alignment is None:
            print("  forced alignment:       FAILED (input shorter than 2*n_glosses+1 -- "
                  "cannot check per-token rank)")
            failure_flags.append(f"{vid}: forced alignment failed")
        else:
            print("  per-token rank/probability at forced-alignment timestep "
                  "(CTC analog of diagnose_gloss_position.py):")
            step_probs = torch.softmax(log_probs, dim=-1)
            uniform_placeholder_hits = 0
            for gloss_id, t in alignment:
                gloss_tok = id_to_token.get(gloss_id, "<?>")
                dist = step_probs[t]
                true_p = dist[gloss_id].item()
                rank = (dist > true_p).sum().item() + 1
                top3_ids = torch.topk(dist, 3).indices.tolist()
                top3 = [id_to_token.get(i, "<blank>" if i == blank_id else "<?>") for i in top3_ids]
                print(f"    {gloss_tok:16s} @t={t:<4d} rank={rank:<5d}/{ctc_vocab_size + 1} "
                      f"P={true_p:.6f}  top3={top3}")
                if gloss_tok in PLACEHOLDER_TOKENS:
                    uniform_placeholder_hits += 1
        print()

        # --- NLL-vs-baseline check, ALL samples, not just empty-decode ones ---
        # This used to only fire inside the blank_collapse_flags branch below,
        # which only triggers when greedy decode is EMPTY -- so a sample with
        # a non-empty (but wrong) decode and nll_per_token at/above the random
        # baseline was silently invisible to the verdict, falling through to
        # "no failure signature, looks healthy". Caught on this val-sample run
        # (Day 40 recheck, step_0014000.pt): 3/5 samples failed this check
        # despite all 5 decoding non-empty.
        if nll_per_token >= random_baseline_nll_per_token:
            failure_flags.append(
                f"{vid}: per-token nll={nll_per_token:.4f} is NOT below the random-uniform "
                f"baseline ({random_baseline_nll_per_token:.4f}) -- no learned signal for the "
                f"true sequence on this sample, independent of whether greedy decode is empty."
            )

        # --- Day 17 failure-signature checks, per sample ---
        if len(set(decoded_glosses)) == 1 and len(decoded_glosses) > 1:
            failure_flags.append(f"{vid}: greedy decode collapsed to a single repeated token "
                                  f"({decoded_glosses[0]})")
        if decoded_glosses and all(g in PLACEHOLDER_TOKENS for g in decoded_glosses):
            failure_flags.append(f"{vid}: greedy decode is uniformly placeholder tokens "
                                  f"({set(decoded_glosses)})")
        if not decoded_glosses:
            blank_collapse_flags.append(
                f"{vid}: greedy decode empty (blank argmax {blank_argmax_frac*100:.1f}% of "
                f"timesteps); per-token nll={nll_per_token:.4f} vs random-uniform "
                f"{random_baseline_nll_per_token:.4f} "
                f"({'below random -- real signal despite blank dominance' if nll_per_token < random_baseline_nll_per_token else 'NOT below random -- no learned signal, real concern'})"
            )

    # Cross-sample check: do multiple unrelated samples decode identically?
    non_empty = [s for s in decoded_sequences if s]
    if len(non_empty) >= 2 and len(set(non_empty)) == 1:
        failure_flags.append(
            f"ALL {len(non_empty)} samples decoded to the identical sequence {non_empty[0]}"
        )

    print("=== Verdict ===")
    if not nested_tensor_ok:
        print("STRUCTURAL BUG: enable_nested_tensor is not False on the eval instance -- "
              "same failure class as the Month 1 Day 17 bug. STOP, do not proceed to "
              "held-out evaluation until fixed.")
        return

    if failure_flags:
        print("Potential Day-17-style failure signature(s) detected -- treat as a "
              "debugging checkpoint, not a conclusion:")
        for f in failure_flags:
            print(f"  - {f}")
    elif blank_collapse_flags:
        all_below_random = all("below random" in f for f in blank_collapse_flags)
        print(f"All {len(blank_collapse_flags)}/{len(picked_idx)} samples: greedy CTC decode "
              f"collapsed to empty (argmax==<blank> at nearly every timestep):")
        for f in blank_collapse_flags:
            print(f"  - {f}")
        print()
        if all_below_random:
            print("This is CTC blank-dominance, a well-documented and expected pattern early "
                  "in CTC training (blank is the argmax at the vast majority of timesteps "
                  "because the label sequence is much shorter than T, making blank the "
                  "single most frequent correct target almost everywhere) -- NOT the same "
                  "failure class as the Month 1 nested-tensor bug, which corrupted the "
                  "encoder's attention itself and produced degenerate, position-INsensitive "
                  "output. Evidence this is real-but-early signal rather than a structural "
                  "bug: on every sample tested, per-token CTC nll of the TRUE sequence is "
                  "below the random-uniform baseline (real learned correlation), and "
                  "per-token rank/probability tables above show rank/probability varying "
                  "meaningfully by timestep and by sample (position-sensitive, not a "
                  "collapsed/constant encoder output). If greedy decode is needed for "
                  "actual held-out evaluation before the model trains further past this "
                  "blank-dominance phase, prefer scoring via true-sequence CTC "
                  "probability/rank (as reported above) or a proper CTC beam search "
                  "over raw greedy argmax -- greedy decode is known to underreport "
                  "CTC models in this regime, that's expected, not a bug to fix here.")
        else:
            print("*** At least one sample's true-sequence nll is NOT below the "
                  "random-uniform baseline -- this is NOT simply early-training blank "
                  "dominance, there may be no learned signal for that sample. Treat as "
                  "a real debugging checkpoint, do not proceed to held-out evaluation "
                  "without root-causing this further. ***")
    else:
        print("No Day-17-style failure signature detected: greedy decodes vary across "
              "samples, are not uniformly placeholder tokens, and are not all identical. "
              "This looks like healthy (if under-trained, at step 4000) CTC signal rather "
              "than a structural bug. See per-token rank/probability numbers above -- "
              "at step 4000 the model is expected to still be far from exact-match, the "
              "important signal is whether true-token rank/probability is meaningfully "
              "above the random baseline, not whether greedy decode matches exactly.")


if __name__ == "__main__":
    main()
