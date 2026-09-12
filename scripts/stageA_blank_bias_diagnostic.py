"""
scripts/stageA_blank_bias_diagnostic.py

Diagnoses WHY blank_logit_bias=3.0 moved blank_argmax_rate the wrong way
(0.9555 -> 0.9782 on the fixed 25-sample probe set) in Stage A's smoke
test, using the seeded rerun's saved checkpoints (control vs blank_alone,
step 500, checkpoints/stageA_smoke/{control,blank_alone}/step_0500.pt --
train_seed=55, identical batch order for both configs).

Read-only diagnostic. Loads both checkpoints, compares them on the SAME
25-sample probe subset (seed=54). No training, nothing modified.

Three checks, matching the three questions asked:
  1. Per-timestep: where control and blank_alone disagree on argmax,
     what did blank_alone move TO -- the true gloss, <UNK>, some other
     arbitrary token, or nothing confident at all (diffuse)?
  2. Raw ctc_log_probs comparison at inference time (bias is NEVER
     reapplied here -- both models are called via the plain
     forward_ctc_boundary_and_gate(), so this is each model's own
     GENUINE, unbiased, learned P(blank), not an artifact of the
     training-time loss reweighting): did blank_alone's own learned
     P(blank) actually go UP relative to control (compensating /
     "fighting back" against the bias), or is blank still similarly
     dominant in both and the bias just wasn't strong enough to move it?
  3. Pathology check: did the FULL distribution's entropy increase
     (model getting vaguer everywhere, not just blank-preferring), and
     did encoder hidden-state variance across timesteps drop (less
     temporally-discriminative representations)?
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

from src.models.sign_transformer import SignTransformer
from src.models.train_utils import load_checkpoint_partial, get_device
from src.models.dataset import SignLanguageDataset
from src.preprocessing.sample_writer import load_sample

VOCAB_MERGED_PATH = "data/vocab_merged.json"
DECODER_VOCAB_PATH = "data/vocab.json"
ASLLRP_TOKEN_TO_ID_PATH = "data/asllrp_token_to_id.json"
TEST_NPZ_DIR = "data/processed/asllrp/test"

CONTROL_CKPT = "checkpoints/stageA_smoke/control/step_0500.pt"
BLANK_ALONE_CKPT = "checkpoints/stageA_smoke/blank_alone/step_0500.pt"

SCREEN_SEED = 54
SCREEN_SIZE = 25


def load_model(ckpt_path, ctc_vocab_size, decoder_vocab_size, device):
    model = SignTransformer(
        vocab_size=decoder_vocab_size, ctc_vocab_size=ctc_vocab_size, use_bimanual_gate=True,
    ).to(device)
    load_checkpoint_partial(
        model, ckpt_path, prefixes_to_load=("encoder.", "ctc_head.", "boundary_head.", "bimanual_gate_head."),
    )
    model.eval()
    return model


def main():
    import random
    device = get_device()
    with open(VOCAB_MERGED_PATH) as f:
        vocab_merged = json.load(f)
    with open(DECODER_VOCAB_PATH) as f:
        decoder_vocab = json.load(f)
    id_to_token = {v: k for k, v in vocab_merged.items()}
    ctc_vocab_size = len(vocab_merged)
    blank_id = ctc_vocab_size
    unk_id = vocab_merged.get("<UNK>")

    dataset = SignLanguageDataset(
        npz_dir=TEST_NPZ_DIR, vocab_path=VOCAB_MERGED_PATH, asllrp_token_to_id_path=ASLLRP_TOKEN_TO_ID_PATH,
    )
    n = len(dataset)
    rng = random.Random(SCREEN_SEED)
    probe_indices = sorted(rng.sample(range(n), min(SCREEN_SIZE, n)))
    print(f"Probe subset: {len(probe_indices)} samples (seed={SCREEN_SEED}, same as Stage A)")

    print(f"\nLoading control:     {CONTROL_CKPT}")
    control = load_model(CONTROL_CKPT, ctc_vocab_size, len(decoder_vocab), device)
    print(f"Loading blank_alone: {BLANK_ALONE_CKPT}")
    blank_alone = load_model(BLANK_ALONE_CKPT, ctc_vocab_size, len(decoder_vocab), device)

    # =====================================================================
    # Item 1 + 2: per-timestep comparison, and raw (unbiased) P(blank)
    # =====================================================================
    total_timesteps = 0
    disagree_timesteps = 0
    cat_blank_to_content = 0
    cat_content_to_blank = 0
    cat_content_to_content = 0
    sub_true_gloss = 0
    sub_unk = 0
    sub_other_arbitrary = 0
    sub_diffuse = 0  # new prediction's own probability is low (<0.3): not confidently anything

    control_blank_probs_all = []
    blank_alone_blank_probs_all = []
    control_full_entropy_all = []
    blank_alone_full_entropy_all = []
    control_hidden_var_all = []
    blank_alone_hidden_var_all = []

    example_disagreements = []  # a handful for the report, item 1/2 illustration

    for idx in probe_indices:
        item = dataset[idx]
        meta = load_sample(dataset.npz_paths[idx]).metadata
        landmarks = item["landmarks"].unsqueeze(0).to(device)
        T = landmarks.shape[1]
        mask = torch.ones(1, T, dtype=torch.bool, device=device)
        true_tokens = item["target_tokens"].tolist()
        true_ids = set(true_tokens[1:-1])

        with torch.no_grad():
            c_logp, _c_bl, _c_gl = control.forward_ctc_boundary_and_gate(landmarks, mask)
            b_logp, _b_bl, _b_gl = blank_alone.forward_ctc_boundary_and_gate(landmarks, mask)
            # Hidden-state variance: reuse the shared encoder forward directly
            c_hidden = control.encoder(landmarks, mask)
            b_hidden = blank_alone.encoder(landmarks, mask)

        c_logp, b_logp = c_logp[0], b_logp[0]  # (T, V+1)
        c_probs, b_probs = c_logp.exp(), b_logp.exp()
        c_argmax, b_argmax = c_logp.argmax(-1), b_logp.argmax(-1)

        control_blank_probs_all.extend(c_probs[:, blank_id].tolist())
        blank_alone_blank_probs_all.extend(b_probs[:, blank_id].tolist())

        c_entropy = -(c_probs * (c_probs + 1e-12).log()).sum(-1)
        b_entropy = -(b_probs * (b_probs + 1e-12).log()).sum(-1)
        control_full_entropy_all.extend(c_entropy.tolist())
        blank_alone_full_entropy_all.extend(b_entropy.tolist())

        # per-timestep variance of the hidden vector, then averaged across T for this sample
        control_hidden_var_all.append(c_hidden[0].var(dim=0).mean().item())
        blank_alone_hidden_var_all.append(b_hidden[0].var(dim=0).mean().item())

        total_timesteps += T
        for t in range(T):
            if c_argmax[t].item() == b_argmax[t].item():
                continue
            disagree_timesteps += 1
            old_id, new_id = c_argmax[t].item(), b_argmax[t].item()
            old_is_blank, new_is_blank = old_id == blank_id, new_id == blank_id

            if old_is_blank and not new_is_blank:
                cat_blank_to_content += 1
                new_prob = b_probs[t, new_id].item()
                if new_id in true_ids:
                    sub_true_gloss += 1
                elif unk_id is not None and new_id == unk_id:
                    sub_unk += 1
                elif new_prob < 0.3:
                    sub_diffuse += 1
                else:
                    sub_other_arbitrary += 1
                if len(example_disagreements) < 8:
                    example_disagreements.append({
                        "video_id": meta.video_id, "t": t, "direction": "blank->content",
                        "old_token": "<blank>", "new_token": id_to_token.get(new_id, f"id{new_id}"),
                        "new_prob": round(new_prob, 4),
                        "new_is_true_gloss": new_id in true_ids,
                        "control_p_blank": round(c_probs[t, blank_id].item(), 4),
                        "blank_alone_p_blank": round(b_probs[t, blank_id].item(), 4),
                    })
            elif not old_is_blank and new_is_blank:
                cat_content_to_blank += 1
                if len(example_disagreements) < 8:
                    example_disagreements.append({
                        "video_id": meta.video_id, "t": t, "direction": "content->blank",
                        "old_token": id_to_token.get(old_id, f"id{old_id}"), "new_token": "<blank>",
                        "old_was_true_gloss": old_id in true_ids,
                        "control_p_blank": round(c_probs[t, blank_id].item(), 4),
                        "blank_alone_p_blank": round(b_probs[t, blank_id].item(), 4),
                    })
            else:
                cat_content_to_content += 1

    print("\n" + "=" * 90)
    print("ITEM 1: per-timestep disagreement breakdown (control vs blank_alone, step 500)")
    print("=" * 90)
    print(f"Total timesteps checked: {total_timesteps}")
    print(f"Timesteps where argmax DIFFERS: {disagree_timesteps} ({100*disagree_timesteps/total_timesteps:.2f}%)")
    print(f"  blank -> content   : {cat_blank_to_content}")
    print(f"    -> true gloss token       : {sub_true_gloss}")
    print(f"    -> <UNK>                  : {sub_unk}")
    print(f"    -> other arbitrary token  : {sub_other_arbitrary}")
    print(f"    -> diffuse (new pred prob<0.3, not confidently anything) : {sub_diffuse}")
    print(f"  content -> blank   : {cat_content_to_blank}   <-- this direction is what would explain blank_rate rising")
    print(f"  content -> content (different non-blank token): {cat_content_to_content}")

    print("\nExample disagreements (up to 8, first-found):")
    for e in example_disagreements:
        print(f"  {e}")

    print("\n" + "=" * 90)
    print("ITEM 2: raw (unbiased, eval-time) P(blank) -- both models called with NO bias reapplied")
    print("=" * 90)
    mean_c_blank = sum(control_blank_probs_all) / len(control_blank_probs_all)
    mean_b_blank = sum(blank_alone_blank_probs_all) / len(blank_alone_blank_probs_all)
    print(f"Mean P(blank) across ALL {total_timesteps} timesteps (not just argmax=blank ones):")
    print(f"  control     : {mean_c_blank:.4f}")
    print(f"  blank_alone : {mean_b_blank:.4f}")
    print(f"  Diff (blank_alone - control): {mean_b_blank - mean_c_blank:+.4f}")
    if mean_b_blank > mean_c_blank:
        print("  => blank_alone's OWN genuine (unbiased) preference for blank is HIGHER than control's.")
        print("     This is consistent with training 'fighting back' against the bias, not a bias-too-weak story.")
    else:
        print("  => blank_alone's genuine preference for blank is similar to or lower than control's.")

    print("\n" + "=" * 90)
    print("ITEM 3: pathology check -- full-distribution entropy, encoder hidden-state variance")
    print("=" * 90)
    mean_c_ent = sum(control_full_entropy_all) / len(control_full_entropy_all)
    mean_b_ent = sum(blank_alone_full_entropy_all) / len(blank_alone_full_entropy_all)
    print(f"Mean FULL-distribution entropy (nats) per timestep, over {ctc_vocab_size+1} classes:")
    print(f"  control     : {mean_c_ent:.4f}")
    print(f"  blank_alone : {mean_b_ent:.4f}")
    print(f"  Diff: {mean_b_ent - mean_c_ent:+.4f}  "
          f"({'blank_alone MORE diffuse overall' if mean_b_ent > mean_c_ent else 'blank_alone NOT more diffuse -- similar or sharper'})")

    mean_c_var = sum(control_hidden_var_all) / len(control_hidden_var_all)
    mean_b_var = sum(blank_alone_hidden_var_all) / len(blank_alone_hidden_var_all)
    print(f"\nMean per-timestep encoder hidden-state variance (averaged over hidden_dim, then over samples):")
    print(f"  control     : {mean_c_var:.6f}")
    print(f"  blank_alone : {mean_b_var:.6f}")
    print(f"  Diff: {mean_b_var - mean_c_var:+.6f}  "
          f"({'blank_alone LESS temporally-discriminative (lower variance)' if mean_b_var < mean_c_var else 'blank_alone NOT less discriminative -- similar or higher variance'})")

    out = {
        "total_timesteps": total_timesteps, "disagree_timesteps": disagree_timesteps,
        "cat_blank_to_content": cat_blank_to_content, "cat_content_to_blank": cat_content_to_blank,
        "cat_content_to_content": cat_content_to_content,
        "sub_true_gloss": sub_true_gloss, "sub_unk": sub_unk,
        "sub_other_arbitrary": sub_other_arbitrary, "sub_diffuse": sub_diffuse,
        "mean_p_blank_control": mean_c_blank, "mean_p_blank_blank_alone": mean_b_blank,
        "mean_full_entropy_control": mean_c_ent, "mean_full_entropy_blank_alone": mean_b_ent,
        "mean_hidden_var_control": mean_c_var, "mean_hidden_var_blank_alone": mean_b_var,
        "example_disagreements": example_disagreements,
    }
    with open("logs/stageA_blank_bias_diagnostic.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nFull results saved to logs/stageA_blank_bias_diagnostic.json")


if __name__ == "__main__":
    main()
