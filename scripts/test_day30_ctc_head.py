"""
test_day30_ctc_head.py

Day 30 verification: 
1. Instantiate SignTransformer with the new merged CTC vocab.
2. Partially reload the Day 12 WLASL checkpoint into it (encoder only,
   per the decision below).
3. Confirm the loaded/skipped/new_only breakdown matches expectations
   by eye -- not just "did it run without crashing."
4. Run one real forward_ctc() pass on a real ASLLRP batch, confirm
   output shape and that log_softmax produces finite values.

NOTE: target_tokens produced by SignLanguageDataset in this script are
NOT reliable yet -- that dataset does raw-string vocab lookup
(sample.metadata.gloss.split("|") -> token_to_id.get()), which bypasses
Day 29's asllrp_token_to_id.json normalization (case-fold, suffix-strip,
fingerspelling/classifier routing). Fine for today's check, since Day 30
only needs real landmarks through forward_ctc() to confirm shape and
finiteness -- targets aren't used below at all. This MUST be fixed
before Day 32's real CTC training loop, or almost every ASLLRP target
will silently resolve to <UNK>. Flagged here so it isn't forgotten
between now and then.
"""

import json
import torch
from torch.utils.data import DataLoader

from src.models.sign_transformer import SignTransformer
from src.models.train_utils import load_checkpoint_partial, get_device
from src.models.dataset import SignLanguageDataset, collate_fn


VOCAB_MERGED_PATH = "data/vocab_merged.json"
DECODER_VOCAB_PATH = "data/vocab.json"  # 1850 entries -- old WLASL decoder vocab
OLD_CHECKPOINT_PATH = "checkpoints/day12_latest.pt"
ASLLRP_NPZ_DIR = "data/processed/asllrp/train" # confirm this is the corrected,
                                            # post-cap-fix directory -- NOT
                                            # asllrp_OLD_150cap_bug or _v2


def main():
    device = get_device()

    with open(VOCAB_MERGED_PATH) as f:
        vocab_merged = json.load(f)
    with open(DECODER_VOCAB_PATH) as f:
        decoder_vocab = json.load(f)
    ctc_vocab_size = len(vocab_merged)
    print(f"Merged vocab size: {ctc_vocab_size}")

    # Model: old decoder vocab (1850, WLASL) kept alive alongside the
    # new CTC head (ctc_vocab_size + 1, blank included).
    model = SignTransformer(
        vocab_size=len(decoder_vocab),
        ctc_vocab_size=ctc_vocab_size,
    ).to(device)

    # --- Decision, stated explicitly, not left to the function default ---
    # The decoder is deprioritized now that CTC is the real recognition
    # path for ASLLRP going forward; reinitializing it rather than
    # warm-starting is acceptable since it isn't part of C2's claim.
    # If this changes later, this is the one line to edit.
    loaded, skipped, new_only = load_checkpoint_partial(
        model, OLD_CHECKPOINT_PATH, prefixes_to_load=("encoder.",)
    )

    # Manual sanity check, not just "did it run": ctc_head.* and
    # boundary_head.* must be the only things in new_only. If anything
    # else shows up here, something in the architecture changed since
    # the checkpoint was saved and needs to be understood before
    # proceeding. (boundary_head added to SignTransformer unconditionally
    # after this test was originally written -- this check was stale
    # for that reason, unrelated to vocab_size/vocab.json.)
    unexpected_new = [
        n for n in new_only
        if not (n.startswith("ctc_head.") or n.startswith("boundary_head."))
    ]
    assert not unexpected_new, f"Unexpected new-only params: {unexpected_new}"
    print("\n[OK] new_only contains exactly the expected ctc_head/boundary_head params.")

    # --- Real forward pass on a real ASLLRP batch ---
    dataset = SignLanguageDataset(
        npz_dir=ASLLRP_NPZ_DIR,
        vocab_path=VOCAB_MERGED_PATH,
    )
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
    batch = next(iter(loader))
    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

    log_probs = model.forward_ctc(batch["landmarks"], batch["encoder_attention_mask"])
    print(f"\nCTC log_probs shape: {log_probs.shape}")  # expect (B, T, ctc_vocab_size + 1)
    assert log_probs.shape[-1] == ctc_vocab_size + 1
    assert torch.isfinite(log_probs).all(), "Non-finite values in CTC log_probs!"
    print("[OK] forward_ctc() output shape and finiteness confirmed.")


if __name__ == "__main__":
    main()