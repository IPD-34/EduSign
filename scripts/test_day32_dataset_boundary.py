"""
scripts/test_day32_dataset_boundary.py

Confirms:
1. ASLLRP gloss lookup now resolves through asllrp_token_to_id.json,
   not raw .get() against vocab_merged.json -- checked by confirming
   real ASLLRP glosses do NOT all collapse to <UNK>.
2. boundary_target is present, correctly shaped, and has real
   nonzero content for an ASLLRP sample (not all-zero, which would
   mean build_boundary_targets silently did nothing).
3. collate_fn pads boundary_target aligned with landmarks/
   encoder_attention_mask, not target_tokens.
"""

import torch
from torch.utils.data import DataLoader

from src.models.dataset import SignLanguageDataset, collate_fn

ASLLRP_NPZ_DIR = "data/processed/asllrp/train"
VOCAB_MERGED_PATH = "data/vocab_merged.json"
ASLLRP_TOKEN_TO_ID_PATH = "data/asllrp_token_to_id.json"


def main():
    dataset = SignLanguageDataset(
        npz_dir=ASLLRP_NPZ_DIR,
        vocab_path=VOCAB_MERGED_PATH,
        asllrp_token_to_id_path=ASLLRP_TOKEN_TO_ID_PATH,
    )

    # --- Check 1: gloss lookup no longer collapses to <UNK> ---
    unk_id = dataset.unk_id
    sample_item = dataset[0]
    target_tokens = sample_item["target_tokens"]
    non_special = target_tokens[1:-1]  # strip BOS/EOS
    unk_count = (non_special == unk_id).sum().item()
    total = non_special.shape[0]
    print(f"Sample 0: {total} gloss tokens, {unk_count} resolved to <UNK> ({100*unk_count/total:.1f}%)")
    if unk_count == total:
        print("[WARNING] ALL tokens are <UNK> -- lookup may still be broken.")
    else:
        print("[OK] Not all tokens collapsed to <UNK>.")

    # --- Check 2: boundary_target is real, not all-zero ---
    boundary_target = sample_item["boundary_target"]
    T = sample_item["landmarks"].shape[0]
    assert boundary_target.shape[0] == T, f"boundary_target length {boundary_target.shape[0]} != T {T}"
    nonzero_count = (boundary_target > 0).sum().item()
    print(f"\nboundary_target shape: {boundary_target.shape}, nonzero positions: {nonzero_count}/{T}")
    if nonzero_count == 0:
        print("[WARNING] boundary_target is all-zero -- build_boundary_targets may not be firing.")
    else:
        print("[OK] boundary_target has real positive content.")

    # --- Check 3: batch-level padding alignment ---
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
    batch = next(iter(loader))
    print(f"\nBatch landmarks shape: {batch['landmarks'].shape}")
    print(f"Batch encoder_attention_mask shape: {batch['encoder_attention_mask'].shape}")
    print(f"Batch boundary_target shape: {batch['boundary_target'].shape}")
    assert batch["boundary_target"].shape == batch["encoder_attention_mask"].shape, \
        "boundary_target and encoder_attention_mask shapes must match -- both are per-encoder-timestep"
    print("[OK] boundary_target aligns with encoder_attention_mask shape, not target_tokens.")

    # Confirm padded positions in boundary_target are zero (never
    # incorrectly marked positive past the real sequence length)
    for i in range(batch["landmarks"].shape[0]):
        real_len = batch["num_frames"][i]
        padded_region = batch["boundary_target"][i, real_len:]
        assert torch.all(padded_region == 0), f"Sample {i}: padding region has nonzero boundary_target"
    print("[OK] Padded regions of boundary_target are all zero.")


if __name__ == "__main__":
    main()