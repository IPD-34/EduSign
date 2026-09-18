"""
scripts/signbart_style_prototype_40word.py

CHEAP VALIDATION, not a faithful reimplementation: tests the ONE core
claimed SignBart differentiator -- separate x/y coordinate encoding
streams fused via cross-attention, instead of this project's current
SignEncoder fusing x/y/z/visibility into one flat vector and passing it
through a single linear projection -- in isolation, on the already-
validated 40-word ASL Citizen point, to get a real go/no-go signal
before spending real engineering time on a full port.

SIMPLIFICATIONS, stated explicitly (this is a 1-2 hour prototype, not a
paper reproduction):
  - Only x/y are split into separate streams; z and pose-visibility are
    dropped entirely for this test (SignBart's exact treatment of
    depth/confidence was not independently confirmed before writing
    this -- the research agent covering SignBart is running in
    parallel with this prototype, not gating it, per "validate cheaply
    first").
  - Coordinate normalization: centers every point on the pose nose
    landmark (index 0) and scale-normalizes by inter-shoulder distance
    (pose landmarks 11,12) per frame -- a standard, well-established
    normalization pattern, NOT independently verified against
    SignBart's own published formula. Flagged, not hidden.
  - Cross-attention fusion: x-stream and y-stream each project to
    hidden_dim, cross-attend to each other (x queries y, y queries x),
    concat, project back to hidden_dim -- then feeds into the SAME
    downstream temporal transformer (4 layers, same hidden_dim) the
    baseline SignEncoder uses, so the comparison isolates the input
    fusion mechanism, not confounded by a deeper/different temporal
    stack.
  - CANNOT load the pretrained encoder checkpoint (different input
    projection shape/mechanism) -- trains from scratch. This is a REAL
    asymmetry vs. the baseline's pretrained-encoder head start, not
    papered over: if this prototype's number is in the same range as
    the from-scratch portion of the baseline's learning curve, that's
    still a meaningful signal; if it's dramatically worse, some of that
    gap is legitimately attributable to missing pretraining, not
    necessarily the architecture itself.

Same locked setup otherwise: same 40-word ASL Citizen data, same 30
epochs, same batch size/optimizer/LR, same max-over-time CTC eval.
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.models.positional_encoding import PositionalEncoding
from src.models.train_utils import get_device
from src.preprocessing.sample_writer import load_sample

from scripts.asl_citizen_vocab_scaling import (
    load_gloss_ranking, ctc_loss_for_batch, PROCESSED_DIR,
    BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, NUM_EPOCHS, HIDDEN_DIM,
    GRAD_CLIP_MAX_NORM, TOP_K_FOR_TOP5, peek_gloss_split,
)

N_POINTS = 33 + 478 + 21 + 21  # pose+face+left+right, x/y only per point = 553

NOSE_IDX = 0        # pose landmark 0
L_SHOULDER_IDX = 11  # pose landmark 11
R_SHOULDER_IDX = 12  # pose landmark 12


def extract_xy_streams(sample):
    """Returns (x_feat, y_feat), each (T, 553) float32, after nose-centering
    and shoulder-width scale normalization applied to x and y jointly
    (using pose's own x/y before splitting) so both streams share one
    consistent, real body-relative scale rather than being normalized
    independently."""
    pose = sample.pose      # (T,33,4): x,y,z,vis
    face = sample.face      # (T,478,3)
    lh = sample.left_hand   # (T,21,3)
    rh = sample.right_hand  # (T,21,3)
    T = pose.shape[0]

    cx = pose[:, NOSE_IDX, 0:1]  # (T,1)
    cy = pose[:, NOSE_IDX, 1:2]
    shoulder_dx = pose[:, L_SHOULDER_IDX, 0] - pose[:, R_SHOULDER_IDX, 0]
    shoulder_dy = pose[:, L_SHOULDER_IDX, 1] - pose[:, R_SHOULDER_IDX, 1]
    scale = np.sqrt(shoulder_dx**2 + shoulder_dy**2).reshape(T, 1)
    scale = np.where(scale < 1e-6, 1.0, scale)  # guard degenerate/missing-pose frames

    def norm_xy(arr):
        x = (arr[..., 0] - cx) / scale
        y = (arr[..., 1] - cy) / scale
        return x, y

    px, py = norm_xy(pose)
    fx, fy = norm_xy(face)
    lx, ly = norm_xy(lh)
    rx, ry = norm_xy(rh)

    x_feat = np.concatenate([px, fx, lx, rx], axis=1).astype(np.float32)  # (T,553)
    y_feat = np.concatenate([py, fy, ly, ry], axis=1).astype(np.float32)
    return x_feat, y_feat


class IsolatedSignDatasetXY(Dataset):
    def __init__(self, npz_dir, word_to_label, split_name):
        all_paths = sorted(Path(npz_dir).glob("*.npz"))
        self.paths = []
        for p in all_paths:
            gloss, split = peek_gloss_split(p)
            if gloss in word_to_label and split == split_name:
                self.paths.append(p)
        self.word_to_label = word_to_label

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        sample = load_sample(self.paths[idx])
        x_feat, y_feat = extract_xy_streams(sample)
        label = self.word_to_label[sample.metadata.gloss]
        return {"x": torch.tensor(x_feat), "y": torch.tensor(y_feat), "label": label}


def collate_fn_xy(batch):
    max_len = max(item["x"].shape[0] for item in batch)
    B = len(batch)
    x = torch.zeros(B, max_len, N_POINTS, dtype=torch.float32)
    y = torch.zeros(B, max_len, N_POINTS, dtype=torch.float32)
    mask = torch.zeros(B, max_len, dtype=torch.bool)
    labels = torch.zeros(B, dtype=torch.long)
    lengths = torch.zeros(B, dtype=torch.long)
    for i, item in enumerate(batch):
        T = item["x"].shape[0]
        x[i, :T] = item["x"]
        y[i, :T] = item["y"]
        mask[i, :T] = True
        labels[i] = item["label"]
        lengths[i] = T
    return {"x": x, "y": y, "mask": mask, "labels": labels, "lengths": lengths}


class SignEncoderXY(nn.Module):
    """SignBart-style differentiator, isolated: separate x/y projections,
    cross-attention fusion, THEN the same temporal transformer stack the
    baseline SignEncoder uses (4 layers, same hidden_dim) -- so any
    accuracy delta vs. the baseline is attributable to the input fusion
    mechanism, not a deeper/different backbone."""
    def __init__(self, hidden_dim=HIDDEN_DIM, num_layers=4, num_heads=8, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.x_proj = nn.Linear(N_POINTS, hidden_dim)
        self.y_proj = nn.Linear(N_POINTS, hidden_dim)
        self.cross_attn_x = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True, dropout=dropout)
        self.cross_attn_y = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True, dropout=dropout)
        self.combine = nn.Linear(hidden_dim * 2, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim, max_len=1000, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)
        self.hidden_dim = hidden_dim

    def forward(self, x_raw, y_raw, mask):
        x = self.x_proj(x_raw)
        y = self.y_proj(y_raw)
        key_padding_mask = ~mask.bool()
        x_cross, _ = self.cross_attn_x(query=x, key=y, value=y, key_padding_mask=key_padding_mask)
        y_cross, _ = self.cross_attn_y(query=y, key=x, value=x, key_padding_mask=key_padding_mask)
        fused = self.combine(torch.cat([x_cross, y_cross], dim=-1))
        fused = self.positional_encoding(fused)
        encoded = self.transformer_encoder(fused, src_key_padding_mask=key_padding_mask)
        return encoded


class CTCIsolatedSignModelXY(nn.Module):
    def __init__(self, vocab_size, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.encoder = SignEncoderXY(hidden_dim=hidden_dim)
        self.ctc_head = nn.Linear(hidden_dim, vocab_size + 1)
        self.blank_id = vocab_size

    def forward(self, x, y, mask):
        encoded = self.encoder(x, y, mask)
        logits = self.ctc_head(encoded)
        return torch.log_softmax(logits, dim=-1)


@torch.no_grad()
def evaluate_xy(model, loader, device, k=TOP_K_FOR_TOP5):
    model.eval()
    correct1, correctk, total = 0, 0, 0
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        mask = batch["mask"].to(device)
        labels = batch["labels"].to(device)
        log_probs = model(x, y, mask)
        class_scores = log_probs[..., :-1].max(dim=1).values
        topk = class_scores.topk(min(k, class_scores.shape[-1]), dim=-1).indices
        correct1 += (topk[:, 0] == labels).sum().item()
        correctk += (topk == labels.unsqueeze(1)).any(dim=1).sum().item()
        total += labels.shape[0]
    model.train()
    return correct1 / max(total, 1), correctk / max(total, 1), total


def main():
    device = get_device()
    print(f"Device: {device}")
    ranked, train_c, val_c, test_c = load_gloss_ranking()
    k = 40
    word_list = ranked[:k]
    word_to_label = {w: i for i, w in enumerate(sorted(word_list))}

    train_ds = IsolatedSignDatasetXY(PROCESSED_DIR / "train", word_to_label, "train")
    val_ds = IsolatedSignDatasetXY(PROCESSED_DIR / "val", word_to_label, "val")
    test_ds = IsolatedSignDatasetXY(PROCESSED_DIR / "test", word_to_label, "test")
    print(f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} classes={k}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_xy)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_xy)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_xy)

    torch.manual_seed(42)
    model = CTCIsolatedSignModelXY(vocab_size=k).to(device)
    print("NOTE: training from scratch -- no pretrained encoder loaded (different input mechanism, incompatible shapes)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    history = []
    best_val_top1 = -1.0
    best_state = None

    model.train()
    for epoch in range(1, NUM_EPOCHS + 1):
        total_loss, n_batches = 0.0, 0
        for batch in train_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            mask = batch["mask"].to(device)
            labels = batch["labels"].to(device)
            lengths = batch["lengths"].to(device)

            log_probs = model(x, y, mask)
            loss = ctc_loss_for_batch(log_probs, labels, lengths, model.blank_id)
            if not torch.isfinite(loss):
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_MAX_NORM)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        val_top1, val_top5, val_n = evaluate_xy(model, val_loader, device)
        mean_loss = total_loss / max(n_batches, 1)
        history.append({"epoch": epoch, "train_loss": mean_loss, "val_top1": val_top1, "val_top5": val_top5})
        if epoch % 5 == 0 or epoch == 1:
            print(f"[epoch {epoch:3d}] train_loss={mean_loss:.4f}  val_top1={val_top1:.4f}  val_top5={val_top5:.4f} (n={val_n})")

        if val_top1 > best_val_top1:
            best_val_top1 = val_top1
            best_state = {kk: v.clone() for kk, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    test_top1, test_top5, test_n = evaluate_xy(model, test_loader, device)
    val_test_gap = best_val_top1 - test_top1
    print(f"\n=== SIGNBART-STYLE XY PROTOTYPE, 40 WORDS: FINAL TEST RESULT ===")
    print(f"best_val_top1={best_val_top1:.4f}  test_top1={test_top1:.4f}  test_top5={test_top5:.4f}  "
          f"val_test_gap={val_test_gap:+.4f}  (n_test={test_n}, chance={1/k:.4f})")

    result = {
        "vocab_size": k, "n_train": len(train_ds), "n_val": len(val_ds), "n_test": len(test_ds),
        "history": history, "best_val_top1": best_val_top1,
        "test_top1": test_top1, "test_top5": test_top5, "test_n": test_n,
        "val_test_gap": val_test_gap, "pretrained": False,
    }
    with open("logs/signbart_style_prototype_40word_result.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\nSaved to logs/signbart_style_prototype_40word_result.json")

    print("\n=== COMPARISON vs baseline SignEncoder (same 40-word locked setup) ===")
    print("baseline (pretrained, 30ep): test_top1=0.4304  test_top5=0.7929")
    print(f"XY-crossattn (scratch, 30ep): test_top1={test_top1:.4f}  test_top5={test_top5:.4f}")


if __name__ == "__main__":
    main()
