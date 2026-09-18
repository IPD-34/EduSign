"""
scripts/train_asl_citizen_classifier.py

Quick, isolated-sign classifier validation for the ASL Citizen slice
(40 words, ASL Citizen's own signer-disjoint train/val/test split --
see asl_citizen_reader.py for confirmation those are non-overlapping
by Participant ID). NOT the full CTC+boundary+gate pipeline -- this is
a much simpler task (closed-set single-sign classification, one label
per clip), so it gets a much simpler model: the EXISTING pretrained
SignEncoder (reused, not retrained from scratch) + masked mean-pool +
one linear classification head, trained with plain cross-entropy.

Reuses feature_utils.flatten_frame_features and sample_writer.load_sample
directly rather than src.models.dataset.SignLanguageDataset -- that
class builds CTC targets (gloss-to-ID via the ASLLRP/merged-vocab
routing) and boundary targets neither of which this task needs or
wants; a minimal custom Dataset keeps this a true "simple classifier,"
not a repurposed sequence-model dataset.
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.models.encoder import SignEncoder
from src.models.feature_utils import flatten_frame_features
from src.preprocessing.sample_writer import load_sample
from src.models.train_utils import get_device

PROCESSED_DIR = Path("data/processed/asl_citizen_validation_v2")  # hand-detection-fixed extraction
RESUME_CHECKPOINT = "checkpoints/stageB1_real/control/best_by_wer.pt"  # current best-known checkpoint

WORDS = [
    'bathroom', 'bye', 'family', 'father', 'friend', 'he', 'hello', 'help', 'his', 'home',
    'hungry', 'hurt', 'me', 'medicine', 'mother', 'my', 'no', 'our', 'pain', 'please',
    'rest', 'school', 'sick', 'sleep', 'sorry', 'student', 'that', 'thirsty', 'tired', 'toilet',
    'water', 'we', 'when', 'where', 'who', 'why', 'work', 'yes', 'you', 'your',
]
WORD_TO_LABEL = {w: i for i, w in enumerate(WORDS)}

BATCH_SIZE = 16
LEARNING_RATE = 3e-4
NUM_EPOCHS = 30
HIDDEN_DIM = 256


class IsolatedSignDataset(Dataset):
    def __init__(self, npz_dir):
        self.paths = sorted(Path(npz_dir).glob("*.npz"))
        if not self.paths:
            raise RuntimeError(f"No .npz files found in {npz_dir}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        sample = load_sample(self.paths[idx])
        features = flatten_frame_features(sample)
        landmarks = torch.tensor(features, dtype=torch.float32)
        label = WORD_TO_LABEL[sample.metadata.gloss]
        return {"landmarks": landmarks, "label": label, "video_id": sample.metadata.video_id}


def collate_fn(batch):
    max_len = max(item["landmarks"].shape[0] for item in batch)
    feat_dim = batch[0]["landmarks"].shape[1]
    B = len(batch)
    landmarks = torch.zeros(B, max_len, feat_dim, dtype=torch.float32)
    mask = torch.zeros(B, max_len, dtype=torch.bool)
    labels = torch.zeros(B, dtype=torch.long)
    for i, item in enumerate(batch):
        T = item["landmarks"].shape[0]
        landmarks[i, :T] = item["landmarks"]
        mask[i, :T] = True
        labels[i] = item["label"]
    return {"landmarks": landmarks, "mask": mask, "labels": labels}


class SimpleSignClassifier(nn.Module):
    def __init__(self, num_classes, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.encoder = SignEncoder(hidden_dim=hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, landmarks, mask):
        encoded = self.encoder(landmarks, mask)  # (B, T, hidden_dim)
        mask_f = mask.unsqueeze(-1).float()
        pooled = (encoded * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)  # masked mean-pool
        return self.classifier(pooled)


def load_pretrained_encoder(encoder: nn.Module, checkpoint_path: str):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    old_state = checkpoint["model_state_dict"]
    new_state = encoder.state_dict()
    loaded = []
    for name in new_state:
        ckpt_key = f"encoder.{name}"
        if ckpt_key in old_state and old_state[ckpt_key].shape == new_state[name].shape:
            new_state[name] = old_state[ckpt_key]
            loaded.append(name)
    encoder.load_state_dict(new_state)
    print(f"[load_pretrained_encoder] Loaded {len(loaded)}/{len(new_state)} params from {checkpoint_path}")
    return loaded


@torch.no_grad()
def evaluate(model, loader, device, top_k=5):
    model.eval()
    correct1, correct5, total = 0, 0, 0
    for batch in loader:
        landmarks = batch["landmarks"].to(device)
        mask = batch["mask"].to(device)
        labels = batch["labels"].to(device)
        logits = model(landmarks, mask)
        top5 = logits.topk(min(top_k, logits.shape[-1]), dim=-1).indices
        correct1 += (top5[:, 0] == labels).sum().item()
        correct5 += (top5 == labels.unsqueeze(1)).any(dim=1).sum().item()
        total += labels.shape[0]
    model.train()
    return correct1 / max(total, 1), correct5 / max(total, 1), total


def main():
    device = get_device()

    train_ds = IsolatedSignDataset(PROCESSED_DIR / "train")
    val_ds = IsolatedSignDataset(PROCESSED_DIR / "val")
    test_ds = IsolatedSignDataset(PROCESSED_DIR / "test")
    print(f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} classes={len(WORDS)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = SimpleSignClassifier(num_classes=len(WORDS)).to(device)
    load_pretrained_encoder(model.encoder, RESUME_CHECKPOINT)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()

    history = []
    best_val_acc = -1.0
    best_state = None

    model.train()
    for epoch in range(1, NUM_EPOCHS + 1):
        total_loss, n_batches = 0.0, 0
        for batch in train_loader:
            landmarks = batch["landmarks"].to(device)
            mask = batch["mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(landmarks, mask)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        val_top1, val_top5, val_n = evaluate(model, val_loader, device)
        mean_loss = total_loss / max(n_batches, 1)
        history.append({"epoch": epoch, "train_loss": mean_loss, "val_top1": val_top1, "val_top5": val_top5})
        print(f"[epoch {epoch:3d}] train_loss={mean_loss:.4f}  val_top1={val_top1:.4f}  val_top5={val_top5:.4f} (n={val_n})")

        if val_top1 > best_val_acc:
            best_val_acc = val_top1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\nRestored best-val-checkpoint (val_top1={best_val_acc:.4f}) for final test evaluation.")

    test_top1, test_top5, test_n = evaluate(model, test_loader, device)
    print(f"\n=== FINAL, SIGNER-HELD-OUT TEST RESULT ===")
    print(f"test_top1={test_top1:.4f}  test_top5={test_top5:.4f}  (n={test_n}, {len(WORDS)}-way classification, chance={1/len(WORDS):.4f})")

    out = {
        "words": WORDS, "n_train": len(train_ds), "n_val": len(val_ds), "n_test": len(test_ds),
        "history": history, "best_val_top1": best_val_acc,
        "test_top1": test_top1, "test_top5": test_top5, "test_n": test_n,
        "chance_accuracy": 1 / len(WORDS),
    }
    with open("logs/asl_citizen_classifier_results_v2_handfix.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nResults saved to logs/asl_citizen_classifier_results_v2_handfix.json")


if __name__ == "__main__":
    main()
