"""
scripts/asl_citizen_vocab_scaling.py

Vocabulary-scaling experiment for the closed-vocab demo (Track 1): does
accuracy degrade as the ASL Citizen vocabulary grows from 40 to 100 to
200 (to 400) words, and if so, is that degradation genuine class-count
difficulty or just data starvation? Deliberately does NOT reuse the
bare linear-probe classifier from the earlier 40-word test (34.1%
top-1) -- that was explicitly caveated as unrepresentative of a real
ceiling. This uses the project's ACTUAL CTC-head architecture instead:
nn.CTCLoss (the same class combined_loss.py uses for real training),
with a length-1 target per isolated-sign clip -- not a bespoke
classification head. This is a real, if unusual, use of CTC: a clip
with one true class c has a target sequence of length 1 ([c]), and
the CTC forward-backward algorithm marginalizes over every alignment
of that one symbol across all T encoder timesteps, exactly the same
mechanism a continuous multi-gloss utterance uses, just with a
length-1 target instead of length-N.

EVAL METRIC, CHOSEN AND JUSTIFIED (not the bare linear-probe's plain
softmax-over-pooled-features): max-over-time pooling of the CTC head's
own per-timestep class log-probabilities -- for each candidate class c
(excluding blank), score(c) = max_t log_softmax(ctc_logits)[t, c].
This is a standard, established technique for isolated-word/keyword
CTC scoring (not invented for this task), built entirely from the same
CTC head a continuous decode would use -- not a separate probe. Top-1/
top-5 come from ranking classes by this score.

LOCKED EXPERIMENTAL SETUP (Step 2) -- stated explicitly before any
run, vocabulary size (== number of classes == which top-K ASL Citizen
glosses are included) is the ONLY variable that changes across runs:
  - Architecture: SignEncoder (pretrained, loaded from
    checkpoints/stageB1_real/control/best_by_wer.pt's "encoder."
    weights -- same checkpoint every prior experiment in this project
    has used) + a FRESH nn.Linear(hidden_dim, vocab_size+1) CTC head
    (necessarily fresh per run since vocab_size differs -- the old
    ctc_head's shape cannot be reused across different K).
  - Data-per-class handling: ALL real examples for each included class
    are used, uncapped, unaugmented -- no per-class subsampling to
    force equal counts (ASL Citizen's own near-uniform ~15-24
    examples/class per Step 1's real measurement means this doesn't
    need forcing anyway).
  - Splits: ASL Citizen's own official signer-disjoint train/val/test
    CSVs (already confirmed non-overlapping by Participant ID in the
    earlier ASL Citizen task) -- restricted to whichever top-K word
    list a given run uses, nothing else changed about the split logic.
  - Epochs: 30, FIXED across every run regardless of vocab size --
    deliberately NOT scaled up for larger K, so epoch count is not a
    second confounded variable. Flagged as an explicit interpretive
    caveat: if a larger-K run looks under-trained relative to a
    smaller-K run, that is a real possible confound worth naming
    separately from genuine "harder because more classes" difficulty.
  - Augmentation: OFF for every run (matching this project's own
    Stage B1 finding that augmentation did not help on real data at
    scale -- consistent with the standing default, not a new choice
    made for this task).
  - Batch size, optimizer, LR, weight_decay, grad-clip: identical
    across every run, values below, unchanged from the earlier ASL
    Citizen classifier script.
  - Extraction pipeline: the ALREADY hand-detection-fixed pipeline
    (extractor.py/hand_detector.py's pose-anchored crop+upscale fix,
    validated and confirmed improved on this exact data source in the
    prior ASL Citizen task) -- reused unmodified, not re-derived.
"""
import sys
import json
import random
import csv
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.models.encoder import SignEncoder
from src.models.feature_utils import flatten_frame_features
from src.preprocessing.sample_writer import load_sample
from src.models.train_utils import get_device


def peek_gloss_split(path):
    """Read ONLY the metadata member of a .npz sample, not the full
    landmark arrays. load_sample() decompresses pose/face/both hands
    (np.savez_compressed -> real zlib cost) just to expose .metadata --
    fine for one file at train time, wasteful when done for every file
    in a growing multi-thousand-file directory purely to filter by
    gloss (paid 3x per vocab size x 3 vocab sizes = 9x over). This is
    only used for the initial filter pass; actual training still reads
    the full sample once per epoch via load_sample() in __getitem__."""
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["metadata"]))
    return meta["gloss"], meta["split"]

SPLITS_DIR = Path(
    r"C:\Users\KRUPAS~1\AppData\Local\Temp\claude\D--College-Projects---Resume-bidirectional-sign-language"
    r"\1f4ac0b6-58be-4111-abc9-012b28f39d49\scratchpad\asl_citizen\ASL_Citizen\splits"
)
PROCESSED_DIR = Path("data/processed/asl_citizen_scaling")  # written by the extraction step
RESUME_CHECKPOINT = "checkpoints/stageB1_real/control/best_by_wer.pt"

# --- LOCKED across every run ---
BATCH_SIZE = 16
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 30
HIDDEN_DIM = 256
GRAD_CLIP_MAX_NORM = 1.0
TOP_K_FOR_TOP5 = 5

VOCAB_SIZES = [40, 100, 200]  # 400 appended separately once/if that data is extracted


def load_gloss_ranking():
    """FIXED, deterministic word list -- loaded, not recomputed. An
    earlier version of this ranking (independently recomputed via
    Python's set() iteration, which is hash-randomization-dependent)
    disagreed with itself by 20/200 words across separate runs, since
    ASL Citizen's per-word counts cluster too tightly for most
    pairwise comparisons in the sort key to have a unique winner. This
    file is the single, alphabetically-tie-broken, deterministic
    ranking of exactly the 200 words that were actually downloaded and
    extracted -- see process_asl_citizen_scaling_parallel.py's own
    comment for the full diagnosis."""
    with open("logs/asl_citizen_scaling_word_list_FIXED.json") as f:
        ranked = json.load(f)

    def load_csv(path):
        with open(path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    train = load_csv(SPLITS_DIR / "train.csv")
    val = load_csv(SPLITS_DIR / "val.csv")
    test = load_csv(SPLITS_DIR / "test.csv")
    train_c = Counter(r["Gloss"].strip().lower() for r in train)
    val_c = Counter(r["Gloss"].strip().lower() for r in val)
    test_c = Counter(r["Gloss"].strip().lower() for r in test)
    return ranked, train_c, val_c, test_c


class IsolatedSignDataset(Dataset):
    def __init__(self, npz_dir, word_to_label, split_name):
        all_paths = sorted(Path(npz_dir).glob("*.npz"))
        self.paths = []
        self.glosses = []  # parallel to self.paths, cached from the peek below
        for p in all_paths:
            gloss, split = peek_gloss_split(p)  # metadata-only read, not a full load_sample()
            if gloss in word_to_label and split == split_name:
                self.paths.append(p)
                self.glosses.append(gloss)
        self.word_to_label = word_to_label

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        sample = load_sample(self.paths[idx])
        features = flatten_frame_features(sample)
        landmarks = torch.tensor(features, dtype=torch.float32)
        label = self.word_to_label[sample.metadata.gloss]
        return {"landmarks": landmarks, "label": label}


def collate_fn(batch):
    max_len = max(item["landmarks"].shape[0] for item in batch)
    feat_dim = batch[0]["landmarks"].shape[1]
    B = len(batch)
    landmarks = torch.zeros(B, max_len, feat_dim, dtype=torch.float32)
    mask = torch.zeros(B, max_len, dtype=torch.bool)
    labels = torch.zeros(B, dtype=torch.long)
    lengths = torch.zeros(B, dtype=torch.long)
    for i, item in enumerate(batch):
        T = item["landmarks"].shape[0]
        landmarks[i, :T] = item["landmarks"]
        mask[i, :T] = True
        labels[i] = item["label"]
        lengths[i] = T
    return {"landmarks": landmarks, "mask": mask, "labels": labels, "lengths": lengths}


class CTCIsolatedSignModel(nn.Module):
    """Real CTC-head architecture, not a linear probe -- see module docstring."""
    def __init__(self, vocab_size, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.encoder = SignEncoder(hidden_dim=hidden_dim)
        self.ctc_head = nn.Linear(hidden_dim, vocab_size + 1)  # +1 for blank
        self.blank_id = vocab_size

    def forward(self, landmarks, mask):
        encoded = self.encoder(landmarks, mask)  # (B, T, hidden_dim)
        logits = self.ctc_head(encoded)  # (B, T, vocab_size+1)
        log_probs = torch.log_softmax(logits, dim=-1)
        return log_probs


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
    return len(loaded), len(new_state)


def ctc_loss_for_batch(log_probs, labels, lengths, blank_id):
    """log_probs: (B,T,C). labels: (B,) single class id per clip.
    Length-1 target per clip -- see module docstring for why this is a
    real, if unusual, application of nn.CTCLoss, not a workaround."""
    log_probs_t_major = log_probs.permute(1, 0, 2)  # (T,B,C)
    target_lengths = torch.ones(labels.shape[0], dtype=torch.long)
    loss_fn = nn.CTCLoss(blank=blank_id, zero_infinity=True)
    return loss_fn(log_probs_t_major, labels, lengths, target_lengths)


@torch.no_grad()
def evaluate(model, loader, device, k=TOP_K_FOR_TOP5):
    model.eval()
    correct1, correctk, total = 0, 0, 0
    for batch in loader:
        landmarks = batch["landmarks"].to(device)
        mask = batch["mask"].to(device)
        labels = batch["labels"].to(device)
        log_probs = model(landmarks, mask)  # (B,T,C)
        # max-over-time pooling, excluding the blank column -- see module docstring
        class_scores = log_probs[..., :-1].max(dim=1).values  # (B, vocab_size)
        topk = class_scores.topk(min(k, class_scores.shape[-1]), dim=-1).indices
        correct1 += (topk[:, 0] == labels).sum().item()
        correctk += (topk == labels.unsqueeze(1)).any(dim=1).sum().item()
        total += labels.shape[0]
    model.train()
    return correct1 / max(total, 1), correctk / max(total, 1), total


def run_one_vocab_size(k, ranked_glosses, device, num_epochs=NUM_EPOCHS, save_checkpoint_path=None):
    print(f"\n{'='*90}\nVOCAB SIZE: {k} words\n{'='*90}")
    word_list = ranked_glosses[:k]
    word_to_label = {w: i for i, w in enumerate(sorted(word_list))}  # alphabetical, reproducible

    train_ds = IsolatedSignDataset(PROCESSED_DIR / "train", word_to_label, "train")
    val_ds = IsolatedSignDataset(PROCESSED_DIR / "val", word_to_label, "val")
    test_ds = IsolatedSignDataset(PROCESSED_DIR / "test", word_to_label, "test")
    print(f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} classes={k}")

    train_class_counts = Counter(train_ds.word_to_label[g] for g in train_ds.glosses)
    counts = list(train_class_counts.values())
    print(f"real per-class TRAIN counts: min={min(counts)} max={max(counts)} mean={sum(counts)/len(counts):.2f} "
          f"(n_classes_with_data={len(counts)}/{k})")
    if len(counts) < k:
        print(f"WARNING: {k - len(counts)} classes have ZERO train examples in the extracted data -- "
              f"data starvation at this vocab size, flagging explicitly.")

    # val/test zero-class check -- extraction may still be in progress when this runs,
    # so a class with zero val or test examples is a real possible confound (that class
    # is simply absent from the eval metric, not "hard") and must be caught, not silently
    # baked into best_val_top1/test_top1 -- see Step 4's confound-isolation requirement.
    zero_class_report = {}
    for split_name, ds in (("val", val_ds), ("test", test_ds)):
        split_counts = Counter(ds.word_to_label[g] for g in ds.glosses)
        n_zero = k - len(split_counts)
        zero_class_report[split_name] = n_zero
        if n_zero > 0:
            print(f"WARNING: {n_zero}/{k} classes have ZERO {split_name} examples in the CURRENTLY extracted "
                  f"data -- extraction may still be in progress. These classes are silently absent from "
                  f"{split_name} accuracy, not genuinely harder. Confound, not difficulty.")
        else:
            print(f"{split_name} coverage: all {k} classes have >=1 example (n={len(split_counts)}/{k})")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    torch.manual_seed(42)
    model = CTCIsolatedSignModel(vocab_size=k).to(device)
    n_loaded, n_total = load_pretrained_encoder(model.encoder, RESUME_CHECKPOINT)
    print(f"Loaded {n_loaded}/{n_total} pretrained encoder params from {RESUME_CHECKPOINT}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    history = []
    best_val_top1 = -1.0
    best_state = None

    model.train()
    for epoch in range(1, num_epochs + 1):
        total_loss, n_batches = 0.0, 0
        for batch in train_loader:
            landmarks = batch["landmarks"].to(device)
            mask = batch["mask"].to(device)
            labels = batch["labels"].to(device)
            lengths = batch["lengths"].to(device)

            log_probs = model(landmarks, mask)
            loss = ctc_loss_for_batch(log_probs, labels, lengths, model.blank_id)

            if not torch.isfinite(loss):
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_MAX_NORM)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        val_top1, val_top5, val_n = evaluate(model, val_loader, device)
        mean_loss = total_loss / max(n_batches, 1)
        history.append({"epoch": epoch, "train_loss": mean_loss, "val_top1": val_top1, "val_top5": val_top5})
        if epoch % 5 == 0 or epoch == 1:
            print(f"[epoch {epoch:3d}] train_loss={mean_loss:.4f}  val_top1={val_top1:.4f}  val_top5={val_top5:.4f} (n={val_n})")

        if val_top1 > best_val_top1:
            best_val_top1 = val_top1
            best_state = {kk: v.clone() for kk, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    if save_checkpoint_path is not None:
        # No prior vocab-scaling run ever persisted a checkpoint (best_state
        # lived only in memory, used to restore before test eval, then
        # discarded) -- this project has no ASL-Citizen-trained checkpoint
        # anywhere on disk otherwise. Saving here so the WLASL cross-dataset
        # check has something real to load.
        Path(save_checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": model.state_dict(),
            "word_to_label": word_to_label,
            "vocab_size": k,
        }, save_checkpoint_path)
        print(f"Saved checkpoint to {save_checkpoint_path}")

    test_top1, test_top5, test_n = evaluate(model, test_loader, device)
    val_test_gap = best_val_top1 - test_top1
    print(f"\n=== VOCAB SIZE {k}: FINAL TEST RESULT ===")
    print(f"best_val_top1={best_val_top1:.4f}  test_top1={test_top1:.4f}  test_top5={test_top5:.4f}  "
          f"val_test_gap={val_test_gap:+.4f}  (n_test={test_n}, chance={1/k:.4f})")

    return {
        "vocab_size": k, "n_train": len(train_ds), "n_val": len(val_ds), "n_test": len(test_ds),
        "train_class_counts": {"min": min(counts), "max": max(counts), "mean": sum(counts)/len(counts)},
        "n_classes_with_train_data": len(counts),
        "zero_class_classes": {"val": zero_class_report["val"], "test": zero_class_report["test"]},
        "history": history, "best_val_top1": best_val_top1,
        "test_top1": test_top1, "test_top5": test_top5, "test_n": test_n,
        "val_test_gap": val_test_gap, "chance_accuracy": 1 / k,
    }


def main():
    device = get_device()
    ranked, train_c, val_c, test_c = load_gloss_ranking()

    results = []
    out_path = "logs/asl_citizen_vocab_scaling_results.json"
    for k in VOCAB_SIZES:
        r = run_one_vocab_size(k, ranked, device)
        results.append(r)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\n\nDone. Results in {out_path}")
    print("\n=== SUMMARY TABLE ===")
    print(f"{'vocab':>6} {'test_top1':>10} {'test_top5':>10} {'val_test_gap':>13} {'min_train_n':>12}")
    for r in results:
        print(f"{r['vocab_size']:>6} {r['test_top1']:>10.4f} {r['test_top5']:>10.4f} {r['val_test_gap']:>13.4f} {r['train_class_counts']['min']:>12}")


if __name__ == "__main__":
    main()
