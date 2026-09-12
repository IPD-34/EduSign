"""diagnose_gloss_position.py -- check whether the model is learning
the hard (gloss) position vs just the easy (EOS) position, and how
close the true gloss token is to the top of the model's ranking."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from src.preprocessing.sample_writer import load_sample
from src.models.feature_utils import flatten_frame_features
from src.models.sign_transformer import SignTransformer
from src.models.train_utils import load_checkpoint, get_device

with open("data/vocab.json", "r", encoding="utf-8") as f:
    token_to_id = json.load(f)
id_to_token = {v: k for k, v in token_to_id.items()}
pad_id = token_to_id["<PAD>"]
bos_id = token_to_id["<BOS>"]
eos_id = token_to_id["<EOS>"]

device = get_device()
model = SignTransformer(vocab_size=len(token_to_id), pad_token_id=pad_id)
model.to(device)
dummy_opt = torch.optim.AdamW(model.parameters())
step = load_checkpoint(model, dummy_opt, "checkpoints/day12_latest.pt", device=device)
model.eval()
print(f"Loaded checkpoint step {step}\n")

test_samples = [("00336", "abdomen")]  # add a couple more real train IDs if you have them handy

for video_id, true_gloss in test_samples:
    npz_path = Path("data/processed/wlasl/train") / f"{video_id}.npz"
    sample = load_sample(npz_path)
    landmarks = torch.from_numpy(flatten_frame_features(sample)).float().unsqueeze(0).to(device)
    encoder_mask = torch.ones(1, landmarks.shape[1], dtype=torch.bool, device=device)
    target_tokens = torch.tensor([[bos_id]], device=device)

    with torch.no_grad():
        logits = model(landmarks, encoder_mask, target_tokens, target_padding_mask=None)

    probs = torch.softmax(logits[0, -1, :], dim=-1)
    true_id = token_to_id.get(true_gloss, token_to_id["<UNK>"])
    true_prob = probs[true_id].item()
    rank = (probs > true_prob).sum().item() + 1  # 1 = top rank
    top5_ids = torch.topk(probs, 5).indices.tolist()
    top5_tokens = [id_to_token.get(i, "<UNK>") for i in top5_ids]

    print(f"{video_id} (true={true_gloss}, true_id={true_id})")
    print(f"  true-gloss probability: {true_prob:.6f}  (random baseline: {1/len(token_to_id):.6f})")
    print(f"  true-gloss rank:        {rank} / {len(token_to_id)}")
    print(f"  top-5 predictions:      {top5_tokens}")
    print()