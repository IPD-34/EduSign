"""sanity_check_train_clip.py -- confirm predict_gloss() works on a
clip the model has actually been trained on."""
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

device = get_device()
model = SignTransformer(vocab_size=len(token_to_id), pad_token_id=pad_id)
model.to(device)
dummy_opt = torch.optim.AdamW(model.parameters())
step = load_checkpoint(model, dummy_opt, "checkpoints/day12_latest.pt", device=device)
model.eval()
print(f"Loaded checkpoint step {step}")

# swap in a few real train IDs you know were in the training set
for video_id, true_gloss in [("00336", "abdomen")]:
    npz_path = Path("data/processed/wlasl/train") / f"{video_id}.npz"
    sample = load_sample(npz_path)
    landmarks = torch.from_numpy(flatten_frame_features(sample)).float().unsqueeze(0).to(device)
    encoder_mask = torch.ones(1, landmarks.shape[1], dtype=torch.bool, device=device)
    target_tokens = torch.tensor([[bos_id]], device=device)
    with torch.no_grad():
        logits = model(landmarks, encoder_mask, target_tokens, target_padding_mask=None)
    predicted_id = logits[0, -1, :].argmax().item()
    print(f"{video_id}: true={true_gloss}, predicted={id_to_token.get(predicted_id, '<UNK>')}")