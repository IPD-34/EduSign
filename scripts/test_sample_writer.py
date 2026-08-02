from pathlib import Path

from src.preprocessing.sample_writer import (
    load_sample,
    write_sample,
)
from src.preprocessing.sequence_builder import SequenceBuilder


builder = SequenceBuilder()

sample = builder.build(
    Path("data/raw/wlasl/videos/68011.mp4"),
    video_id="68011",
    gloss="book",
    split="train",
    source="youtube",
)

output_path = Path("data/processed/test/68011.npz")

write_sample(sample, output_path)

loaded = load_sample(output_path)

print("\n=== Metadata ===")
print(loaded.metadata)

print("\n=== Shapes ===")
print("Pose       :", loaded.pose.shape)
print("Face       :", loaded.face.shape)
print("Left Hand  :", loaded.left_hand.shape)
print("Right Hand :", loaded.right_hand.shape)

print("\n=== Presence ===")
print("Pose       :", loaded.presence.pose.shape)
print("Face       :", loaded.presence.face.shape)
print("Left Hand  :", loaded.presence.left_hand.shape)
print("Right Hand :", loaded.presence.right_hand.shape)

print("\nSample saved and loaded successfully!")