from pathlib import Path

from src.preprocessing.sequence_builder import SequenceBuilder

builder = SequenceBuilder()

sample = builder.build(
    Path("data/raw/wlasl/videos/68011.mp4"),
    video_id="68011",
    gloss="book",
    split="train",
    source="youtube",
)

print("\n=== Metadata ===")
print(sample.metadata)

print("\n=== Landmark Shapes ===")
print("Pose       :", sample.pose.shape)
print("Face       :", sample.face.shape)
print("Left Hand  :", sample.left_hand.shape)
print("Right Hand :", sample.right_hand.shape)

print("\n=== Presence Shapes ===")
print("Pose       :", sample.presence.pose.shape)
print("Face       :", sample.presence.face.shape)
print("Left Hand  :", sample.presence.left_hand.shape)
print("Right Hand :", sample.presence.right_hand.shape)

print("\nSequence built successfully!")