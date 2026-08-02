from pathlib import Path

from src.preprocessing.video_reader import VideoReader


video = Path(
    "data/raw/wlasl/videos/68011.mp4"
)

reader = VideoReader()

result = reader.read(video)

print()

print("Frames :", result["num_frames"])
print("FPS    :", result["fps"])
print("Height :", result["height"])
print("Width  :", result["width"])

print()

print(
    "First frame shape:",
    result["frames"][0].shape,
)