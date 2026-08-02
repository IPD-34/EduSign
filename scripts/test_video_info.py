from pathlib import Path

from src.preprocessing.video_reader import VideoReader

reader = VideoReader()

video = reader.read(
    Path("data/raw/wlasl/videos/70212.mp4")
)

print(f"Frames: {video['num_frames']}")
print(f"FPS: {video['fps']}")