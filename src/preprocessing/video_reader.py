"""
video_reader.py

Reads a video from disk, optionally restricted to specific frame indices.

- probe(): cheap metadata read (frame count + fps), no frame decoding —
  used to feed frame_sampler.select_frame_indices() before deciding what
  to actually decode.
- read(): sequential decode, keeping only requested indices. OpenCV
  seeking (CAP_PROP_POS_FRAMES) is unreliable on compressed video with
  sparse keyframes, so this decodes sequentially but stops immediately
  after the last requested index — trailing padding footage is never
  decoded at all. HolisticExtractor is only ever called on kept frames,
  by the caller.

No landmark extraction. No preprocessing.
"""

from pathlib import Path
from typing import List, Optional

import cv2


class VideoReader:

    def probe(self, video_path: Path) -> dict:
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(video_path)

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open video: {video_path}")

        fps = capture.get(cv2.CAP_PROP_FPS)
        num_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()

        return {"fps": fps, "num_frames": num_frames}

    def read(self, video_path: Path, frame_indices: Optional[List[int]] = None):
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(video_path)

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open video: {video_path}")

        fps = capture.get(cv2.CAP_PROP_FPS)
        wanted = set(frame_indices) if frame_indices is not None else None
        stop_after = max(frame_indices) if frame_indices else None

        frames = []
        kept_indices = []
        i = 0

        while True:
            success, frame = capture.read()
            if not success:
                break

            if wanted is None or i in wanted:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
                kept_indices.append(i)

            if stop_after is not None and i >= stop_after:
                break

            i += 1

        capture.release()

        result_height = frames[0].shape[0] if frames else 0
        result_width = frames[0].shape[1] if frames else 0

        return {
            "frames": frames,
            "frame_indices": kept_indices,
            "fps": fps,
            "num_frames": len(frames),
            "height": result_height,
            "width": result_width,
        }