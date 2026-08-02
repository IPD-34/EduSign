"""
One-off check: computes pixel-level difference between consecutive
frames (104-108) to confirm whether frames are truly identical
(read/seek bug) or just visually similar (signer at rest).
"""
import cv2
import glob
import numpy as np

def main():
    video_path = glob.glob('data/raw/asllrp/videos/*.mov')[0]
    cap = cv2.VideoCapture(video_path)

    frames = {}
    frame_idx = 0
    targets = set([104, 105, 106, 107, 108])
    max_target = max(targets)

    while frame_idx <= max_target:
        ret, frame = cap.read()
        if not ret:
            print(f'ended at {frame_idx}')
            break
        if frame_idx in targets:
            frames[frame_idx] = frame.copy()
        frame_idx += 1

    cap.release()

    keys = sorted(frames.keys())
    for i in range(len(keys) - 1):
        a, b = frames[keys[i]], frames[keys[i + 1]]
        diff = np.abs(a.astype(int) - b.astype(int)).sum()
        print(f'frame {keys[i]} vs {keys[i+1]}: pixel diff sum = {diff}')

if __name__ == '__main__':
    main()