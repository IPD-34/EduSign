"""
One-off check: dumps frames around a single-frame SIGN marker
(DOMINANT_HAND START_FRAME=118, END_FRAME=118) to visually confirm
whether ASLLRP frame indices are 0-indexed or 1-indexed.
"""
import cv2
import glob
import os

VIDEO_GLOB = 'data/raw/asllrp/videos/*.mov'
OUTPUT_DIR = 'logs/frame_checks'
FRAMES_TO_CHECK = [116, 117, 118, 119, 120]  # centered on SIGN marker frame=118

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    video_path = glob.glob(VIDEO_GLOB)[0]
    print(f"Using video: {video_path}")

    cap = cv2.VideoCapture(video_path)

    frame_idx = 0
    target_set = set(FRAMES_TO_CHECK)
    max_target = max(FRAMES_TO_CHECK)

    while frame_idx <= max_target:
        ret, frame = cap.read()
        if not ret:
            print(f"Video ended early at frame {frame_idx}")
            break

        if frame_idx in target_set:
            out_path = os.path.join(OUTPUT_DIR, f'sign_frame_{frame_idx}.png')
            cv2.imwrite(out_path, frame)
            print(f"Saved frame {frame_idx} -> {out_path}")

        frame_idx += 1

    cap.release()

if __name__ == '__main__':
    main()