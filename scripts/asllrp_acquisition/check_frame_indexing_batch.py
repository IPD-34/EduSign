"""
check_frame_indexing_batch.py

Batch version of the frame-indexing check: pulls several DOMINANT_HAND
single-frame markers (START_FRAME == END_FRAME) from different signs
across ASLLRP XML files, and dumps +/-2 frames around each for visual
comparison. One sample could be normal articulation variance — a
consistent pattern across several independent signs is real signal.
"""
import cv2
import glob
import os
import re
import xml.etree.ElementTree as ET

XML_GLOB = 'data/raw/asllrp/xml/*.xml'
VIDEO_DIR = 'data/raw/asllrp/videos'
OUTPUT_DIR = 'logs/frame_checks/batch'
NUM_SAMPLES = 5  # how many distinct sign markers to check
WINDOW = 2        # +/- frames around each marker


def find_single_frame_markers(xml_path):
    """Yields (media_filename, sign_label, frame) for every
    DOMINANT_HAND (or NON_DOMINANT_HAND) tag where START_FRAME == END_FRAME."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    media_file = None
    media_tag = root.find('.//MEDIA-FILE')
    if media_tag is not None:
        media_file = media_tag.attrib.get('FILE-NAME')

    for sign in root.iter('SIGN'):
        label_tag = sign.find('LABEL')
        label = label_tag.text if label_tag is not None else 'UNKNOWN'

        for hand_tag in ('DOMINANT_HAND', 'NON_DOMINANT_HAND'):
            hand = sign.find(hand_tag)
            if hand is not None:
                start = hand.attrib.get('START_FRAME')
                end = hand.attrib.get('END_FRAME')
                if start is not None and start == end:
                    yield media_file, label, int(start)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    xml_files = glob.glob(XML_GLOB)
    print(f"Found {len(xml_files)} XML files")

    collected = []
    for xml_path in xml_files:
        try:
            for media_file, label, frame in find_single_frame_markers(xml_path):
                collected.append((xml_path, media_file, label, frame))
                if len(collected) >= NUM_SAMPLES:
                    break
        except Exception as e:
            print(f"  Skipping {xml_path}: {e}")
        if len(collected) >= NUM_SAMPLES:
            break

    print(f"\nCollected {len(collected)} single-frame markers to check:\n")
    for xml_path, media_file, label, frame in collected:
        print(f"  {os.path.basename(xml_path)} | sign='{label}' | frame={frame} | video={media_file}")

    print()

    for i, (xml_path, media_file, label, frame) in enumerate(collected):
        # media_file from XML might not exactly match video filename on disk
        # (extension may differ, e.g. .mov referenced but stored differently)
        base_name = os.path.splitext(media_file)[0] if media_file else None
        matches = glob.glob(os.path.join(VIDEO_DIR, f"{base_name}.*")) if base_name else []

        if not matches:
            print(f"[{i}] Could not find video for '{media_file}' — skipping")
            continue

        video_path = matches[0]
        cap = cv2.VideoCapture(video_path)

        frame_idx = 0
        targets = set(range(frame - WINDOW, frame + WINDOW + 1))
        max_target = max(targets)

        safe_label = re.sub(r'[^A-Za-z0-9_]', '_', label)

        while frame_idx <= max_target:
            ret, fr = cap.read()
            if not ret:
                break
            if frame_idx in targets:
                out_path = os.path.join(
                    OUTPUT_DIR,
                    f"sample{i}_{safe_label}_frame{frame_idx}.png"
                )
                cv2.imwrite(out_path, fr)
            frame_idx += 1

        cap.release()
        print(f"[{i}] Saved frames around {frame} for sign '{label}' -> {OUTPUT_DIR}")

    print("\nDone. Open the images in logs/frame_checks/batch/ and compare "
          "each sign's marker frame against its neighbors.")


if __name__ == '__main__':
    main()