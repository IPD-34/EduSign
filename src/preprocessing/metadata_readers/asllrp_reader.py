"""
Reads ASLLRP SignStream XML annotation files and yields one training
instance per UTTERANCE (a continuous, multi-gloss sentence-level clip).

Option A (deliberate scope decision): ASLLRP's raw annotation only
gives a single reference/anchor frame per SIGN
(DOMINANT_HAND START_FRAME == END_FRAME) -- not a true per-gloss span.
This reader does NOT fabricate per-gloss boundaries from that.
It exposes:
  - the video file an utterance belongs to
  - the frame span to extract (utterance-level START_FRAME/END_FRAME)
  - the ordered gloss sequence (SIGN LABELs, in order) for CTC supervision,
    stored as a "|"-joined string in the gloss field (see ASLLRPSample)

C2's boundary-prediction loss has no real per-gloss ground truth from
this reader -- only CTC-style ordered-sequence supervision at the
utterance level. Document this explicitly wherever C2's claims are
written up.

FRAME INDEXING: ASLLRP's START_FRAME/END_FRAME are 1-indexed (empirically
confirmed via 5-sample batch visual inspection -- see project notes).
frame_sampler.select_frame_indices() already handles the -1 adjustment
for frame_start; this reader passes raw 1-indexed values through
unchanged, exactly as wlasl_reader.py does for WLASL's frame_start.

SPLIT ASSIGNMENT: ASLLRP's XML carries no train/val/test split info.
Assigned here via deterministic hash of utterance_id (80/10/10) so
Month 2's held-out evaluation has real, reproducible ASLLRP data,
not just training data. Override _assign_split() if a different
policy (e.g. split by collection_id, keeping whole sessions together)
is preferred.
"""

from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass
class ASLLRPUtterance:
    utterance_id: str
    collection_id: str
    collection_name: str
    video_filename: str          # e.g. "1-Introduction-SD.mov"
    video_path: Path             # resolved path; existence checked in samples()
    start_frame: int
    end_frame: int
    translation: str             # English gloss string, reference only
    glosses: list[str]           # ordered SIGN LABELs -> CTC target sequence
    xml_source: Path             # originating file, for debugging


@dataclass
class ASLLRPSample:
    """DatasetProcessor-compatible view of an ASLLRPUtterance."""
    video_id: str
    gloss: str          # "|"-joined ordered gloss sequence
    split: str
    source: str
    video_path: Path
    frame_start: int    # raw 1-indexed value; frame_sampler applies -1
    frame_end: int
    translation: str


class ASLLRPReader:
    """Iterates ASLLRP XML files, yields ASLLRPUtterance instances."""

    def __init__(self, xml_dir: str | Path, video_dir: str | Path):
        self.xml_dir = Path(xml_dir)
        self.video_dir = Path(video_dir)

        if not self.xml_dir.exists():
            raise FileNotFoundError(f"ASLLRP xml_dir does not exist: {self.xml_dir}")
        if not self.video_dir.exists():
            raise FileNotFoundError(f"ASLLRP video_dir does not exist: {self.video_dir}")

    def _iter_xml_files(self) -> Iterator[Path]:
        yield from sorted(self.xml_dir.glob("*.xml"))

    @staticmethod
    def _clean(text: str | None) -> str:
        """SignStream wraps text content in literal single quotes -- strip them."""
        if text is None:
            return ""
        t = text.strip()
        if len(t) >= 2 and t[0] == "'" and t[-1] == "'":
            t = t[1:-1]
        return t

    @staticmethod
    def _assign_split(utterance_id: str) -> str:
        """Deterministic 80/10/10 split via hash -- stable across reruns,
        no random-seed dependency, no leakage risk from reshuffling."""
        h = int(hashlib.md5(utterance_id.encode("utf-8")).hexdigest(), 16)
        bucket = h % 100
        if bucket < 80:
            return "train"
        elif bucket < 90:
            return "val"
        else:
            return "test"

    def _parse_file(self, xml_path: Path) -> Iterator[ASLLRPUtterance]:
        try:
            tree = ET.parse(xml_path)
        except ET.ParseError as e:
            print(f"[ASLLRPReader] WARNING: failed to parse {xml_path.name}: {e}")
            return
        root = tree.getroot()

        for collection in root.iter("COLLECTION"):
            collection_id = collection.get("ID", "")
            collection_name = collection.get("NAME", "")

            media_file_el = collection.find(".//MEDIA-FILES/MEDIA-FILE")
            if media_file_el is None:
                continue
            video_filename = media_file_el.get("FILE-NAME", "")
            video_path = self.video_dir / video_filename

            for utterance in collection.iter("UTTERANCE"):
                utterance_id = utterance.get("ID", "")
                try:
                    start_frame = int(utterance.get("START_FRAME", -1))
                    end_frame = int(utterance.get("END_FRAME", -1))
                except (TypeError, ValueError):
                    start_frame, end_frame = -1, -1

                if start_frame < 0 or end_frame < 0 or end_frame <= start_frame:
                    continue  # malformed/degenerate span -- skip, don't fabricate

                translation_el = utterance.find("TRANSLATION")
                translation = self._clean(
                    translation_el.text if translation_el is not None else None
                )

                glosses: list[str] = []
                for sign in utterance.iter("SIGN"):
                    label_el = sign.find("LABEL")
                    label = self._clean(label_el.text if label_el is not None else None)
                    if label:
                        glosses.append(label)

                if not glosses:
                    continue  # nothing to supervise on

                yield ASLLRPUtterance(
                    utterance_id=utterance_id,
                    collection_id=collection_id,
                    collection_name=collection_name,
                    video_filename=video_filename,
                    video_path=video_path,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    translation=translation,
                    glosses=glosses,
                    xml_source=xml_path,
                )

    def __iter__(self) -> Iterator[ASLLRPUtterance]:
        for xml_path in self._iter_xml_files():
            yield from self._parse_file(xml_path)

    def samples(self) -> Iterator[ASLLRPSample]:
        """DatasetProcessor entry point -- yields processor-compatible samples,
        with correctly-named frame_start/frame_end fields, video existence
        checked (missing videos skipped + logged, not silently included),
        and a deterministic split assigned."""
        missing = 0
        total = 0

        for utt in self:
            total += 1

            if not utt.video_path.exists():
                missing += 1
                print(f"[ASLLRPReader] WARNING: video not found, skipping "
                      f"utterance {utt.utterance_id}: {utt.video_path}")
                continue

            yield ASLLRPSample(
                video_id=f"asllrp_{utt.utterance_id}",
                gloss="|".join(utt.glosses),
                split=self._assign_split(utt.utterance_id),
                source="asllrp",
                video_path=utt.video_path,
                frame_start=utt.start_frame,
                frame_end=utt.end_frame,
                translation=utt.translation,
            )

        if total:
            print(f"[ASLLRPReader] {missing}/{total} utterances skipped "
                  f"(video file not found)")