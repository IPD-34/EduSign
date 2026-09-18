"""
asl_citizen_reader.py

Reads a LOCAL SLICE of ASL Citizen (Microsoft Research) metadata and
exposes each video as a structured sample -- same interface as
WLASLReader (video_id, gloss, split, source, video_path,
frame_start/frame_end), so DatasetProcessor (dataset_processor.py)
consumes it identically, unmodified.

This is deliberately scoped to a LOCAL SLICE, not the full 83,399-video
corpus: ASL Citizen ships as one 42.8GB zip (videos + splits/*.csv all
together, confirmed via remotezip's central-directory listing over
HTTP range requests -- no full download needed to inspect it). For a
small validation run, only the videos for a chosen word list were
range-downloaded individually (see scripts/download_asl_citizen_slice.py
in the session scratchpad) into a local directory laid out as
<local_dir>/videos/<split>/<Video file>.

ASL Citizen's own splits/{train,val,test}.csv are used AS-IS for the
split assignment (source == split-file membership) -- confirmed by
direct inspection that these three files are already fully
signer-disjoint (zero Participant ID overlap in either direction),
matching the paper's own signer-independent evaluation protocol. This
reader does not invent its own train/test split.

source is set to the literal string "asl_citizen" (never "asllrp") --
src/models/dataset.py's gloss-to-ID routing only special-cases
source == "asllrp"; anything else (this included) takes the plain
WLASL-style token_to_id lookup path, and boundary_target correctly
falls back to an all-zero array for non-ASLLRP sources with no code
changes needed, exactly as it already does for WLASL.

KNOWN FORMAT QUIRK, NOT HANDLED HERE (see extraction report): some ASL
Citizen glosses carry a trailing sense-disambiguation digit baked into
the Gloss column itself (e.g. "SOCCER2", "WRISTWATCH3" -- note this is
NOT the same as the "Video file" column, which for the same rows reads
"...SOCCER 2.mp4" / "...WRISTWATCH 3.mp4" WITH a space; the Gloss
column strips it). None of the 40 words selected for this validation
slice happen to have such a variant, so this reader does no digit-
suffix normalization -- flagged explicitly rather than silently
handled, so a future larger-scale extraction doesn't assume this was
already solved.
"""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ASLCitizenSample:
    video_id: str
    gloss: str
    split: str
    source: str
    video_path: Path
    frame_start: Optional[int] = None
    frame_end: Optional[int] = None


class ASLCitizenSliceReader:
    """
    local_dir: directory containing videos/<split>/<Video file>.mp4
    splits_dir: directory containing the (locally extracted) official
                train.csv/val.csv/test.csv from ASL_Citizen/splits/.
    word_filter: optional set of lowercased glosses to restrict to --
                 used here to scope the reader to exactly the 40-word
                 validation slice, matching which videos were actually
                 downloaded (a reader that yielded rows for videos not
                 present on disk would fail loudly at extraction time,
                 not silently skip, unlike WLASLReader's existence
                 check -- deliberate: for a small, fully-downloaded
                 slice, a missing file is a real bug worth surfacing).
    """

    def __init__(self, local_dir: Path, splits_dir: Path, word_filter: set[str] | None = None):
        self.local_dir = Path(local_dir)
        self.splits_dir = Path(splits_dir)
        self.word_filter = word_filter

    def samples(self):
        for split in ("train", "val", "test"):
            csv_path = self.splits_dir / f"{split}.csv"
            with open(csv_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    gloss = row["Gloss"].strip().lower()
                    if self.word_filter is not None and gloss not in self.word_filter:
                        continue
                    fname = row["Video file"]
                    video_path = self.local_dir / "videos" / split / fname
                    if not video_path.exists():
                        continue  # not part of the downloaded slice
                    video_id = Path(fname).stem  # filename's own hash prefix is globally unique
                    yield ASLCitizenSample(
                        video_id=f"aslciti_{video_id}",
                        gloss=gloss,
                        split=split,
                        source="asl_citizen",
                        video_path=video_path,
                    )
