"""
scripts/process_asl_citizen_slice.py

Runs the MediaPipe extraction pipeline (DatasetProcessor ->
SequenceBuilder -> SampleWriter, same code path as WLASL/ASLLRP) over
the small, already-downloaded ASL Citizen validation slice (40 words,
1,226 videos, 683.9MB -- see ASLCitizenSliceReader's docstring for how
that slice was selected and downloaded).

HAND-DETECTION FIX RE-RUN: extractor.py/hand_detector.py were changed
to fix a real, confirmed hand-detection failure on this data source
(root cause: scale/framing, not confidence -- see hand_roi.py's
docstring). Output now goes to data/processed/asl_citizen_validation_v2/
-- a SEPARATE directory from the original data/processed/
asl_citizen_validation/ (produced before the fix), so old and new
hand-presence statistics can be compared side by side on the exact
same 1,226 videos rather than overwriting the "before" data.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.dataset_processor import DatasetProcessor
from src.preprocessing.metadata_readers.asl_citizen_reader import ASLCitizenSliceReader

LOCAL_DIR = Path(
    r"C:\Users\KRUPAS~1\AppData\Local\Temp\claude\D--College-Projects---Resume-bidirectional-sign-language"
    r"\1f4ac0b6-58be-4111-abc9-012b28f39d49\scratchpad\asl_citizen"
)
SPLITS_DIR = LOCAL_DIR / "ASL_Citizen" / "splits"
PROCESSED_DIR = Path("data/processed/asl_citizen_validation_v2")

WORDS = [
    'bathroom', 'bye', 'family', 'father', 'friend', 'he', 'hello', 'help', 'his', 'home',
    'hungry', 'hurt', 'me', 'medicine', 'mother', 'my', 'no', 'our', 'pain', 'please',
    'rest', 'school', 'sick', 'sleep', 'sorry', 'student', 'that', 'thirsty', 'tired', 'toilet',
    'water', 'we', 'when', 'where', 'who', 'why', 'work', 'yes', 'you', 'your',
]


def main():
    reader = ASLCitizenSliceReader(LOCAL_DIR, SPLITS_DIR, word_filter=set(WORDS))
    processor = DatasetProcessor(reader, PROCESSED_DIR)
    processor.process()


if __name__ == "__main__":
    main()
