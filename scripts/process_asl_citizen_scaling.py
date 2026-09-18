"""
scripts/process_asl_citizen_scaling.py

Runs the EXISTING, already hand-detection-fixed MediaPipe extraction
pipeline (same code path validated and improved in the prior ASL
Citizen task -- not re-derived here) over the top-200-word ASL Citizen
vocabulary-scaling slice (6,744 videos, 3.7GB).

Output: data/processed/asl_citizen_scaling/{train,val,test}/*.npz --
covers the 40/100/200-word points of the scaling curve via local
filtering at training time (the word lists nest by construction, since
they are all prefixes of one frequency-ranked list).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.dataset_processor import DatasetProcessor
from src.preprocessing.metadata_readers.asl_citizen_reader import ASLCitizenSliceReader

LOCAL_DIR = Path(
    r"C:\Users\KRUPAS~1\AppData\Local\Temp\claude\D--College-Projects---Resume-bidirectional-sign-language"
    r"\1f4ac0b6-58be-4111-abc9-012b28f39d49\scratchpad\asl_citizen_scaling"
)
SPLITS_DIR = Path(
    r"C:\Users\KRUPAS~1\AppData\Local\Temp\claude\D--College-Projects---Resume-bidirectional-sign-language"
    r"\1f4ac0b6-58be-4111-abc9-012b28f39d49\scratchpad\asl_citizen\ASL_Citizen\splits"
)
PROCESSED_DIR = Path("data/processed/asl_citizen_scaling")


def main():
    import json
    with open("logs/_tmp_asl_citizen_top400_words.json") as f:
        top400 = json.load(f)
    word_filter = set(top400[:200])  # exactly what was downloaded

    reader = ASLCitizenSliceReader(LOCAL_DIR, SPLITS_DIR, word_filter=word_filter)
    processor = DatasetProcessor(reader, PROCESSED_DIR)
    processor.process()


if __name__ == "__main__":
    main()
