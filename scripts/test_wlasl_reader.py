import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.metadata_readers.wlasl_reader import WLASLReader

reader = WLASLReader()

print("\nFirst five samples:\n")

for i, sample in enumerate(reader.samples()):

    print(sample)

    if i == 4:
        break