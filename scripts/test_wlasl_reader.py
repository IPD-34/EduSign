from src.preprocessing.metadata_readers.wlasl_reader import WLASLReader

reader = WLASLReader()

print("\nFirst five samples:\n")

for i, sample in enumerate(reader.samples()):

    print(sample)

    if i == 4:
        break