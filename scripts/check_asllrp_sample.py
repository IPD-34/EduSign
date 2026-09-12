import numpy as np
from pathlib import Path

sample_path = next(Path("data/processed/asllrp/train").rglob("*.npz"))
print(f"Checking: {sample_path}")

d = np.load(sample_path, allow_pickle=True)
print(f"Keys in file: {list(d.keys())}")

if "landmarks" in d:
    print(f"landmarks shape: {d['landmarks'].shape}")
else:
    # Your actual .npz schema might store things differently
    # (e.g. separate pose/face/hand arrays, or a pickled SampleMetadata
    # object) -- the printed key list above will tell you the real
    # field names to check next.
    for k in d.keys():
        try:
            print(f"  {k}: shape={d[k].shape}")
        except AttributeError:
            print(f"  {k}: {type(d[k])}")