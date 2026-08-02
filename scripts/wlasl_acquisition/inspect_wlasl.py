import json
from pathlib import Path

json_path = Path("data/raw/wlasl/metadata/WLASL_v0.3.json")

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

print("Total glosses:", len(data))

print("\nFirst entry:\n")
print(data[0])