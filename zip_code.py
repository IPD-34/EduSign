import zipfile
from pathlib import Path

files_to_zip = []

for folder in ["src", "scripts"]:
    if Path(folder).exists():
        files_to_zip.extend(
            f for f in Path(folder).rglob("*")
            if "__pycache__" not in f.parts
        )

for f in [
    Path("main.py"),
    Path("pyproject.toml"),
    Path("requirements.txt"),
]:
    if f.exists():
        files_to_zip.append(f)

with zipfile.ZipFile("edusign_code.zip", "w", zipfile.ZIP_DEFLATED) as zf:
    for f in files_to_zip:
        if f.is_file():
            zf.write(f, arcname=str(f).replace("\\", "/"))

print("Done! Created edusign_code.zip")