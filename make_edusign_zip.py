from pathlib import Path
import zipfile

root = Path.cwd()
zip_path = root / "edusign_scripts.zip"

items = [
    root / "main.py",
    root / "scripts",
    root / "src",
    root / "pyproject.toml",
    root / "uv.lock",
    root / "README.md",
]

if zip_path.exists():
    zip_path.unlink()

count = 0

with zipfile.ZipFile(
    zip_path,
    "w",
    compression=zipfile.ZIP_DEFLATED,
) as z:

    for item in items:
        if not item.exists():
            print(f"WARNING: missing: {item}")
            continue

        if item.is_file():
            arcname = item.relative_to(root).as_posix()
            z.write(item, arcname)
            count += 1
            continue

        for file in item.rglob("*"):
            if not file.is_file():
                continue

            if "__pycache__" in file.parts:
                continue

            if file.suffix == ".pyc":
                continue

            if file.name == "train_utils_before_atomic.py":
                continue

            arcname = file.relative_to(root).as_posix()
            z.write(file, arcname)
            count += 1

print(f"Created: {zip_path}")
print(f"Files added: {count}")
print(f"ZIP size: {zip_path.stat().st_size} bytes")
