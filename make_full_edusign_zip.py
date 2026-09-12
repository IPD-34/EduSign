from pathlib import Path
import zipfile

root = Path.cwd()
zip_path = root / "edusign_full_project.zip"

if zip_path.exists():
    zip_path.unlink()

# Directories/files to exclude
exclude_dirs = {
    ".venv",
    ".git",
    "__pycache__",
    "data",
    "test_checkpoints",
    "kaggle_download",
    "logs",
}

exclude_names = {
    "train_utils_before_atomic.py",
    "edusign_scripts.zip",
    "edusign_code.zip",
    "edusign_month2_asllrp_data.zip",
    "edusign_wlasl_data.zip",
    "edusign_full_project.zip",
}

exclude_suffixes = {
    ".pyc",
}

# The ONE checkpoint we explicitly want
checkpoint = root / "checkpoints" / "month2_asllrp" / "step_0058000.pt"

if not checkpoint.exists():
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

count = 0

with zipfile.ZipFile(
    zip_path,
    "w",
    compression=zipfile.ZIP_DEFLATED,
) as z:

    for file in root.rglob("*"):

        if not file.is_file():
            continue

        rel = file.relative_to(root)

        # Exclude directories
        if any(part in exclude_dirs for part in rel.parts):
            continue

        # Exclude filenames
        if file.name in exclude_names:
            continue

        # Exclude Python bytecode
        if file.suffix in exclude_suffixes:
            continue

        # Exclude all checkpoints from recursive scan.
        # We add only the verified Month 2 checkpoint below.
        if rel.parts and rel.parts[0] == "checkpoints":
            continue

        arcname = rel.as_posix()
        z.write(file, arcname)
        count += 1

    # Explicitly add the verified checkpoint
    z.write(
        checkpoint,
        "checkpoints/month2_asllrp/step_0058000.pt",
    )
    count += 1

print(f"Created: {zip_path}")
print(f"Files added: {count}")
print(f"ZIP size: {zip_path.stat().st_size / 1e6:.2f} MB")
