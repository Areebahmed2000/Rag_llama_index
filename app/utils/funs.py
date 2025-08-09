from fastapi import UploadFile
import shutil
from pathlib import Path


def save_uploaded_file(uploaded_file: UploadFile, save_dir: Path) -> Path:
    """Save uploaded file to the given directory."""
    file_location = save_dir / uploaded_file.filename
    with open(file_location, "wb") as f:
        shutil.copyfileobj(uploaded_file.file, f)
    return file_location