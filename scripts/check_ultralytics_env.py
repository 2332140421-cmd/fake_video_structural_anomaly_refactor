#!/usr/bin/env python3
"""Check the local Ultralytics/YOLO runtime environment."""

from __future__ import annotations

import sys
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "checkpoints" / "yolov8n.pt"


def _ensure_project_environment() -> None:
    """Re-execute with the project .venv Python when available."""

    project_python = PROJECT_ROOT / ".venv" / "bin" / "python"
    current_python = Path(sys.executable).resolve()
    if project_python.exists() and current_python != project_python.resolve():
        os.execv(str(project_python), [str(project_python), *sys.argv])


def main() -> None:
    """Print Python, Ultralytics, Torch, CUDA, and YOLO weight status."""

    print(f"python executable: {sys.executable}")
    print(f"python version: {sys.version.split()[0]}")

    try:
        import ultralytics

        print(f"ultralytics version: {ultralytics.__version__}")
    except ImportError:
        print(
            "ultralytics is not installed. Install it with: "
            "python -m pip install ultralytics"
        )
        return

    try:
        import torch

        print(f"torch version: {torch.__version__}")
        print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    except ImportError:
        print("torch is not installed, but ultralytics requires torch.")
        return

    DEFAULT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"expected YOLO weight path: {DEFAULT_MODEL_PATH}")
    print(f"weights exist: {DEFAULT_MODEL_PATH.exists()}")

    if not DEFAULT_MODEL_PATH.exists():
        print(
            "YOLO weights are missing. You can either let Ultralytics load "
            "'yolov8n.pt' automatically in the smoke test, or manually place "
            f"the file at: {DEFAULT_MODEL_PATH}"
        )
        return

    try:
        from ultralytics import YOLO

        model = YOLO(str(DEFAULT_MODEL_PATH))
        print(f"YOLO model loaded successfully: {type(model).__name__}")
    except Exception as exc:
        print(f"Failed to load YOLO model: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    _ensure_project_environment()
    main()
