#!/usr/bin/env python3
"""Check the local depth-estimation runtime environment."""

from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _ensure_project_environment() -> None:
    """Re-execute with the project .venv Python when available."""

    project_python = PROJECT_ROOT / ".venv" / "bin" / "python"
    current_python = Path(sys.executable).resolve()
    if project_python.exists() and current_python != project_python.resolve():
        os.execv(str(project_python), [str(project_python), *sys.argv])


def main() -> None:
    """Print Python, Torch, CUDA, and depth dependency status."""

    print(f"python executable: {sys.executable}")
    print(f"python version: {sys.version.split()[0]}")

    try:
        import torch

        print(f"torch version: {torch.__version__}")
        print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    except ImportError:
        print("torch is not installed. Do not reinstall blindly; check YOLO first.")

    missing: list[str] = []

    try:
        import transformers
        from transformers import pipeline  # noqa: F401

        print(f"transformers version: {transformers.__version__}")
        print("transformers.pipeline import: ok")
    except ImportError as exc:
        print(f"transformers import failed: {exc}")
        missing.append("transformers")

    try:
        from PIL import Image  # noqa: F401

        print("PIL.Image import: ok")
    except ImportError as exc:
        print(f"PIL.Image import failed: {exc}")
        missing.append("pillow")

    try:
        import numpy as np

        print(f"numpy version: {np.__version__}")
    except ImportError as exc:
        print(f"numpy import failed: {exc}")
        missing.append("numpy")

    if missing:
        print("Missing dependency/dependencies detected.")
        print("Install depth dependencies with: pip install transformers pillow")
    else:
        print("Depth model environment check passed.")


if __name__ == "__main__":
    _ensure_project_environment()
    main()
