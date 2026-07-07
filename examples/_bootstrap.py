"""Example runner bootstrap for the project-local Python environment."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def ensure_project_environment() -> Path:
    """Re-execute examples with the project .venv Python when needed."""

    project_root = Path(__file__).resolve().parents[1]
    project_python = project_root / ".venv" / "bin" / "python"
    current_python = Path(sys.executable).resolve()

    if not project_python.exists():
        raise RuntimeError(
            "Project Python environment is missing. Expected: "
            f"{project_python}. Create it before running examples."
        )

    if current_python != project_python.resolve():
        os.execv(str(project_python), [str(project_python), *sys.argv])

    return project_root
