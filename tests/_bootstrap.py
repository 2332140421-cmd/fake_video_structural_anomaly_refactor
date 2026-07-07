"""Test runner bootstrap for direct execution from IDE terminals.

Some IDE run buttons execute test files with /usr/bin/python3 directly. That
interpreter does not contain this project's dependencies. This helper re-runs
the current test file with the project-local environment and pytest.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def ensure_project_test_environment(test_file: str) -> None:
    """Re-execute a directly run test file with .venv/bin/python and pytest."""

    project_root = Path(__file__).resolve().parents[1]
    project_python = project_root / ".venv" / "bin" / "python"
    current_python = Path(sys.executable).resolve()

    if not project_python.exists():
        raise RuntimeError(
            "Project Python environment is missing. Expected: "
            f"{project_python}. Run from the project root: ./scripts/run_tests.sh"
        )

    if current_python != project_python.resolve():
        os.execv(
            str(project_python),
            [str(project_python), "-m", "pytest", str(Path(test_file).resolve())],
        )

