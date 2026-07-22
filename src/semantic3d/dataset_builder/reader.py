"""Read-only helpers for structural-enhancement datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


class DatasetReader:
    """Read structural tables without implicitly joining label files."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def rows(self, relative_path: str) -> list[dict[str, Any]]:
        path = self.root / relative_path
        if not path.exists():
            return []
        return pq.read_table(path).to_pylist()

    def manifest(self) -> dict[str, Any]:
        return json.loads((self.root / "dataset_manifest.json").read_text(encoding="utf-8"))

    def labels(self) -> None:
        """Reject implicit label joins; downstream tasks must opt in separately."""

        raise RuntimeError("Labels are isolated and are not readable through DatasetReader")
