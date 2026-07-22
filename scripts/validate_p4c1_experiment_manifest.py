#!/usr/bin/env python3
"""Validate P4-C1 manifest integrity, split isolation, and reproducibility."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from semantic3d.dataset_builder.writer import atomic_write_json  # noqa: E402
from semantic3d.experiment_protocol.manifest_builder import (  # noqa: E402
    validate_manifest_artifacts,
)


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/p4c1_experiment_manifest_v1.yaml")
    parser.add_argument("--manifest-root", default="outputs/p4c1_experiment_manifest")
    args = parser.parse_args()
    root = _resolve(args.manifest_root).resolve()
    result = validate_manifest_artifacts(
        root,
        _resolve(args.config).resolve(),
        project_root=PROJECT_ROOT,
    )
    atomic_write_json(root / "validation_report.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
