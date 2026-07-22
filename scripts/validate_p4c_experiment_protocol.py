#!/usr/bin/env python3
"""Validate a generated P4-C0 experiment protocol."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from semantic3d.dataset_builder.writer import atomic_write_json  # noqa: E402
from semantic3d.experiment_protocol.validation import validate_protocol  # noqa: E402


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-root", default="data/experiment_protocol_v1")
    parser.add_argument(
        "--structural-dataset-root",
        default="outputs/structural_enhancement_dataset/p4b5_six_video_full_observation",
    )
    args = parser.parse_args()
    protocol_root = _resolve(args.protocol_root)
    result = validate_protocol(protocol_root, _resolve(args.structural_dataset_root))
    atomic_write_json(protocol_root / "protocol_validation.json", asdict(result))
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    return 0 if result.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
