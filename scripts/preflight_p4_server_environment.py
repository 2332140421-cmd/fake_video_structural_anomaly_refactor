#!/usr/bin/env python3
"""Run the non-destructive P4-C3A server environment preflight."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from semantic3d.runtime.preflight_report import (  # noqa: E402
    build_server_preflight,
    write_preflight_artifacts,
)
from semantic3d.runtime.runtime_config import load_runtime_config  # noqa: E402


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-config", default="configs/runtime/local_wsl.yaml")
    parser.add_argument("--smoke-config", default="configs/p4c3a_server_smoke_v1.yaml")
    parser.add_argument("--p4c2-validation", default="outputs/p4c2_formal_data_readiness/validation_report.json")
    parser.add_argument("--output-root", default="outputs/p4c3a_server_preflight")
    args = parser.parse_args()
    runtime = load_runtime_config(_resolve(args.runtime_config))
    smoke = yaml.safe_load(_resolve(args.smoke_config).read_text(encoding="utf-8"))
    p4c2 = json.loads(_resolve(args.p4c2_validation).read_text(encoding="utf-8"))
    report = build_server_preflight(runtime, smoke, p4c2)
    validation = write_preflight_artifacts(_resolve(args.output_root), report)
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    return 0 if validation["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

