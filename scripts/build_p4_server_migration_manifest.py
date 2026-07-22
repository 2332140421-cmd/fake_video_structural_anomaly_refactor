#!/usr/bin/env python3
"""Build a deterministic, non-copying P4 server migration manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from semantic3d.runtime.migration_manifest import (  # noqa: E402
    build_migration_manifest,
    write_migration_artifacts,
)
from semantic3d.runtime.runtime_config import load_runtime_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-config", default="configs/runtime/local_wsl.yaml")
    parser.add_argument("--output-root", default="outputs/p4c3a_server_migration")
    args = parser.parse_args()
    runtime = load_runtime_config(PROJECT_ROOT / args.runtime_config)
    manifest = build_migration_manifest(runtime)
    result = write_migration_artifacts(PROJECT_ROOT / args.output_root, manifest)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

