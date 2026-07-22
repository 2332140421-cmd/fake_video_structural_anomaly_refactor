#!/usr/bin/env python3
"""Validate an existing P4-C3A batch plan without execution."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from semantic3d.batch_execution.batch_schema import BatchRecord  # noqa: E402
from semantic3d.batch_execution.batch_validator import validate_batch_plan  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default="outputs/p4c3a_batch_plan/batch_plan.jsonl")
    args = parser.parse_args()
    path = PROJECT_ROOT / args.plan
    rows = tuple(BatchRecord(**json.loads(line)) for line in path.read_text().splitlines() if line)
    result = validate_batch_plan(rows)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

