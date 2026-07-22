#!/usr/bin/env python3
"""Inspect recoverable P4 batch state without modifying it."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from semantic3d.batch_execution.batch_state_store import BatchStateStore  # noqa: E402
from semantic3d.batch_execution.checkpoint_manager import recovery_action  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch_id")
    parser.add_argument("--state-root", default="outputs/p4c3a_batch_state")
    args = parser.parse_args()
    record = BatchStateStore(PROJECT_ROOT / args.state_root).load(args.batch_id)
    print(json.dumps({"record": record.to_dict(), "recovery": recovery_action(record)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

