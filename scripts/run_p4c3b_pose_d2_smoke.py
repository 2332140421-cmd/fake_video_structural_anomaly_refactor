#!/usr/bin/env python3
"""Run the offline P4-C3B-M4 pose and D2 smoke."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from semantic3d.pose_d2.smoke import run_pose_d2_smoke


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/p4c3b_pose_d2_smoke_v1.yaml",
        help="Project-relative M4 smoke configuration.",
    )
    args = parser.parse_args()
    result = run_pose_d2_smoke(PROJECT_ROOT, args.config)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
