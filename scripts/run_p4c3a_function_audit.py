#!/usr/bin/env python3
"""Run the read-only P4-C3A-V six-video function audit."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from semantic3d.function_audit import build_function_audit  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse audit paths; no provider or performance options are exposed."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset_root",
        default="outputs/structural_enhancement_dataset/p4b5_six_video_full_observation",
    )
    parser.add_argument(
        "--output_dir", default="outputs/p4c3a_function_audit"
    )
    parser.add_argument(
        "--archive_root",
        default=os.environ.get("SEMANTIC3D_ARCHIVE_ROOT", "outputs/archive"),
    )
    return parser.parse_args()


def main() -> int:
    """Build the audit and print only its validation booleans."""

    args = parse_args()
    result = build_function_audit(
        ROOT,
        args.output_dir,
        dataset_root=args.dataset_root,
        archive_root=args.archive_root,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
