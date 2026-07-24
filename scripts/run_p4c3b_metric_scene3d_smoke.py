#!/usr/bin/env python3
"""Run P4-C3B-M2 from saved M1 depth arrays and formal instance masks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from semantic3d.metric_scene3d.smoke import run_metric_scene3d_smoke


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/p4c3b_metric_scene3d_v1.yaml",
        help="M2 smoke configuration path.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/p4c3b_metric_scene3d",
        help="Audit output directory.",
    )
    return parser.parse_args()


def main() -> None:
    """Execute the offline M2 smoke and print its validation status."""

    args = parse_args()
    result = run_metric_scene3d_smoke(
        config_path=PROJECT_ROOT / args.config,
        output_dir=PROJECT_ROOT / args.output_dir,
        project_root=PROJECT_ROOT,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
