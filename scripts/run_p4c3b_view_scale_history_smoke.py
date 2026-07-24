#!/usr/bin/env python3
"""Run the offline P4-C3B-M3 view and scale-history smoke."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from semantic3d.method_completion.view_scale_smoke import (  # noqa: E402
    run_view_scale_history_smoke,
)


def parse_args() -> argparse.Namespace:
    """Parse deterministic smoke paths."""

    parser = argparse.ArgumentParser(
        description="Audit metric object view and same-track scale history."
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs/p4c3b_view_scale_history_v1.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "outputs/p4c3b_view_scale_history"),
    )
    return parser.parse_args()


def main() -> int:
    """Run the smoke and print its validation status."""

    args = parse_args()
    result = run_view_scale_history_smoke(
        config_path=args.config,
        output_dir=args.output_dir,
        project_root=PROJECT_ROOT,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["all_frozen_hashes_unchanged"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
