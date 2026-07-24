#!/usr/bin/env python3
"""Run the P4-C3B-M1 offline UniDepthV2 quality smoke."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from semantic3d.method_completion.metric_provider_smoke import (  # noqa: E402
    run_metric_provider_smoke,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Run a small offline metric-depth/intrinsics smoke without scoring labels."
    )
    parser.add_argument(
        "--config",
        default="configs/p4c3b_metric_provider_smoke_v1.yaml",
        help="P4-C3B-M1 smoke configuration.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/p4c3b_metric_provider_smoke",
        help="Output directory for manifests and quality audits.",
    )
    return parser.parse_args()


def main() -> int:
    """Execute the configured smoke and print its validation status."""

    args = parse_args()
    try:
        report = run_metric_provider_smoke(
            config_path=PROJECT_ROOT / args.config,
            output_dir=PROJECT_ROOT / args.output_dir,
        )
    except Exception as exc:
        print(f"P4-C3B-M1 smoke failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

