#!/usr/bin/env python3
"""Generate P4-C3A-M method-completion and pre-freeze audit artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from semantic3d.method_completion.audit import build_method_completion_audit


def parse_args() -> argparse.Namespace:
    """Parse deterministic output arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/p4c3a_method_completion"),
    )
    return parser.parse_args()


def main() -> int:
    """Build the audit without executing learned providers."""

    args = parse_args()
    result = build_method_completion_audit(PROJECT_ROOT, args.output_dir)
    print(f"Saved P4-C3A-M audit to {args.output_dir}")
    for key in (
        "static_branch_complete",
        "relative_scale_depth_branch_complete",
        "absolute_scale_branch_status",
        "cross_frame_scale_stability_complete",
        "d2_synthetic_verified",
        "d2_six_video_verified",
        "d3_code_status",
        "localization_evidence_mapping_complete",
        "ready_for_git_freeze",
    ):
        print(f"{key}={result[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
