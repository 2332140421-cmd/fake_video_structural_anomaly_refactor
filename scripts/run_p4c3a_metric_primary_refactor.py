#!/usr/bin/env python3
"""Build the P4-C3A-MD2 synthetic and six-video read-only audit outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from semantic3d.method_completion.md2_audit import build_md2_audit  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/p4c3a_metric_primary_refactor",
    )
    parser.add_argument(
        "--dataset_root",
        type=Path,
        default=PROJECT_ROOT / "outputs/structural_enhancement_dataset/p4b5_six_video_full_observation",
    )
    parser.add_argument(
        "--strict_result_path",
        type=Path,
        default=Path("/mnt/e/fake_video_structural_anomaly_archive/outputs/evaluation/rsd_strict_v2/per_pair_rsd_details.csv"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build_md2_audit(
        PROJECT_ROOT,
        args.output_dir,
        dataset_root=args.dataset_root,
        strict_result_path=args.strict_result_path,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
