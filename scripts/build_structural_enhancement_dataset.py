#!/usr/bin/env python3
"""Build the versioned, label-isolated P4-B structural dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from semantic3d.dataset_builder.pipeline import StructuralEnhancementDatasetBuilder  # noqa: E402


def _builder_class(config_path: str):
    """Select the versioned builder without changing the legacy P4-B path."""

    payload = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    profile = str(payload.get("dataset", {}).get("pipeline_profile", "p4b_v1"))
    if profile == "p4b5_full_observation_v1":
        from semantic3d.dataset_builder.p4b5_pipeline import (  # noqa: PLC0415
            P4B5StructuralEnhancementDatasetBuilder,
        )

        return P4B5StructuralEnhancementDatasetBuilder
    return StructuralEnhancementDatasetBuilder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="P4-B YAML configuration")
    parser.add_argument("--video-id", help="Optional source stem or stable video_id")
    parser.add_argument("--stage", help="Run all dependencies through this stage")
    parser.add_argument("--resume", action="store_true", help="Reuse complete content-addressed stage caches")
    parser.add_argument("--force-stage", help="Force this stage and all transitive downstream stages")
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true", help="Print a plan without writing or loading models")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    builder = _builder_class(args.config)(
        args.config,
        device=args.device,
        num_workers=args.num_workers,
        selected_video_id=args.video_id,
    )
    if args.dry_run:
        print(json.dumps({"dry_run": True, "plan": builder.dry_run(args.stage)}, indent=2))
        return 0
    result = builder.run(target_stage=args.stage, resume=args.resume, force_stage=args.force_stage)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
