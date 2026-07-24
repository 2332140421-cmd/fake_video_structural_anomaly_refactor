#!/usr/bin/env python3
"""Build an A2 evidence JSONL from precomputed M6 branch audit tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from semantic3d.minimal_training.evidence_bridge import (
    build_a2_evidence_manifest,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert M6 branch contribution/availability audits to the "
            "P4-C3C-A2 evidence manifest without reading fusion scores."
        )
    )
    parser.add_argument("--formal-manifest", required=True, type=Path)
    parser.add_argument("--branch-contribution-manifest", required=True, type=Path)
    parser.add_argument("--branch-availability-manifest", required=True, type=Path)
    parser.add_argument("--sample-mapping-manifest", type=Path)
    parser.add_argument(
        "--feature-contract",
        type=Path,
        default=Path("configs/p4c3c_a2_m6_feature_contract_v1.yaml"),
    )
    parser.add_argument(
        "--bridge-config",
        type=Path,
        default=Path("configs/p4c3c_a3_evidence_bridge_v1.yaml"),
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = build_a2_evidence_manifest(
        formal_manifest=args.formal_manifest,
        branch_contribution_manifest=args.branch_contribution_manifest,
        branch_availability_manifest=args.branch_availability_manifest,
        sample_mapping_manifest=args.sample_mapping_manifest,
        feature_contract=args.feature_contract,
        bridge_config=args.bridge_config,
        output_path=args.output,
    )
    print(
        json.dumps(
            {
                "output_path": str(result.output_path),
                "sample_count": result.sample_count,
                "output_sha256": result.output_sha256,
                "branch_order": list(result.branch_order),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
