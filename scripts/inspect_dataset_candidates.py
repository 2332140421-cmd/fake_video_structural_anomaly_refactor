#!/usr/bin/env python3
"""Validate and summarize a metadata-only dataset candidate registry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from semantic3d.dataset_builder.candidate_audit import (
    load_candidate_registry,
    summarize_candidate_registry,
)

DEFAULT_REGISTRY = "configs/data_registry/p4c3c_a3b0_dataset_candidates_v1.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a frozen candidate registry without network access, media "
            "download, dataset extraction, inference, or training."
        )
    )
    parser.add_argument("--registry", default=DEFAULT_REGISTRY)
    parser.add_argument(
        "--output",
        help="Optional JSON summary path; omitted by default for a read-only run.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = load_candidate_registry(args.registry)
    summary = summarize_candidate_registry(payload)
    rendered = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
