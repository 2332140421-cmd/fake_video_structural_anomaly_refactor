#!/usr/bin/env python3
"""Plan source-group-preserving server batches without executing them."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from semantic3d.batch_execution.batch_planner import plan_batches  # noqa: E402
from semantic3d.batch_execution.batch_report import write_batch_plan  # noqa: E402
from semantic3d.runtime.runtime_config import load_runtime_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-config", default="configs/runtime/local_wsl.yaml")
    parser.add_argument("--formal-split-plan", default="outputs/p4c2_formal_data_readiness/formal_split_plan.jsonl")
    parser.add_argument("--p4c2-metadata", default="outputs/p4c2_formal_data_readiness/build_metadata.json")
    parser.add_argument("--output-root", default="outputs/p4c3a_batch_plan")
    args = parser.parse_args()
    runtime = load_runtime_config(PROJECT_ROOT / args.runtime_config)
    metadata = json.loads((PROJECT_ROOT / args.p4c2_metadata).read_text(encoding="utf-8"))
    source_rows = [
        json.loads(line)
        for line in (PROJECT_ROOT / args.formal_split_plan).read_text(encoding="utf-8").splitlines()
        if line
    ]
    for row in source_rows:
        row["dataset_id"] = row["dataset_name"]
        row.setdefault("planned_input_bytes", 0)
        row.setdefault("planned_temporary_bytes", 0)
        row.setdefault("planned_output_bytes", 0)
    plans = plan_batches(
        source_rows,
        input_manifest_sha256=metadata["p4c2_readiness_manifest_sha256"],
        runtime_profile=runtime.runtime_profile,
        batch_storage_limit=runtime.batch_storage_limit,
        software_commit="",
        protocol_sha256=metadata["protocol_sha256"],
        manifest_sha256=metadata["p4c1_manifest_sha256"],
    )
    result = write_batch_plan(PROJECT_ROOT / args.output_root, plans)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["validation"]["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

