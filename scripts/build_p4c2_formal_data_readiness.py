#!/usr/bin/env python3
"""Build P4-C2 registries and readiness artifacts without formal data processing."""

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

from semantic3d.dataset_builder.writer import atomic_write_json  # noqa: E402
from semantic3d.experiment_protocol.p4c2_builder import (  # noqa: E402
    build_p4c2_readiness,
    readiness_manifest_sha256,
)
from semantic3d.experiment_protocol.p4c2_report import write_p4c2_artifacts  # noqa: E402
from semantic3d.experiment_protocol.p4c2_validation import (  # noqa: E402
    validate_p4c2_artifacts,
)


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def build_and_write(config_path: str | Path, output_root: str | Path | None = None) -> dict[str, object]:
    """Build twice, write deterministic artifacts, and run independent validation."""

    config_file = _resolve(config_path).resolve()
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    target = _resolve(output_root or config["output_root"]).resolve()
    first = build_p4c2_readiness(config_file, project_root=PROJECT_ROOT)
    second = build_p4c2_readiness(config_file, project_root=PROJECT_ROOT)
    first_hash = readiness_manifest_sha256(first)
    second_hash = readiness_manifest_sha256(second)
    if first_hash != second_hash:
        raise RuntimeError("P4-C2 repeated in-memory builds produced different hashes")
    metadata = write_p4c2_artifacts(
        target,
        first,
        deterministic_rebuild_sha256=second_hash,
    )
    validation = validate_p4c2_artifacts(
        target,
        config_file,
        project_root=PROJECT_ROOT,
    )
    atomic_write_json(target / "validation_report.json", validation)
    return {
        "output_root": str(target),
        "dataset_count": metadata["dataset_count"],
        "lineage_record_count": metadata["lineage_record_count"],
        "formal_split_record_count": metadata["formal_split_record_count"],
        "task_eligibility_record_count": metadata["task_eligibility_record_count"],
        "ready_for_formal_batch_build": metadata["ready_for_formal_batch_build"],
        "blockers": validation["blockers"],
        "formal_split_counts": validation["formal_split_counts"],
        "p4c2_readiness_manifest_sha256": first_hash,
        "deterministic_rebuild_matches": first_hash == second_hash,
        "validation_valid": validation["valid"],
        "downloads_performed": False,
        "model_inference_performed": False,
        "formal_build_started": False,
        "model_training_performed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/p4c2_formal_data_readiness_v1.yaml")
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()
    result = build_and_write(args.config, args.output_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["validation_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

