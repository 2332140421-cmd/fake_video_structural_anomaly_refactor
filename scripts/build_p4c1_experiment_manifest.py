#!/usr/bin/env python3
"""Build the frozen P4-C1 clip manifest without training or fitting."""

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
from semantic3d.experiment_protocol.manifest_builder import (  # noqa: E402
    build_p4c1_manifest,
    manifest_sha256,
    validate_manifest_artifacts,
)
from semantic3d.experiment_protocol.manifest_report import (  # noqa: E402
    write_manifest_artifacts,
)


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def build_and_write(config_path: str | Path, output_root: str | Path | None = None) -> dict[str, object]:
    """Build twice, persist deterministic artifacts, and validate them."""

    config_file = _resolve(config_path).resolve()
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    target = _resolve(output_root or config["output_root"]).resolve()
    first = build_p4c1_manifest(config_file, project_root=PROJECT_ROOT)
    second = build_p4c1_manifest(config_file, project_root=PROJECT_ROOT)
    first_hash = manifest_sha256(first)
    second_hash = manifest_sha256(second)
    if first_hash != second_hash:
        raise RuntimeError("Deterministic P4-C1 rebuild produced a different manifest hash")
    metadata = write_manifest_artifacts(
        target,
        first,
        required_modalities=config["availability"]["required_for_usable"],
        deterministic_rebuild_sha256=second_hash,
    )
    validation = validate_manifest_artifacts(
        target,
        config_file,
        project_root=PROJECT_ROOT,
    )
    atomic_write_json(target / "validation_report.json", validation)
    result = {
        "output_root": str(target),
        "sample_count": metadata["sample_count"],
        "usable_count": metadata["usable_count"],
        "excluded_count": metadata["excluded_count"],
        "manifest_sha256": metadata["manifest_sha256"],
        "protocol_sha256": metadata["protocol_sha256"],
        "deterministic_rebuild_matches": metadata["deterministic_rebuild_matches"],
        "leakage_error_count": metadata["leakage_error_count"],
        "leakage_warning_count": metadata["leakage_warning_count"],
        "validation_valid": validation["valid"],
        "model_training_performed": False,
        "statistical_fitting_performed": False,
        "threshold_selection_performed": False,
        "classification_performance_computed": False,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/p4c1_experiment_manifest_v1.yaml",
        help="P4-C1 manifest configuration",
    )
    parser.add_argument("--output-root", default=None, help="Optional output override")
    args = parser.parse_args()
    result = build_and_write(args.config, args.output_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["validation_valid"] and result["leakage_error_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
