#!/usr/bin/env python3
"""Freeze an audited candidate catalog into an immutable strict prior version."""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"


def _ensure_project_environment() -> None:
    """Re-execute with the project-local interpreter when needed."""

    if not PROJECT_PYTHON.exists():
        raise RuntimeError(f"Project environment is missing: {PROJECT_PYTHON}")
    if Path(sys.executable).resolve() != PROJECT_PYTHON.resolve():
        os.execv(str(PROJECT_PYTHON), [str(PROJECT_PYTHON), *sys.argv])


_ensure_project_environment()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml  # noqa: E402

from scripts.audit_physical_scale_priors import (  # noqa: E402
    build_audit_rows,
    load_candidates,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Freeze audited physical priors.")
    parser.add_argument(
        "--candidate_config",
        default=str(PROJECT_ROOT / "configs/scale_priors_candidates.yaml"),
    )
    parser.add_argument(
        "--source_report",
        default="outputs/results/scale_prior_source_report.csv",
    )
    parser.add_argument(
        "--audit_report",
        default="outputs/results/scale_prior_audit_report.csv",
    )
    parser.add_argument(
        "--output_config",
        default=str(PROJECT_ROOT / "configs/scale_priors_strict_v1.yaml"),
    )
    parser.add_argument("--prior_version", default="strict_physical_v1")
    parser.add_argument("--prior_created_at", default="2026-07-14T00:00:00+08:00")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    """Return a stable SHA-256 hash for one file."""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _audit_by_label(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    """Index deterministic audit rows by normalized candidate label."""

    return {str(row["label"]): row for row in rows}


def build_frozen_config(
    candidate_path: Path,
    source_report_path: str,
    audit_report_path: str,
    prior_version: str,
    prior_created_at: str,
) -> dict[str, Any]:
    """Build a frozen config solely from candidate metadata and audit rules."""

    data = load_candidates(candidate_path)
    audit_rows = build_audit_rows(data)
    audits = _audit_by_label(audit_rows)
    priors: dict[str, dict[str, Any]] = {}
    excluded: dict[str, dict[str, Any]] = {}
    for label, candidate in sorted(data["scale_prior_candidates"].items()):
        audit = audits[label]
        item = {
            "characteristic_dimension": candidate.get("characteristic_dimension"),
            "dimension_definition": candidate.get("dimension_definition"),
            "unit": candidate.get("unit"),
            "min": candidate.get("min"),
            "max": candidate.get("max"),
            "estimation_method": candidate.get("estimation_method"),
            "source_count": candidate.get("source_count", 0),
            "sources": candidate.get("sources", []),
            "audit_status": audit["audit_status"],
            "reliable": audit["audit_status"] == "reliable_single",
            "reliability_reason": audit["audit_reason"],
            "pose_sensitivity": candidate.get("pose_sensitivity"),
            "multimodal_warning": candidate.get("multimodal_warning", False),
            "reviewed_at": candidate.get("reviewed_at"),
            "prior_version": prior_version,
        }
        if candidate.get("min") is not None and candidate.get("max") is not None:
            priors[label] = item
        else:
            excluded[label] = item

    known = set(priors) | set(excluded)
    aliases = {
        alias: target
        for alias, target in data.get("aliases", {}).items()
        if target in known
    }
    try:
        candidate_config_path = str(candidate_path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        candidate_config_path = str(candidate_path)
    return {
        "metadata": {
            "prior_version": prior_version,
            "prior_created_at": prior_created_at,
            "source_report_path": source_report_path,
            "audit_report_path": audit_report_path,
            "candidate_config_path": candidate_config_path,
            "candidate_file_hash": sha256_file(candidate_path),
            "strict_enabled_status": "reliable_single",
            "prior_source": "physical",
            "independent_of_evaluation_videos": True,
            "characteristic_dimension": data["global_characteristic_dimension"],
        },
        "scale_priors": priors,
        "excluded_priors": excluded,
        "aliases": aliases,
    }


def freeze_config(data: Mapping[str, Any], output_path: Path) -> None:
    """Write a new frozen version and refuse to overwrite an existing file."""

    if output_path.exists():
        raise FileExistsError(
            f"Frozen prior already exists and will not be overwritten: {output_path}. "
            "Create a new v2 path/version instead."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(dict(data), sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )


def _validate_report_exists(path: Path, expected_header: str) -> None:
    """Require the independently generated report before freezing."""

    if not path.exists():
        raise FileNotFoundError(f"Required report does not exist: {path}")
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if expected_header not in (reader.fieldnames or []):
            raise ValueError(f"Unexpected report schema for {path}")


def main() -> None:
    """Freeze a new strict physical prior version."""

    args = parse_args()
    candidate_path = Path(args.candidate_config)
    source_report = PROJECT_ROOT / args.source_report
    audit_report = PROJECT_ROOT / args.audit_report
    _validate_report_exists(source_report, "source_id")
    _validate_report_exists(audit_report, "audit_status")
    frozen = build_frozen_config(
        candidate_path,
        args.source_report,
        args.audit_report,
        args.prior_version,
        args.prior_created_at,
    )
    freeze_config(frozen, Path(args.output_config))
    enabled = sum(item["reliable"] for item in frozen["scale_priors"].values())
    print(f"Frozen strict prior: {args.output_config}")
    print(f"prior_version: {args.prior_version}")
    print(f"reliable_single entries enabled: {enabled}")
    print("This command never reads evaluation observations or video labels.")


if __name__ == "__main__":
    main()
