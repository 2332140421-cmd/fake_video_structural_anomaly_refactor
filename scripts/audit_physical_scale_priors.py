#!/usr/bin/env python3
"""Audit physical scale-prior candidates without using evaluation videos."""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
AUDIT_STATUSES = {
    "reliable_single",
    "conditional_multimodal",
    "pose_sensitive",
    "insufficient_source",
    "unsupported",
}
SOURCE_FIELDS = [
    "label",
    "source_id",
    "title",
    "publisher",
    "url",
    "source_type",
    "used_for_interval",
    "physical_dimensions",
    "source_unit",
    "conversion_to_m",
    "note",
]
AUDIT_FIELDS = [
    "label",
    "characteristic_dimension",
    "dimension_definition",
    "unit",
    "min",
    "max",
    "estimation_method",
    "source_count",
    "traceable_sources",
    "same_physical_dimension",
    "unit_normalized",
    "supported_interval_basis",
    "severe_multimodality",
    "pose_sensitivity",
    "interval_ratio",
    "interval_not_too_broad",
    "sufficient_information",
    "audit_status",
    "reliable",
    "audit_reason",
    "reviewed_at",
    "prior_version",
]


def _ensure_project_environment() -> None:
    """Re-execute with the project-local interpreter when needed."""

    if not PROJECT_PYTHON.exists():
        raise RuntimeError(f"Project environment is missing: {PROJECT_PYTHON}")
    if Path(sys.executable).resolve() != PROJECT_PYTHON.resolve():
        os.execv(str(PROJECT_PYTHON), [str(PROJECT_PYTHON), *sys.argv])


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Audit sourced physical scale priors.")
    parser.add_argument(
        "--candidate_config",
        default=str(PROJECT_ROOT / "configs/scale_priors_candidates.yaml"),
    )
    parser.add_argument(
        "--source_report",
        default=str(PROJECT_ROOT / "outputs/results/scale_prior_source_report.csv"),
    )
    parser.add_argument(
        "--audit_report",
        default=str(PROJECT_ROOT / "outputs/results/scale_prior_audit_report.csv"),
    )
    return parser.parse_args()


def load_candidates(path: Path) -> dict[str, Any]:
    """Load a candidate YAML and validate its top-level separation metadata."""

    import yaml

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict) or not isinstance(data.get("scale_prior_candidates"), dict):
        raise ValueError("Candidate config requires scale_prior_candidates mapping.")
    statement = str(data.get("independence_statement", ""))
    if "No projected" not in statement or "pilot" not in statement:
        raise ValueError("Candidate config must contain an explicit independence statement.")
    return data


def build_source_rows(data: Mapping[str, Any]) -> list[dict[str, object]]:
    """Flatten traceable source metadata for human and machine review."""

    rows: list[dict[str, object]] = []
    for label, candidate in data["scale_prior_candidates"].items():
        for source in candidate.get("sources", []):
            rows.append(
                {
                    "label": label,
                    "source_id": source.get("source_id", ""),
                    "title": source.get("title", ""),
                    "publisher": source.get("publisher", ""),
                    "url": source.get("url", ""),
                    "source_type": source.get("source_type", ""),
                    "used_for_interval": source.get("used_for_interval", False),
                    "physical_dimensions": source.get("dimensions", ""),
                    "source_unit": source.get("unit", ""),
                    "conversion_to_m": source.get("conversion_to_m", ""),
                    "note": source.get("note", ""),
                }
            )
    return rows


def _source_checks(candidate: Mapping[str, Any]) -> tuple[bool, bool, bool, int]:
    """Check traceability, physical dimension consistency, and publisher support."""

    sources = list(candidate.get("sources", []))
    traceable = bool(sources) and all(
        str(source.get("url", "")).startswith("https://")
        and str(source.get("title", "")).strip()
        and str(source.get("publisher", "")).strip()
        for source in sources
    )
    interval_sources = [source for source in sources if bool(source.get("used_for_interval"))]
    dimensions = {str(source.get("dimensions", "")) for source in interval_sources}
    same_dimension = bool(interval_sources) and len(dimensions) == 1
    conversions_valid = bool(interval_sources) and all(
        isinstance(source.get("conversion_to_m"), (int, float))
        and float(source["conversion_to_m"]) > 0
        for source in interval_sources
    )
    publishers = len({str(source.get("publisher", "")) for source in interval_sources})
    return traceable, same_dimension, conversions_valid, publishers


def audit_candidate(
    label: str,
    candidate: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> dict[str, object]:
    """Apply deterministic reliability rules to one physical prior candidate."""

    low = candidate.get("min")
    high = candidate.get("max")
    has_interval = isinstance(low, (int, float)) and isinstance(high, (int, float))
    valid_interval = bool(has_interval and float(low) > 0 and float(high) > float(low))
    sources = list(candidate.get("sources", []))
    declared_count = int(candidate.get("source_count", -1))
    source_count_matches = declared_count == len(sources)
    traceable, same_dimension, conversions_valid, publisher_count = _source_checks(candidate)
    unit_normalized = candidate.get("unit") == policy.get("allowed_unit", "m") and conversions_valid
    interval_basis = str(candidate.get("interval_basis", "none"))
    supported_basis = interval_basis in {
        "sample_statistics",
        "official_standard",
        "explicit_specifications",
    }
    multimodal = bool(candidate.get("multimodal_warning", False))
    pose = str(candidate.get("pose_sensitivity", "high"))
    ratio = float(high) / float(low) if valid_interval else math.nan
    not_too_broad = bool(
        valid_interval and ratio <= float(policy.get("max_interval_ratio", 2.5))
    )

    source_types = {str(source.get("source_type", "")) for source in sources}
    if interval_basis in {"sample_statistics", "official_standard"}:
        source_support = len(sources) >= int(policy.get("accepted_statistical_source_count", 1))
    elif interval_basis == "explicit_specifications":
        source_support = (
            len(sources) >= int(policy.get("minimum_product_spec_sources", 3))
            and publisher_count >= int(policy.get("minimum_product_spec_publishers", 2))
        )
    else:
        source_support = False
    sufficient = all(
        [
            valid_interval,
            source_count_matches,
            traceable,
            same_dimension,
            unit_normalized,
            supported_basis,
            source_support,
            bool(source_types),
            not_too_broad,
        ]
    )

    reasons: list[str] = []
    if not valid_interval or not sources:
        status = "unsupported"
        reasons.append("no sourced positive physical interval")
    elif not source_count_matches or not traceable or not same_dimension or not unit_normalized:
        status = "unsupported"
        reasons.append("source traceability, dimension, or unit validation failed")
    elif not supported_basis or not source_support or not not_too_broad:
        status = "insufficient_source"
        reasons.append("interval basis, source diversity, or interval width is insufficient")
    elif multimodal:
        status = "conditional_multimodal"
        reasons.append(str(candidate.get("multimodal_reason", "severe category multimodality")))
    elif pose == "high":
        status = "pose_sensitive"
        reasons.append("projected area is highly pose/viewpoint dependent")
    elif sufficient:
        status = "reliable_single"
        reasons.append("all strict source, dimension, unit, support, pose, and width checks passed")
    else:
        status = "insufficient_source"
        reasons.append("insufficient information for strict use")

    if status not in AUDIT_STATUSES:
        raise AssertionError(f"Unexpected audit status for {label}: {status}")
    return {
        "label": label,
        "characteristic_dimension": candidate.get("characteristic_dimension", ""),
        "dimension_definition": candidate.get("dimension_definition", ""),
        "unit": candidate.get("unit", ""),
        "min": low if low is not None else "",
        "max": high if high is not None else "",
        "estimation_method": candidate.get("estimation_method", ""),
        "source_count": declared_count,
        "traceable_sources": traceable,
        "same_physical_dimension": same_dimension,
        "unit_normalized": unit_normalized,
        "supported_interval_basis": supported_basis,
        "severe_multimodality": multimodal,
        "pose_sensitivity": pose,
        "interval_ratio": ratio,
        "interval_not_too_broad": not_too_broad,
        "sufficient_information": sufficient,
        "audit_status": status,
        "reliable": status == "reliable_single",
        "audit_reason": "; ".join(reasons),
        "reviewed_at": candidate.get("reviewed_at", ""),
        "prior_version": candidate.get("prior_version", ""),
    }


def build_audit_rows(data: Mapping[str, Any]) -> list[dict[str, object]]:
    """Audit all candidates in stable label order."""

    return [
        audit_candidate(label, candidate, data.get("audit_policy", {}))
        for label, candidate in sorted(data["scale_prior_candidates"].items())
    ]


def save_csv(rows: list[dict[str, object]], path: Path, fields: list[str]) -> None:
    """Save an audit/source report with stable fields."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> None:
    """Generate source and deterministic audit reports."""

    args = parse_args()
    data = load_candidates(Path(args.candidate_config))
    source_rows = build_source_rows(data)
    audit_rows = build_audit_rows(data)
    save_csv(source_rows, Path(args.source_report), SOURCE_FIELDS)
    save_csv(audit_rows, Path(args.audit_report), AUDIT_FIELDS)
    counts: dict[str, int] = {}
    for row in audit_rows:
        status = str(row["audit_status"])
        counts[status] = counts.get(status, 0) + 1
    print(f"Saved source report: {args.source_report}")
    print(f"Saved audit report: {args.audit_report}")
    print(f"Audit status counts: {counts}")
    print("No evaluation video data was read by this audit.")


if __name__ == "__main__":
    _ensure_project_environment()
    main()
