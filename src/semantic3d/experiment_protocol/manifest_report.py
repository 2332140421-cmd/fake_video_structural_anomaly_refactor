"""Deterministic writers and human-readable reports for P4-C1."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from collections import Counter
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Iterable, Mapping

from semantic3d.dataset_builder.writer import atomic_write_bytes, atomic_write_json

from .manifest_schema import (
    ALLOWED_SPLITS,
    ExperimentSampleRecord,
    LeakageFinding,
    ManifestBuildResult,
    SampleAvailability,
)


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize a value deterministically for hashing."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def manifest_jsonl_bytes(records: Iterable[ExperimentSampleRecord]) -> bytes:
    """Return canonical JSONL bytes in stable sample order."""

    rows = sorted(records, key=lambda row: row.sample_id)
    return b"".join(canonical_json_bytes(row.to_dict()) + b"\n" for row in rows)


def _csv_bytes(rows: Iterable[Mapping[str, Any]], columns: Iterable[str]) -> bytes:
    handle = io.StringIO(newline="")
    names = tuple(columns)
    writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                name: json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                if isinstance(value, (dict, list, tuple))
                else value
                for name, value in row.items()
            }
        )
    return handle.getvalue().encode("utf-8")


def split_summary_rows(records: Iterable[ExperimentSampleRecord]) -> list[dict[str, Any]]:
    """Summarize frozen sample counts without classification metrics."""

    rows = list(records)
    output = []
    for split in ("train", "validation", "test", "official_conflict"):
        members = [row for row in rows if row.split == split]
        output.append(
            {
                "split": split,
                "sample_count": len(members),
                "usable_count": sum(row.usable for row in members),
                "excluded_count": sum(not row.usable for row in members),
                "source_video_count": len({row.source_video_id for row in members}),
                "source_group_count": len({row.source_group_id for row in members}),
                "real_sample_count": sum(row.authenticity_label == 0 for row in members),
                "fake_sample_count": sum(row.authenticity_label == 1 for row in members),
                "unknown_label_count": sum(row.authenticity_label is None for row in members),
            }
        )
    return output


def exclusion_summary_rows(records: Iterable[ExperimentSampleRecord]) -> list[dict[str, Any]]:
    """Count every exclusion reason, including multi-cause samples."""

    counts: Counter[str] = Counter()
    for row in records:
        if not row.usable:
            counts.update(reason for reason in row.exclusion_reason.split("|") if reason)
    return [
        {"exclusion_reason": reason, "sample_count": count}
        for reason, count in sorted(counts.items())
    ]


def availability_summary_rows(
    availability: Iterable[SampleAvailability],
    required_modalities: Iterable[str],
) -> list[dict[str, Any]]:
    """Aggregate technical data availability by modality."""

    rows = list(availability)
    required = set(required_modalities)
    probes = {
        "video": lambda row: row.video_exists and row.video_readable and row.video_hash_matches,
        "frames": lambda row: row.decoded_frame_count >= row.expected_frame_count,
        "objects": lambda row: row.valid_object_count > 0,
        "depth": lambda row: row.valid_depth_count > 0,
        "camera": lambda row: row.camera_observation_count > 0,
        "pose": lambda row: row.valid_pose_count > 0,
        "tracks": lambda row: row.valid_track_point_count > 0,
        "semantic3d": lambda row: row.valid_semantic3d_count > 0,
        "camera_identity": lambda row: row.camera_identity_available,
    }
    counts = {
        "video": lambda row: int(row.video_exists and row.video_readable and row.video_hash_matches),
        "frames": lambda row: row.decoded_frame_count,
        "objects": lambda row: row.valid_object_count,
        "depth": lambda row: row.valid_depth_count,
        "camera": lambda row: row.camera_observation_count,
        "pose": lambda row: row.valid_pose_count,
        "tracks": lambda row: row.valid_track_point_count,
        "semantic3d": lambda row: row.valid_semantic3d_count,
        "camera_identity": lambda row: int(row.camera_identity_available),
    }
    output = []
    for name in probes:
        available = sum(bool(probes[name](row)) for row in rows)
        output.append(
            {
                "modality": name,
                "sample_count": len(rows),
                "available_sample_count": available,
                "missing_sample_count": len(rows) - available,
                "availability_ratio": available / len(rows) if rows else 0.0,
                "total_valid_observations": sum(int(counts[name](row)) for row in rows),
                "required_for_usable": name in required,
            }
        )
    return output


def render_manifest_report(
    result: ManifestBuildResult,
    manifest_sha256: str,
    required_modalities: Iterable[str],
) -> str:
    """Render a deterministic Markdown integrity report."""

    records = list(result.records)
    splits = split_summary_rows(records)
    exclusions = exclusion_summary_rows(records)
    availability = availability_summary_rows(result.availability, required_modalities)
    lines = [
        "# P4-C1 Experiment Manifest Report",
        "",
        "This report freezes sample identity, split assignment, input availability, and leakage audit only.",
        "No model training, parameter fitting, threshold selection, or authenticity performance metric was run.",
        "",
        "## Reproducibility",
        "",
        f"- manifest schema: `{records[0].manifest_schema_version if records else 'unknown'}`",
        f"- protocol SHA-256: `{result.protocol_sha256}`",
        f"- P4-C0 config SHA-256: `{result.p4c0_config_sha256}`",
        f"- P4-C1 config SHA-256: `{result.config_sha256}`",
        f"- source label manifest SHA-256: `{result.source_manifest_sha256}`",
        f"- manifest SHA-256: `{manifest_sha256}`",
        f"- structural dataset ID: `{result.structural_dataset_id}`",
        "",
        "## Split Freeze",
        "",
        "| split | samples | usable | excluded | videos | groups |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| {row['split']} | {row['sample_count']} | {row['usable_count']} | {row['excluded_count']} | {row['source_video_count']} | {row['source_group_count']} |"
        for row in splits
    )
    lines.extend(["", "## Data Availability", "", "| modality | available | missing | required |", "|---|---:|---:|---|"])
    lines.extend(
        f"| {row['modality']} | {row['available_sample_count']} | {row['missing_sample_count']} | {str(row['required_for_usable']).lower()} |"
        for row in availability
    )
    lines.extend(["", "## Exclusions", ""])
    if exclusions:
        lines.extend(["| reason | samples |", "|---|---:|"])
        lines.extend(f"| {row['exclusion_reason']} | {row['sample_count']} |" for row in exclusions)
    else:
        lines.append("No technical sample exclusions were recorded.")
    lines.extend(
        [
            "",
            "## Leakage Audit",
            "",
            f"- unique source videos: {result.leakage_summary['unique_source_video_count']}",
            f"- unique source groups: {result.leakage_summary['unique_source_group_count']}",
            f"- error findings: {result.leakage_summary['error_count']}",
            f"- warning findings: {result.leakage_summary['warning_count']}",
            f"- unresolved original/derivative identities: {result.leakage_summary['unresolved_original_derivative_identity_count']}",
            f"- cross-split leakage detected: {str(result.leakage_summary['cross_split_leakage_detected']).lower()}",
            "",
            "Unresolved source identity is retained as a provenance warning. It is not silently treated as proof of independence.",
            "Overlapping clips inherit the source-video split and are not counted as independent source videos.",
            "",
        ]
    )
    return "\n".join(lines)


def write_manifest_artifacts(
    output_root: str | Path,
    result: ManifestBuildResult,
    *,
    required_modalities: Iterable[str],
    deterministic_rebuild_sha256: str,
) -> dict[str, Any]:
    """Write all required P4-C1 artifacts atomically."""

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    records = sorted(result.records, key=lambda row: row.sample_id)
    jsonl_payload = manifest_jsonl_bytes(records)
    manifest_sha256 = hashlib.sha256(jsonl_payload).hexdigest()
    atomic_write_bytes(root / "experiment_manifest.jsonl", jsonl_payload)
    record_rows = [row.to_dict() for row in records]
    columns = tuple(field.name for field in fields(ExperimentSampleRecord))
    atomic_write_bytes(root / "experiment_manifest.csv", _csv_bytes(record_rows, columns))

    excluded = [row for row in record_rows if not row["usable"]]
    atomic_write_bytes(root / "excluded_samples.csv", _csv_bytes(excluded, columns))
    excluded_jsonl = b"".join(canonical_json_bytes(row) + b"\n" for row in excluded)
    atomic_write_bytes(root / "excluded_samples.jsonl", excluded_jsonl)

    split_rows = split_summary_rows(records)
    atomic_write_bytes(root / "split_summary.csv", _csv_bytes(split_rows, tuple(split_rows[0])))
    exclusion_rows = exclusion_summary_rows(records)
    exclusion_columns = ("exclusion_reason", "sample_count")
    atomic_write_bytes(root / "exclusion_summary.csv", _csv_bytes(exclusion_rows, exclusion_columns))
    availability_rows = availability_summary_rows(result.availability, required_modalities)
    atomic_write_bytes(
        root / "data_availability.csv",
        _csv_bytes(availability_rows, tuple(availability_rows[0])),
    )

    finding_rows = [asdict(row) for row in result.leakage_findings]
    finding_columns = tuple(field.name for field in fields(LeakageFinding))
    atomic_write_bytes(root / "leakage_findings.csv", _csv_bytes(finding_rows, finding_columns))
    atomic_write_json(
        root / "leakage_audit.json",
        {**result.leakage_summary, "findings": finding_rows},
    )
    reproducible = manifest_sha256 == deterministic_rebuild_sha256
    metadata = {
        "manifest_schema_version": records[0].manifest_schema_version if records else "unknown",
        "manifest_sha256": manifest_sha256,
        "deterministic_rebuild_sha256": deterministic_rebuild_sha256,
        "deterministic_rebuild_matches": reproducible,
        "protocol_sha256": result.protocol_sha256,
        "p4c0_config_sha256": result.p4c0_config_sha256,
        "p4c1_config_sha256": result.config_sha256,
        "source_manifest_sha256": result.source_manifest_sha256,
        "structural_dataset_id": result.structural_dataset_id,
        "sample_count": len(records),
        "usable_count": sum(row.usable for row in records),
        "excluded_count": sum(not row.usable for row in records),
        "leakage_error_count": int(result.leakage_summary["error_count"]),
        "leakage_warning_count": int(result.leakage_summary["warning_count"]),
        "cross_split_leakage_detected": bool(
            result.leakage_summary["cross_split_leakage_detected"]
        ),
        "split_frozen_from_p4c0": True,
        "residual_values_read": False,
        "model_training_performed": False,
        "statistical_fitting_performed": False,
        "threshold_selection_performed": False,
        "classification_performance_computed": False,
    }
    atomic_write_json(root / "manifest_metadata.json", metadata)
    atomic_write_bytes(
        root / "MANIFEST_REPORT.md",
        render_manifest_report(result, manifest_sha256, required_modalities).encode("utf-8"),
    )
    return metadata
