"""Deterministic P4-C2 artifact writers and readiness report."""

from __future__ import annotations

import csv
import io
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from semantic3d.dataset_builder.writer import atomic_write_bytes, atomic_write_json, sha256_file

from .formal_registry import license_registry_rows
from .formal_schema import P4C2BuildResult
from .p4c2_builder import readiness_manifest_sha256


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _jsonl(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(_canonical_json(row) + b"\n" for row in rows)


def _csv(rows: Iterable[Mapping[str, Any]], columns: Iterable[str]) -> bytes:
    names = tuple(columns)
    handle = io.StringIO(newline="")
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


def render_p4c2_readiness_report(result: P4C2BuildResult) -> str:
    """Render a human-readable readiness report without performance claims."""

    readiness = result.readiness
    task_counts = Counter(row.eligibility for row in result.task_eligibility)
    storage = result.storage_plan
    split_counts = readiness["formal_split_counts"]
    registered_formal = sum(row.dataset_role != "geometry_validation_smoke" for row in result.datasets)
    registered_smoke = sum(row.dataset_role == "geometry_validation_smoke" for row in result.datasets)
    lines = [
        "# P4-C2 Formal Data Build Readiness",
        "",
        "This stage audits registry, lineage, task eligibility, storage, and missingness only.",
        "It did not download data, run model inference, train, fit a distribution, select a threshold, or read test performance.",
        "",
        "## Decision",
        "",
        f"- `ready_for_formal_batch_build={str(readiness['ready_for_formal_batch_build']).lower()}`",
        f"- registered dataset entries: {readiness['registered_dataset_count']}",
        f"- registered formal datasets: {registered_formal}",
        f"- registered smoke datasets: {registered_smoke}",
        f"- verified formal datasets: {readiness['verified_formal_dataset_count']}",
        f"- verified lineage records: {readiness['verified_lineage_count']}",
        f"- unverified lineage records: {readiness['unverified_lineage_count']}",
        "",
        "## Formal Split Plan",
        "",
        f"- train: {split_counts['train']}",
        f"- validation: {split_counts['validation']}",
        f"- test: {split_counts['test']}",
        "",
        "The six local videos remain `geometry_validation_smoke`. Their prior validation-only split is preserved by P4-C1, but it is not promoted into this formal split plan because original-source identity is unverified.",
        "",
        "## Task Eligibility States",
        "",
    ]
    for status in ("eligible", "ineligible", "unknown", "not_applicable", "provider_failed"):
        lines.append(f"- {status}: {task_counts.get(status, 0)}")
    lines.extend(
        [
            "",
            "Every task row separately records declared-role eligibility and formal-experiment eligibility. Provider failure is never treated as anomaly evidence.",
            "",
            "## Storage",
            "",
            f"- audited path: `{storage['storage_snapshot']['path']}`",
            f"- audited available bytes: {storage['storage_snapshot']['available_bytes']}",
            f"- safety margin bytes: {storage['storage_snapshot']['safety_margin_bytes']}",
            f"- full build required with safety bytes: {storage['full_build_required_with_safety_bytes']}",
            f"- full build fits snapshot: {str(storage['full_build_fits_audited_snapshot']).lower()}",
            f"- batches fitting without archive/reclamation: {storage['feasible_batch_count_without_archiving']}/{storage['batch_count']}",
            "",
            "## Blocking Conditions",
            "",
        ]
    )
    lines.extend(f"- `{value}`" for value in readiness["blockers"])
    lines.extend(["", "## Conditions For The Next Stage", ""])
    lines.extend(f"- {value}" for value in readiness["next_stage_conditions"])
    lines.extend(
        [
            "",
            "## Reproducibility",
            "",
            f"- P4-C0 protocol SHA-256: `{result.protocol_sha256}`",
            f"- P4-C1 manifest SHA-256: `{result.p4c1_manifest_sha256}`",
            f"- P4-C2 config SHA-256: `{result.config_sha256}`",
            f"- P4-C2 readiness manifest SHA-256: `{readiness_manifest_sha256(result)}`",
            "",
        ]
    )
    return "\n".join(lines)


def write_p4c2_artifacts(
    output_root: str | Path,
    result: P4C2BuildResult,
    *,
    deterministic_rebuild_sha256: str,
) -> dict[str, Any]:
    """Write the complete deterministic P4-C2 readiness package."""

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    digest = readiness_manifest_sha256(result)
    if digest != deterministic_rebuild_sha256:
        raise RuntimeError("P4-C2 deterministic rebuild hash mismatch")

    datasets = []
    for row in result.datasets:
        item = row.to_dict()
        is_formal = row.dataset_role != "geometry_validation_smoke"
        item.update(
            {
                "is_formal_dataset": is_formal,
                "eligible_for_formal_split": is_formal and row.registry_status == "verified_ready",
            }
        )
        datasets.append(item)
    licenses = license_registry_rows(result.datasets)
    lineage = [row.to_dict() for row in result.lineage]
    splits = [row.to_dict() for row in result.formal_splits]
    tasks = []
    for row in result.task_eligibility:
        item = row.to_dict()
        item["eligibility_scope"] = "video"
        tasks.append(item)
    inventory = list(result.data_inventory)
    atomic_write_json(
        root / "formal_dataset_registry.json",
        {"schema_version": "formal_dataset_registry_v1", "datasets": datasets},
    )
    atomic_write_json(
        root / "dataset_license_registry.json",
        {"schema_version": "dataset_license_registry_v1", "licenses": licenses},
    )
    atomic_write_bytes(root / "source_lineage.jsonl", _jsonl(lineage))
    atomic_write_bytes(root / "formal_split_plan.jsonl", _jsonl(splits))
    task_columns = tuple(tasks[0]) if tasks else (
        "dataset_name",
        "derived_video_id",
        "task_name",
        "eligibility",
    )
    atomic_write_bytes(root / "task_eligibility_matrix.csv", _csv(tasks, task_columns))
    inventory_columns = tuple(inventory[0]) if inventory else ("dataset_name", "derived_video_id")
    atomic_write_bytes(root / "data_inventory.csv", _csv(inventory, inventory_columns))
    atomic_write_json(root / "source_lineage_leakage_audit.json", result.leakage_audit)
    atomic_write_json(root / "storage_batch_build_plan.json", result.storage_plan)
    missingness_payload = yaml.safe_dump(
        dict(result.missingness_plan),
        allow_unicode=True,
        sort_keys=True,
        default_flow_style=False,
    ).encode("utf-8")
    atomic_write_bytes(root / "missingness_audit_plan.yaml", missingness_payload)
    atomic_write_bytes(
        root / "p4c2_readiness_report.md",
        render_p4c2_readiness_report(result).encode("utf-8"),
    )
    metadata = {
        "schema_version": "p4c2_build_metadata_v1",
        "p4c2_readiness_manifest_sha256": digest,
        "deterministic_rebuild_sha256": deterministic_rebuild_sha256,
        "deterministic_rebuild_matches": digest == deterministic_rebuild_sha256,
        "protocol_sha256": result.protocol_sha256,
        "p4c1_manifest_sha256": result.p4c1_manifest_sha256,
        "p4c2_config_sha256": result.config_sha256,
        "dataset_count": len(result.datasets),
        "registered_dataset_entries": len(result.datasets),
        "registered_formal_datasets": sum(
            row.dataset_role != "geometry_validation_smoke" for row in result.datasets
        ),
        "registered_smoke_datasets": sum(
            row.dataset_role == "geometry_validation_smoke" for row in result.datasets
        ),
        "lineage_record_count": len(result.lineage),
        "formal_split_record_count": len(result.formal_splits),
        "task_eligibility_record_count": len(result.task_eligibility),
        "ready_for_formal_batch_build": result.readiness["ready_for_formal_batch_build"],
        "downloads_performed": False,
        "model_inference_performed": False,
        "formal_build_started": False,
        "model_training_performed": False,
        "statistical_fitting_performed": False,
        "threshold_selection_performed": False,
        "test_performance_read": False,
        "authenticity_performance_computed": False,
    }
    atomic_write_json(root / "build_metadata.json", metadata)
    primary_names = (
        "formal_dataset_registry.json",
        "dataset_license_registry.json",
        "source_lineage.jsonl",
        "formal_split_plan.jsonl",
        "task_eligibility_matrix.csv",
        "data_inventory.csv",
        "source_lineage_leakage_audit.json",
        "storage_batch_build_plan.json",
        "missingness_audit_plan.yaml",
        "p4c2_readiness_report.md",
        "build_metadata.json",
    )
    artifact_hashes = {name: sha256_file(root / name) for name in primary_names}
    atomic_write_json(
        root / "artifact_hashes.json",
        {"hash_algorithm": "sha256", "artifacts": artifact_hashes},
    )
    return metadata
