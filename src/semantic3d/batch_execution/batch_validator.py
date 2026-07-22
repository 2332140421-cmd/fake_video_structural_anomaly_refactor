"""Validate deterministic batch plans and persisted batch completion."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

from .artifact_registry import validate_artifacts
from .batch_schema import BatchRecord, BatchStatus


def validate_batch_plan(records: Iterable[BatchRecord]) -> dict[str, Any]:
    """Check IDs, source-group ownership, split isolation, and initial state."""

    rows = tuple(records)
    group_owners: dict[tuple[str, str], list[BatchRecord]] = defaultdict(list)
    for row in rows:
        for group in row.source_group_ids:
            group_owners[(row.dataset_id, group)].append(row)
    checks = {
        "batch_ids_unique": len(rows) == len({row.batch_id for row in rows}),
        "source_groups_not_split_across_batches": all(
            len({row.batch_id for row in members}) == 1 for members in group_owners.values()
        ),
        "source_groups_not_cross_split": all(
            len({row.split for row in members}) == 1 for members in group_owners.values()
        ),
        "all_batches_initially_planned": all(row.status == BatchStatus.PLANNED.value for row in rows),
        "no_batch_execution_started": all(not row.started_at for row in rows),
    }
    return {"valid": all(checks.values()), "checks": checks, "batch_count": len(rows)}


def validate_completed_batch(record: BatchRecord) -> dict[str, Any]:
    """Validate registered artifacts for completion transition."""

    artifact_result = validate_artifacts(dict(record.artifact_sha256))
    valid = bool(record.artifact_paths) and artifact_result["valid"]
    return {"valid": valid, "artifact_validation": artifact_result}

