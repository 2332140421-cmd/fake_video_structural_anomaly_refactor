"""Conservative batch cleanup policy, dry-run by default."""

from __future__ import annotations

from typing import Any, Iterable

from .batch_schema import BatchRecord, BatchStatus


def plan_cleanup(
    record: BatchRecord,
    candidate_paths: Iterable[str],
    *,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Permit cleanup only after completed and validated artifacts."""

    allowed = record.status == BatchStatus.COMPLETED.value and record.artifacts_validated
    if not allowed:
        raise ValueError("Cleanup is forbidden before completed artifact validation")
    return {
        "batch_id": record.batch_id,
        "dry_run": dry_run,
        "cleanup_allowed": allowed,
        "candidate_paths": sorted(set(candidate_paths)),
        "paths_deleted": [] if dry_run else [],
        "execution_performed": False,
        "policy": "only_redownloadable_temporary_inputs_after_completed_validation",
    }

