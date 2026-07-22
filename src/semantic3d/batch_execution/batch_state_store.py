"""Atomic batch state persistence with validated transitions and retries."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from semantic3d.dataset_builder.writer import atomic_write_json

from .batch_schema import BatchRecord, BatchStatus, validate_transition


class BatchStateStore:
    """One-file-per-batch state store suitable for interruption recovery."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, batch_id: str) -> Path:
        return self.root / f"{batch_id}.json"

    def save(self, record: BatchRecord) -> Path:
        """Atomically persist one validated record."""

        return atomic_write_json(self.path_for(record.batch_id), record.to_dict())

    def load(self, batch_id: str) -> BatchRecord:
        """Load one batch without resetting failures or partial artifacts."""

        return BatchRecord(**json.loads(self.path_for(batch_id).read_text(encoding="utf-8")))

    def initialize(self, record: BatchRecord) -> BatchRecord:
        """Create planned state; an existing state is never overwritten."""

        path = self.path_for(record.batch_id)
        if path.exists():
            return self.load(record.batch_id)
        self.save(record)
        return record

    def transition(
        self,
        batch_id: str,
        target: BatchStatus | str,
        *,
        now: str | None = None,
        failure_reason: str = "",
        updates: Mapping[str, Any] | None = None,
    ) -> BatchRecord:
        """Apply a legal transition and atomically preserve its history."""

        current = self.load(batch_id)
        destination = BatchStatus(target)
        validate_transition(current.status, destination)
        timestamp = now or datetime.now(timezone.utc).isoformat()
        changes = dict(updates or {})
        reasons = current.failure_reasons
        failed_count = current.failed_count
        attempt = current.attempt
        if destination == BatchStatus.FAILED:
            if not failure_reason:
                raise ValueError("failed transition requires failure_reason")
            reasons = (*reasons, failure_reason)
            failed_count += 1
        if BatchStatus(current.status) == BatchStatus.FAILED:
            attempt += 1
        if destination == BatchStatus.COMPLETED and not bool(
            changes.get("artifacts_validated", current.artifacts_validated)
        ):
            raise ValueError("completed transition requires artifacts_validated=true")
        history = (
            *current.history,
            {
                "from": current.status,
                "to": destination.value,
                "at": timestamp,
                "failure_reason": failure_reason,
            },
        )
        record = replace(
            current,
            status=destination.value,
            attempt=attempt,
            started_at=current.started_at or timestamp,
            updated_at=timestamp,
            completed_at=timestamp if destination == BatchStatus.COMPLETED else current.completed_at,
            failed_count=failed_count,
            failure_reasons=reasons,
            history=history,
            **changes,
        )
        self.save(record)
        return record

