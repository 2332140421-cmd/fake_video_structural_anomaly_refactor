"""Stable-state recovery helpers for interrupted batches."""

from __future__ import annotations

from typing import Mapping

from .batch_schema import BatchRecord, BatchStatus

_STABLE = {
    BatchStatus.PLANNED.value,
    BatchStatus.DOWNLOADED.value,
    BatchStatus.COMPLETED.value,
    BatchStatus.FAILED.value,
    BatchStatus.CLEANED.value,
}


def last_stable_status(record: BatchRecord) -> str:
    """Return the last stable state recorded before interruption."""

    if record.status in _STABLE:
        return record.status
    stable = BatchStatus.PLANNED.value
    for transition in record.history:
        target = str(transition.get("to", ""))
        if target in _STABLE:
            stable = target
    return stable


def recovery_action(record: BatchRecord) -> Mapping[str, str]:
    """Describe explicit recovery without mutating or resetting state."""

    stable = last_stable_status(record)
    if record.status == BatchStatus.FAILED.value:
        action = "operator_review_then_retry_from_last_stable_state"
    elif record.status in {
        BatchStatus.DOWNLOADING.value,
        BatchStatus.PROCESSING.value,
        BatchStatus.VALIDATING.value,
    }:
        action = "mark_failed_with_interruption_reason_before_retry"
    else:
        action = "no_recovery_required"
    return {"current_status": record.status, "last_stable_status": stable, "action": action}

