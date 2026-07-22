"""Explicit retry policy preserving attempt and failure history."""

from __future__ import annotations

from dataclasses import dataclass

from .batch_schema import BatchRecord, BatchStatus


@dataclass(frozen=True)
class RetryDecision:
    allowed: bool
    next_status: str
    reason: str


def evaluate_retry(record: BatchRecord, *, max_attempts: int = 3) -> RetryDecision:
    """Allow retries only for failed batches below the configured limit."""

    if record.status != BatchStatus.FAILED.value:
        return RetryDecision(False, "", "batch_is_not_failed")
    if record.attempt >= max_attempts:
        return RetryDecision(False, "", "maximum_retry_attempts_reached")
    downloaded = any(row.get("to") == BatchStatus.DOWNLOADED.value for row in record.history)
    return RetryDecision(
        True,
        BatchStatus.PROCESSING.value if downloaded else BatchStatus.DOWNLOADING.value,
        "retry_requires_explicit_state_transition",
    )

