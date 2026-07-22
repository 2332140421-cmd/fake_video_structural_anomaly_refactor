"""Typed batch execution state with explicit legal transitions."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, Mapping


class BatchStatus(StrEnum):
    PLANNED = "planned"
    DOWNLOADING = "downloading"
    DOWNLOADED = "downloaded"
    PROCESSING = "processing"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"
    CLEANED = "cleaned"


LEGAL_TRANSITIONS: Mapping[BatchStatus, frozenset[BatchStatus]] = {
    BatchStatus.PLANNED: frozenset({BatchStatus.DOWNLOADING, BatchStatus.FAILED}),
    BatchStatus.DOWNLOADING: frozenset({BatchStatus.DOWNLOADED, BatchStatus.FAILED}),
    BatchStatus.DOWNLOADED: frozenset({BatchStatus.PROCESSING, BatchStatus.FAILED}),
    BatchStatus.PROCESSING: frozenset({BatchStatus.VALIDATING, BatchStatus.FAILED}),
    BatchStatus.VALIDATING: frozenset({BatchStatus.COMPLETED, BatchStatus.FAILED}),
    BatchStatus.COMPLETED: frozenset({BatchStatus.CLEANED}),
    BatchStatus.FAILED: frozenset({BatchStatus.DOWNLOADING, BatchStatus.PROCESSING}),
    BatchStatus.CLEANED: frozenset(),
}


@dataclass(frozen=True)
class BatchRecord:
    """One planned or runtime batch state record."""

    batch_id: str
    dataset_id: str
    split: str
    source_group_ids: tuple[str, ...]
    input_manifest_sha256: str
    runtime_profile: str
    planned_input_bytes: int
    planned_temporary_bytes: int
    planned_output_bytes: int
    status: str = BatchStatus.PLANNED.value
    attempt: int = 0
    started_at: str = ""
    updated_at: str = ""
    completed_at: str = ""
    processed_count: int = 0
    failed_count: int = 0
    failure_reasons: tuple[str, ...] = ()
    artifact_paths: tuple[str, ...] = ()
    artifact_sha256: Mapping[str, str] = field(default_factory=dict)
    cleanup_status: str = "not_requested"
    software_commit: str = ""
    protocol_sha256: str = ""
    manifest_sha256: str = ""
    artifacts_validated: bool = False
    validation_report_path: str = ""
    history: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_group_ids", tuple(self.source_group_ids))
        object.__setattr__(self, "failure_reasons", tuple(self.failure_reasons))
        object.__setattr__(self, "artifact_paths", tuple(self.artifact_paths))
        object.__setattr__(self, "history", tuple(self.history))
        BatchStatus(self.status)
        if not self.batch_id or not self.dataset_id or not self.input_manifest_sha256:
            raise ValueError("batch_id, dataset_id, and input_manifest_sha256 are required")
        if self.split not in {"train", "validation", "test"}:
            raise ValueError("Batch split must be train, validation, or test")
        if not self.source_group_ids:
            raise ValueError("A batch must contain at least one complete source group")
        if any(value < 0 for value in (self.planned_input_bytes, self.planned_temporary_bytes, self.planned_output_bytes)):
            raise ValueError("Planned byte counts must be non-negative")
        if self.status == BatchStatus.COMPLETED.value and not self.artifacts_validated:
            raise ValueError("A completed batch must have validated artifacts")

    def to_dict(self) -> dict[str, Any]:
        """Return a stable serialization mapping."""

        return asdict(self)


def validate_transition(current: BatchStatus | str, target: BatchStatus | str) -> None:
    """Raise when a state transition could hide incomplete work."""

    source = BatchStatus(current)
    destination = BatchStatus(target)
    if destination not in LEGAL_TRANSITIONS[source]:
        raise ValueError(f"Illegal batch transition: {source.value} -> {destination.value}")
