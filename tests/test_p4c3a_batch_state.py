"""Tests for deterministic batch planning, state safety, locks, and cleanup."""

from __future__ import annotations

from pathlib import Path

import pytest

from semantic3d.batch_execution.batch_lock import BatchLock
from semantic3d.batch_execution.batch_planner import plan_batches, plan_sha256
from semantic3d.batch_execution.batch_schema import BatchRecord, BatchStatus
from semantic3d.batch_execution.batch_state_store import BatchStateStore
from semantic3d.batch_execution.cleanup_policy import plan_cleanup


def _planned() -> BatchRecord:
    return BatchRecord(
        batch_id="batch-unit",
        dataset_id="dataset",
        split="train",
        source_group_ids=("group-a",),
        input_manifest_sha256="a" * 64,
        runtime_profile="unit",
        planned_input_bytes=1,
        planned_temporary_bytes=2,
        planned_output_bytes=3,
    )


def test_batch_id_and_plan_hash_are_deterministic() -> None:
    rows = [
        {"dataset_id": "d", "source_group_id": "g2", "formal_split": "train", "eligible": True, "planned_input_bytes": 2},
        {"dataset_id": "d", "source_group_id": "g1", "formal_split": "train", "eligible": True, "planned_input_bytes": 2},
    ]
    first = plan_batches(rows, input_manifest_sha256="a" * 64, runtime_profile="r", batch_storage_limit=10, software_commit="c", protocol_sha256="p", manifest_sha256="m")
    second = plan_batches(reversed(rows), input_manifest_sha256="a" * 64, runtime_profile="r", batch_storage_limit=10, software_commit="c", protocol_sha256="p", manifest_sha256="m")
    assert [row.batch_id for row in first] == [row.batch_id for row in second]
    assert plan_sha256(first) == plan_sha256(second)


def test_atomic_state_write_and_legal_transitions(tmp_path: Path) -> None:
    store = BatchStateStore(tmp_path)
    store.initialize(_planned())
    downloading = store.transition("batch-unit", BatchStatus.DOWNLOADING, now="t1")
    assert downloading.status == "downloading"
    assert not list(tmp_path.glob("*.tmp"))
    downloaded = store.transition("batch-unit", BatchStatus.DOWNLOADED, now="t2")
    assert downloaded.status == "downloaded"


def test_illegal_transition_is_rejected(tmp_path: Path) -> None:
    store = BatchStateStore(tmp_path)
    store.initialize(_planned())
    with pytest.raises(ValueError, match="Illegal batch transition"):
        store.transition("batch-unit", BatchStatus.COMPLETED, now="t")


def test_failed_batch_preserves_reason_and_retry_attempt(tmp_path: Path) -> None:
    store = BatchStateStore(tmp_path)
    store.initialize(_planned())
    failed = store.transition("batch-unit", BatchStatus.FAILED, now="t1", failure_reason="network")
    assert failed.failure_reasons == ("network",)
    retried = store.transition("batch-unit", BatchStatus.DOWNLOADING, now="t2")
    assert retried.attempt == 1
    assert retried.failure_reasons == ("network",)


def test_cleanup_forbidden_until_completed_validation() -> None:
    with pytest.raises(ValueError, match="Cleanup is forbidden"):
        plan_cleanup(_planned(), ["download.tmp"])
    completed = BatchRecord(
        **{
            **_planned().to_dict(),
            "status": "completed",
            "artifacts_validated": True,
            "completed_at": "t",
        }
    )
    result = plan_cleanup(completed, ["download.tmp"])
    assert result["dry_run"]
    assert not result["execution_performed"]


def test_batch_lock_prevents_second_owner(tmp_path: Path) -> None:
    first = BatchLock(tmp_path, "batch")
    first.acquire()
    try:
        with pytest.raises(RuntimeError, match="already locked"):
            BatchLock(tmp_path, "batch").acquire()
    finally:
        first.release()

