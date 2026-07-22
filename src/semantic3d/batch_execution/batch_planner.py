"""Deterministic source-group-preserving batch planner."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Any, Iterable, Mapping

from .batch_schema import BatchRecord


def _stable_id(
    dataset_id: str, split: str, groups: Iterable[str], input_manifest_sha256: str
) -> str:
    payload = json.dumps(
        [dataset_id, split, sorted(groups), input_manifest_sha256],
        separators=(",", ":"),
    )
    return f"batch_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:20]}"


def plan_batches(
    source_rows: Iterable[Mapping[str, Any]],
    *,
    input_manifest_sha256: str,
    runtime_profile: str,
    batch_storage_limit: int,
    software_commit: str,
    protocol_sha256: str,
    manifest_sha256: str,
) -> tuple[BatchRecord, ...]:
    """Pack whole source groups in stable order without running any batch."""

    if batch_storage_limit <= 0:
        raise ValueError("batch_storage_limit must be positive")
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    group_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in source_rows:
        if not bool(row.get("eligible", True)) or not str(row.get("formal_split", row.get("split", ""))):
            continue
        dataset = str(row["dataset_id"])
        group = str(row["source_group_id"])
        split = str(row.get("formal_split", row.get("split")))
        group_splits[(dataset, group)].add(split)
        grouped[(dataset, split, group)].append(row)
    conflicts = [key for key, values in group_splits.items() if len(values) > 1]
    if conflicts:
        raise ValueError(f"Source groups cross formal splits: {sorted(conflicts)}")

    plans: list[BatchRecord] = []
    by_dataset_split: dict[tuple[str, str], list[tuple[str, int, int, int]]] = defaultdict(list)
    for (dataset, split, group), members in grouped.items():
        by_dataset_split[(dataset, split)].append(
            (
                group,
                sum(int(row.get("planned_input_bytes", 0)) for row in members),
                sum(int(row.get("planned_temporary_bytes", 0)) for row in members),
                sum(int(row.get("planned_output_bytes", 0)) for row in members),
            )
        )
    for (dataset, split), groups in sorted(by_dataset_split.items()):
        current: list[tuple[str, int, int, int]] = []
        current_peak = 0

        def flush() -> None:
            nonlocal current, current_peak
            if not current:
                return
            group_ids = tuple(sorted(row[0] for row in current))
            plans.append(
                BatchRecord(
                    batch_id=_stable_id(dataset, split, group_ids, input_manifest_sha256),
                    dataset_id=dataset,
                    split=split,
                    source_group_ids=group_ids,
                    input_manifest_sha256=input_manifest_sha256,
                    runtime_profile=runtime_profile,
                    planned_input_bytes=sum(row[1] for row in current),
                    planned_temporary_bytes=sum(row[2] for row in current),
                    planned_output_bytes=sum(row[3] for row in current),
                    software_commit=software_commit,
                    protocol_sha256=protocol_sha256,
                    manifest_sha256=manifest_sha256,
                )
            )
            current = []
            current_peak = 0

        for item in sorted(groups, key=lambda row: row[0]):
            peak = item[1] + item[2] + item[3]
            if current and current_peak + peak > batch_storage_limit:
                flush()
            current.append(item)
            current_peak += peak
        flush()
    return tuple(sorted(plans, key=lambda row: (row.dataset_id, row.split, row.batch_id)))


def plan_sha256(records: Iterable[BatchRecord]) -> str:
    """Hash deterministic planned fields; timestamps are initially empty."""

    payload = json.dumps(
        [row.to_dict() for row in sorted(records, key=lambda item: item.batch_id)],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

