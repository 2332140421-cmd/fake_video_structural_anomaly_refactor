"""Deterministic batch plan writer and human-readable report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from semantic3d.dataset_builder.writer import atomic_write_bytes, atomic_write_json

from .batch_planner import plan_sha256
from .batch_schema import BatchRecord
from .batch_validator import validate_batch_plan


def write_batch_plan(output_root: str | Path, records: Iterable[BatchRecord]) -> dict[str, object]:
    """Write a planned-only batch package; never start execution."""

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    rows = tuple(sorted(records, key=lambda row: row.batch_id))
    payload = b"".join(
        json.dumps(row.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        + b"\n"
        for row in rows
    )
    atomic_write_bytes(root / "batch_plan.jsonl", payload)
    validation = validate_batch_plan(rows)
    digest = plan_sha256(rows)
    metadata = {
        "batch_count": len(rows),
        "batch_plan_sha256": digest,
        "validation": validation,
        "formal_download_performed": False,
        "formal_processing_performed": False,
        "formal_batch_started": False,
    }
    atomic_write_json(root / "batch_plan_metadata.json", metadata)
    report = "\n".join(
        [
            "# P4-C3A Batch Plan",
            "",
            f"- batch count: {len(rows)}",
            f"- plan SHA-256: `{digest}`",
            f"- valid: `{str(validation['valid']).lower()}`",
            "- execution started: `false`",
            "",
            "An empty plan is expected while no verified formal split exists.",
            "",
        ]
    )
    atomic_write_bytes(root / "batch_plan_report.md", report.encode("utf-8"))
    return metadata

