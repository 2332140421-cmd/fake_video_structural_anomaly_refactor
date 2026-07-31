"""Read-only relocation of frozen manifest paths to verified server assets."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping, Sequence

RUNTIME_REQUIRED_FIELDS = (
    "sample_id",
    "local_result_json",
    "local_video_path",
)


def load_runtime_path_manifest(path: str | Path) -> dict[str, dict[str, str]]:
    resolved = Path(path).resolve()
    with resolved.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = set(reader.fieldnames or ())
    missing = sorted(set(RUNTIME_REQUIRED_FIELDS) - fields)
    if not rows or missing:
        raise ValueError(f"Runtime path manifest is missing required fields: {missing}.")
    output: dict[str, dict[str, str]] = {}
    for row in rows:
        sample_id = row["sample_id"].strip()
        if not sample_id:
            raise ValueError("Runtime path manifest contains an empty sample_id.")
        if sample_id in output:
            raise ValueError(f"Duplicate runtime path sample_id: {sample_id!r}.")
        output[sample_id] = dict(row)
    return output


def relocate_manifest_rows(
    rows: Sequence[Mapping[str, str]],
    runtime_manifest_path: str | Path,
) -> list[dict[str, str]]:
    """Return relocated copies; never rewrite the frozen provenance manifest."""

    runtime_rows = load_runtime_path_manifest(runtime_manifest_path)
    relocated: list[dict[str, str]] = []
    for source_row in rows:
        row = dict(source_row)
        sample_id = row["sample_id"].strip()
        runtime = runtime_rows.get(sample_id)
        if runtime is None:
            raise ValueError(f"{sample_id}: missing from runtime path manifest.")
        residual_path = Path(runtime["local_result_json"]).expanduser().resolve()
        video_path = Path(runtime["local_video_path"]).expanduser().resolve()
        if not video_path.is_file():
            raise FileNotFoundError(f"{sample_id}: runtime source video is missing.")
        expected_residual_hash = row.get("residual_sha256", "").strip()
        runtime_residual_hash = runtime.get("residual_sha256", "").strip()
        if (
            expected_residual_hash
            and runtime_residual_hash
            and expected_residual_hash != runtime_residual_hash
        ):
            raise ValueError(f"{sample_id}: runtime residual identity differs.")
        expected_video_hash = row.get("source_video_sha256", "").strip()
        runtime_video_hash = runtime.get("source_video_sha256", "").strip()
        if (
            expected_video_hash
            and runtime_video_hash
            and expected_video_hash != runtime_video_hash
        ):
            raise ValueError(f"{sample_id}: runtime source video identity differs.")
        row["_provenance_residual_sequence_path"] = row.get(
            "residual_sequence_path", ""
        )
        row["_provenance_source_video_path"] = row.get("source_video_path", "")
        row["_runtime_relocated"] = "true"
        row["residual_sequence_path"] = str(residual_path)
        row["source_video_path"] = str(video_path)
        if runtime_video_hash:
            row["source_video_sha256"] = runtime_video_hash
        relocated.append(row)
    return relocated


__all__ = [
    "RUNTIME_REQUIRED_FIELDS",
    "load_runtime_path_manifest",
    "relocate_manifest_rows",
]
