"""Build a P4-C0 inventory from structural metadata and isolated labels."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import pyarrow.parquet as pq

from .schema import DatasetCatalog, LabelRecord, SourceGroupRecord, VideoInventoryRecord


def _optional_bool(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "available"}


def _json_or_default(value: str | None, default: Any) -> Any:
    text = str(value or "").strip()
    if not text:
        return default
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON annotation field: {text!r}") from exc


def load_isolated_labels(path: str | Path) -> dict[str, LabelRecord]:
    """Load an explicit label manifest without inferring labels from filenames."""

    manifest = Path(path)
    if not manifest.is_file():
        raise FileNotFoundError(f"Independent labels manifest not found: {manifest}")
    labels: dict[str, LabelRecord] = {}
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        for row_index, row in enumerate(csv.DictReader(handle), start=2):
            source_id = str(row.get("video_id", "")).strip()
            if not source_id:
                raise ValueError(f"Missing video_id in label manifest row {row_index}")
            if source_id in labels:
                raise ValueError(f"Duplicate label video_id: {source_id}")
            raw_label = str(row.get("label", "")).strip()
            if raw_label not in {"0", "1", "", "unknown", "not_available"}:
                raise ValueError(f"Invalid binary label for {source_id}: {raw_label!r}")
            binary_label = int(raw_label) if raw_label in {"0", "1"} else None
            intervals = _json_or_default(row.get("temporal_intervals"), [])
            spatial_bbox = _json_or_default(row.get("spatial_bbox"), [])
            labels[source_id] = LabelRecord(
                video_id=source_id,
                binary_label=binary_label,
                label_name=str(row.get("label_name", "unknown") or "unknown").strip(),
                manipulation_type=str(row.get("manipulation_type", "unknown") or "unknown").strip(),
                temporal_intervals=tuple((float(item[0]), float(item[1])) for item in intervals),
                spatial_mask_path=str(row.get("spatial_mask_path", "") or "").strip(),
                spatial_bbox=tuple(float(value) for value in spatial_bbox),
                object_annotation_path=str(row.get("object_annotation_path", "") or "").strip(),
                source_group_id=str(row.get("source_group_id", "") or "").strip(),
                annotation_quality=str(row.get("annotation_quality", "binary_only") or "binary_only").strip(),
                annotation_source=str(row.get("annotation_source", manifest) or manifest),
                declared_split=str(row.get("split", "") or "").strip(),
                original_source_identity=str(row.get("original_source_identity", "") or "").strip(),
            )
    return labels


def read_parquet_rows(path: str | Path) -> list[dict[str, Any]]:
    """Read a Parquet table as Python mappings."""

    return pq.read_table(Path(path)).to_pylist()


def build_video_inventory(
    structural_dataset_root: str | Path,
    project_root: str | Path,
    labels_by_source_name: Mapping[str, LabelRecord],
    source_groups: Mapping[str, SourceGroupRecord],
    *,
    dataset_name: str,
    protocol_smoke_only: bool = True,
) -> list[VideoInventoryRecord]:
    """Join P4-B.5 video metadata with labels and source-group provenance."""

    root = Path(structural_dataset_root)
    project = Path(project_root)
    videos = read_parquet_rows(root / "manifests/videos.parquet")
    inventory: list[VideoInventoryRecord] = []
    for row in videos:
        source_name = str(row["source_name"])
        if source_name not in labels_by_source_name:
            raise ValueError(f"No explicit label manifest row for source_name={source_name}")
        label = labels_by_source_name[source_name]
        group = source_groups[str(row["video_id"])]
        fps = float(row["fps"])
        frame_count = int(row["frame_count"])
        temporal = bool(label.temporal_intervals)
        spatial = bool(label.spatial_mask_path or label.spatial_bbox)
        object_annotation = bool(label.object_annotation_path)
        exclusions = []
        if protocol_smoke_only:
            exclusions.append("protocol_smoke_not_formal_training_data")
        if group.source_group_review_required:
            exclusions.append("source_group_review_required")
        if label.binary_label is None:
            exclusions.append("binary_label_unavailable")
        inventory.append(
            VideoInventoryRecord(
                video_id=str(row["video_id"]),
                dataset_name=dataset_name,
                source_name=source_name,
                source_path=str((project / str(row["source_relative_path"])).resolve()),
                source_sha256=str(row["source_sha256"]),
                file_size=int(row["file_size"]),
                frame_count=frame_count,
                fps=fps,
                width=int(row["width"]),
                height=int(row["height"]),
                duration_seconds=frame_count / fps if fps > 0 else 0.0,
                binary_label=label.binary_label,
                label_name=label.label_name,
                manipulation_type=label.manipulation_type,
                source_group_id=group.source_group_id,
                source_group_review_required=group.source_group_review_required,
                declared_split=label.declared_split,
                temporal_annotation_available=temporal,
                spatial_annotation_available=spatial,
                object_annotation_available=object_annotation,
                annotation_quality=label.annotation_quality,
                detection_training_eligible=bool(label.binary_label is not None and not protocol_smoke_only),
                video_classification_eligible=bool(label.binary_label is not None and not protocol_smoke_only),
                temporal_localization_eligible=bool(temporal and not protocol_smoke_only),
                spatial_localization_eligible=bool(spatial and not protocol_smoke_only),
                object_localization_eligible=bool(object_annotation and not protocol_smoke_only),
                occlusion_validation_eligible=False,
                geometry_validation_only=protocol_smoke_only,
                exclusion_reasons=tuple(exclusions),
                metadata={
                    "label_inferred_from_filename": False,
                    "structure_dataset_label_isolation": True,
                },
            )
        )
    return inventory


def build_dataset_catalog(
    inventory: Iterable[VideoInventoryRecord],
    *,
    dataset_name: str,
    dataset_version: str,
    source_root: str,
    license_or_usage_note: str,
    expected_storage_bytes: int,
) -> DatasetCatalog:
    """Summarize the local six-video engineering set without inventing metadata."""

    rows = list(inventory)
    return DatasetCatalog(
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        source_root=source_root,
        license_or_usage_note=license_or_usage_note,
        official_split_available=False,
        video_count=len(rows),
        real_count=sum(row.binary_label == 0 for row in rows),
        fake_count=sum(row.binary_label == 1 for row in rows),
        manipulation_types=tuple(sorted({row.manipulation_type for row in rows if row.manipulation_type != "unknown"})),
        temporal_annotation_available=any(row.temporal_annotation_available for row in rows),
        spatial_annotation_available=any(row.spatial_annotation_available for row in rows),
        original_source_identity_available=any(not row.source_group_review_required for row in rows),
        compression_variants=(),
        expected_storage_bytes=int(expected_storage_bytes),
        roles=("geometry_validation",),
        sample_scope="six local videos; protocol smoke only; not formal performance data",
    )


def inventory_rows(records: Iterable[VideoInventoryRecord]) -> list[dict[str, Any]]:
    """Convert inventory dataclasses to writer-friendly rows."""

    return [asdict(record) for record in records]
