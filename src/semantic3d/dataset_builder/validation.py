"""Integrity validation for P4-B structural-enhancement datasets."""

from __future__ import annotations

import json
import math
import csv
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .reader import DatasetReader
from .schema import (
    P4B5_PIPELINE_VERSION,
    P4B5_TABLE_PRIMARY_KEYS,
    Applicability,
    TABLE_PRIMARY_KEYS,
)
from .writer import atomic_write_json, sha256_file, write_parquet


@dataclass
class DatasetValidationReport:
    """Machine-readable validation summary and detailed integrity failures."""

    valid: bool
    checked_table_count: int
    error_count: int
    warning_count: int
    errors: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _error(errors: list[dict[str, Any]], check: str, detail: str, **context: Any) -> None:
    errors.append({"check": check, "detail": detail, "context": json.dumps(context, sort_keys=True)})


def validate_dataset(root: str | Path, *, write_reports: bool = True) -> DatasetValidationReport:
    """Validate IDs, ownership, coordinates, arrays, evidence, and label isolation."""

    dataset_root = Path(root)
    reader = DatasetReader(dataset_root)
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    tables: dict[str, list[dict[str, Any]]] = {}
    manifest_path = dataset_root / "dataset_manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            manifest = {}
            _error(errors, "dataset_manifest", str(exc))
    else:
        manifest = {}
    primary_keys = dict(TABLE_PRIMARY_KEYS)
    is_p4b5 = manifest.get("pipeline_version") == P4B5_PIPELINE_VERSION
    if is_p4b5:
        primary_keys.update(P4B5_TABLE_PRIMARY_KEYS)
    for relative_path, key_columns in primary_keys.items():
        path = dataset_root / relative_path
        if not path.exists():
            _error(errors, "required_table", f"Missing table: {relative_path}")
            tables[relative_path] = []
            continue
        try:
            rows = reader.rows(relative_path)
            tables[relative_path] = rows
        except Exception as exc:
            _error(errors, "parquet_read", str(exc), table=relative_path)
            tables[relative_path] = []
            continue
        keys = [tuple(row.get(column) for column in key_columns) for row in rows]
        duplicates = [key for key, count in Counter(keys).items() if count > 1]
        if duplicates:
            _error(errors, "id_uniqueness", "Duplicate primary keys", table=relative_path, count=len(duplicates))

    videos = {row["video_id"] for row in tables.get("manifests/videos.parquet", [])}
    clips = {row["clip_id"] for row in tables.get("manifests/clips.parquet", [])}
    frames = tables.get("manifests/frames.parquet", [])
    frame_ids = {row["frame_id"] for row in frames}
    owner_counts: Counter[str] = Counter(
        str(row["frame_id"]) for row in frames if bool(row.get("is_owned_frame"))
    )
    source_frames = set(frame_ids)
    for frame_id in source_frames:
        if owner_counts[frame_id] != 1:
            _error(errors, "owner_frame_unique", "Frame must have exactly one owner", frame_id=frame_id, count=owner_counts[frame_id])
    for row in frames:
        if row.get("video_id") not in videos or row.get("clip_id") not in clips:
            _error(errors, "frame_foreign_key", "Frame references unknown video/clip", frame_record_id=row.get("frame_record_id"))
        if bool(row.get("is_owned_frame")) and row.get("owner_clip_id") != row.get("clip_id"):
            _error(errors, "owner_consistency", "Owned row does not match owner_clip_id", frame_record_id=row.get("frame_record_id"))

    clip_rows = {row["clip_id"]: row for row in tables.get("manifests/clips.parquet", [])}
    for clip_id, rows in _group(frames, "clip_id").items():
        indices = [int(row["frame_index"]) for row in rows]
        if indices != sorted(indices):
            _error(errors, "temporal_monotonicity", "Clip frame indices are not monotonic", clip_id=clip_id)
        if clip_id in clip_rows and len({row.get("scene_id") for row in rows}) > 1:
            _error(errors, "scene_cut_boundary", "Clip crosses scene IDs", clip_id=clip_id)

    for relative_path in ("observations/objects.parquet", "observations/masks.parquet", "observations/keypoints.parquet"):
        for row in tables.get(relative_path, []):
            if row.get("video_id") not in videos or row.get("frame_id") not in frame_ids:
                _error(errors, "observation_foreign_key", "Observation references unknown video/frame", table=relative_path)

    coordinate_by_clip = {row["clip_id"]: row.get("coordinate_system_id") for row in clip_rows.values()}
    for relative_path in (
        "evidence/point_evidence.parquet", "evidence/edge_evidence.parquet",
        "evidence/object_evidence.parquet", "evidence/frame_evidence.parquet",
        "evidence/clip_evidence.parquet",
    ):
        for row in tables.get(relative_path, []):
            clip_id = row.get("clip_id")
            if clip_id not in clips:
                _error(errors, "evidence_foreign_key", "Evidence references unknown clip", evidence_id=row.get("evidence_id"))
            coordinate_id = row.get("coordinate_system_id")
            expected_coordinate = coordinate_by_clip.get(clip_id)
            if coordinate_id and expected_coordinate and coordinate_id != expected_coordinate:
                _error(errors, "coordinate_system", "Evidence coordinate differs from its clip", evidence_id=row.get("evidence_id"))
            valid = bool(row.get("valid"))
            raw_value = row.get("raw_value")
            applicability = row.get("applicability")
            if valid and (not _finite(raw_value) or applicability != Applicability.APPLICABLE_VALID.value):
                _error(errors, "evidence_validity", "Valid evidence requires finite value and applicable_valid", evidence_id=row.get("evidence_id"))
            if not valid and _finite(raw_value):
                _error(errors, "nan_missing", "Invalid evidence must not be zero/finite", evidence_id=row.get("evidence_id"))
            if _finite(row.get("statistically_normalized_value")) or row.get("normalization_fit_source") != "none":
                _error(errors, "statistical_normalization", "P4-B must not fit evaluation statistics", evidence_id=row.get("evidence_id"))
            metadata = str(row.get("metadata") or "")
            if '"formal_or_diagnostic":"diagnostic"' in metadata and bool(row.get("included_in_formal_aggregation")):
                _error(errors, "diagnostic_aggregation", "Diagnostic evidence entered formal aggregation", evidence_id=row.get("evidence_id"))

    array_errors = 0
    for relative_path in ("observations/depth.parquet", "observations/masks.parquet", "observations/shared_3d_frames.parquet"):
        for row in tables.get(relative_path, []):
            array_path = row.get("array_path") or row.get("visible_mask_path")
            array_hash = row.get("array_sha256") or row.get("mask_sha256")
            if not array_path:
                continue
            path = Path(str(array_path))
            if not path.is_absolute():
                path = dataset_root / path
            if not path.exists():
                array_errors += 1
                _error(errors, "array_exists", "Referenced array does not exist", path=str(path))
            elif array_hash and sha256_file(path) != array_hash:
                array_errors += 1
                _error(errors, "array_hash", "Referenced array hash differs", path=str(path))

    if not manifest_path.exists():
        _error(errors, "dataset_manifest", "dataset_manifest.json missing")
    if manifest.get("label_isolation") is not True:
        _error(errors, "label_isolation", "dataset manifest must declare label_isolation=true")
    if not (dataset_root / "labels_manifest.parquet").exists():
        _error(errors, "label_isolation", "Separate labels_manifest.parquet is missing")
    else:
        try:
            label_rows = reader.rows("labels_manifest.parquet")
            label_video_ids = [row.get("video_id") for row in label_rows]
            if len(label_video_ids) != len(set(label_video_ids)):
                _error(errors, "label_isolation", "labels_manifest contains duplicate video_id")
            unknown_label_ids = set(label_video_ids) - videos
            if unknown_label_ids:
                _error(errors, "label_foreign_key", "labels_manifest references unknown video_id", count=len(unknown_label_ids))
        except Exception as exc:
            _error(errors, "label_manifest_read", str(exc))
    forbidden = {"label", "label_name", "is_fake", "forgery_label", "tamper_label"}
    for relative_path, rows in tables.items():
        if rows and forbidden.intersection(rows[0]):
            _error(errors, "label_isolation", "Structural table contains forbidden label columns", table=relative_path)

    mask_rows = tables.get("observations/masks.parquet", [])
    for row in mask_rows:
        if bool(row.get("valid")) and (bool(row.get("bbox_fallback")) or not bool(row.get("is_visible_mask"))):
            _error(errors, "formal_mask", "Formal mask is a bbox fallback or not visible-mask evidence", mask_observation_id=row.get("mask_observation_id"))

    if is_p4b5:
        _validate_p4b5(
            dataset_root=dataset_root,
            tables=tables,
            manifest=manifest,
            errors=errors,
            warnings=warnings,
        )

    coverage_video_rows = _coverage_by_video(tables, videos)
    coverage_branch_rows = _coverage_by_branch(tables)
    failure_rows = _failure_reasons(tables)
    report = DatasetValidationReport(
        valid=not errors,
        checked_table_count=sum((dataset_root / path).exists() for path in primary_keys),
        error_count=len(errors),
        warning_count=len(warnings),
        errors=errors,
        warnings=warnings,
        metrics={
            "video_count": len(videos),
            "unique_frame_count": len(frame_ids),
            "clip_count": len(clips),
            "owned_frame_count": len(owner_counts),
            "array_error_count": array_errors,
        },
    )
    if write_reports:
        reports = dataset_root / "reports"
        reports.mkdir(parents=True, exist_ok=True)
        atomic_write_json(reports / "dataset_validation.json", report.to_dict())
        write_parquet(reports / "integrity_errors.parquet", errors, columns=("check", "detail", "context"))
        write_parquet(reports / "coverage_by_video.parquet", coverage_video_rows, columns=("video_id", "frame_count", "owned_frame_count", "shared_3d_valid_count", "shared_3d_owned_ratio", "valid_evidence_count"))
        write_parquet(reports / "coverage_by_branch.parquet", coverage_branch_rows, columns=("branch_name", "applicability", "count", "valid_count"))
        write_parquet(reports / "failure_reasons.parquet", failure_rows, columns=("source_table", "missing_reason", "count"))
        _write_csv(reports / "integrity_errors.csv", errors, ("check", "detail", "context"))
        _write_csv(reports / "coverage_by_video.csv", coverage_video_rows, ("video_id", "frame_count", "owned_frame_count", "shared_3d_valid_count", "shared_3d_owned_ratio", "valid_evidence_count"))
        _write_csv(reports / "coverage_by_branch.csv", coverage_branch_rows, ("branch_name", "applicability", "count", "valid_count"))
        _write_csv(reports / "failure_reasons.csv", failure_rows, ("source_table", "missing_reason", "count"))
    return report


def _nested(value: Any, default: Any) -> Any:
    if value in (None, ""):
        return default
    if isinstance(value, (dict, list, tuple)):
        return value
    try:
        return json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return default


def _validate_p4b5(
    *,
    dataset_root: Path,
    tables: Mapping[str, list[dict[str, Any]]],
    manifest: Mapping[str, Any],
    errors: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
) -> None:
    """Validate the P4-B.5 distinctions and provenance contracts."""

    depth_rows = tables.get("observations/depth.parquet", [])
    for row in depth_rows:
        if not row.get("valid"):
            continue
        if row.get("raw_array_name") != "raw_model_output":
            _error(errors, "p4b5_depth_raw", "Valid depth does not identify raw_model_output", frame_id=row.get("frame_id"))
        if row.get("geometry_array_name") not in {"depth_map", "relative_depth"}:
            _error(errors, "p4b5_depth_geometry", "Valid depth lacks canonical geometry array", frame_id=row.get("frame_id"))
        if bool(row.get("geometry_uses_visualization_depth")):
            _error(errors, "p4b5_visualization_depth", "Visualization depth entered geometry", frame_id=row.get("frame_id"))
        metadata = _nested(row.get("metadata"), {})
        if metadata.get("visualization_depth_used_by_geometry") is not False:
            _error(errors, "p4b5_visualization_depth", "Depth metadata does not exclude visualization depth", frame_id=row.get("frame_id"))
        path = Path(str(row.get("array_path", "")))
        if not path.is_absolute():
            path = dataset_root / path
        if path.exists():
            try:
                with np.load(path) as data:
                    if not {"raw_model_output", "depth_map", "valid_mask"}.issubset(data.files):
                        _error(errors, "p4b5_depth_arrays", "Depth archive lacks raw/canonical/valid arrays", path=str(path))
            except Exception as exc:
                _error(errors, "p4b5_depth_arrays", str(exc), path=str(path))

    shared_frames = tables.get("observations/shared_3d_frames.parquet", [])
    for row in shared_frames:
        if row.get("frame_geometry_scope") != "camera_frame_relative_sparse_3d":
            _error(errors, "p4b5_frame_geometry_scope", "Frame shared 3D is not explicitly frame-relative", frame_id=row.get("frame_id"))
        if bool(row.get("world_coordinates_available")) or bool(row.get("cross_frame_subtraction_allowed")):
            _error(errors, "p4b5_frame_sequence_isolation", "Per-frame relative geometry authorizes world/cross-frame operations", frame_id=row.get("frame_id"))
        if row.get("sequence_scale_status") != "relative_per_frame":
            _error(errors, "p4b5_frame_sequence_isolation", "Frame shared 3D masquerades as sequence-aligned", frame_id=row.get("frame_id"))

    point_2d = tables.get("observations/point_tracks_2d.parquet", [])
    point_3d = tables.get("observations/point_tracks_3d.parquet", [])
    source_2d_ids = {str(row["point_track_2d_observation_id"]) for row in point_2d}
    for row in point_2d:
        if not row.get("global_point_track_id") or not row.get("clip_point_track_id") or not row.get("global_object_track_id"):
            _error(errors, "p4b5_point_id_mapping", "Point global/local/object IDs are incomplete", observation_id=row.get("point_track_2d_observation_id"))
        if row.get("valid") and not bool(row.get("observed_independently")):
            _error(errors, "p4b5_independent_tracking", "Valid 2D point is not independently observed", observation_id=row.get("point_track_2d_observation_id"))
    masks_by_key = {
        (str(row.get("frame_id")), str(row.get("object_track_id"))): row
        for row in tables.get("observations/masks.parquet", []) if row.get("valid")
    }
    mask_cache: dict[str, np.ndarray] = {}
    for row in point_2d:
        if not row.get("valid") or row.get("point_role") != "internal_stable_point":
            continue
        mask_row = masks_by_key.get((str(row["frame_id"]), str(row["global_object_track_id"])))
        if mask_row is None:
            _error(errors, "p4b5_mask_point_support", "Formal internal point has no formal mask", observation_id=row.get("point_track_2d_observation_id"))
            continue
        path = Path(str(mask_row.get("array_path", "")))
        if not path.is_absolute():
            path = dataset_root / path
        key = str(path)
        if key not in mask_cache and path.exists():
            with np.load(path) as data:
                mask_cache[key] = np.asarray(data["visible_mask"], dtype=bool)
        mask = mask_cache.get(key)
        uv = _nested(row.get("pixel_uv"), None)
        if mask is None or uv is None:
            _error(errors, "p4b5_mask_point_support", "Formal internal point mask/pixel is unreadable", observation_id=row.get("point_track_2d_observation_id"))
            continue
        column, pixel_row = int(round(float(uv[0]))), int(round(float(uv[1])))
        if not (0 <= pixel_row < mask.shape[0] and 0 <= column < mask.shape[1] and mask[pixel_row, column]):
            _error(errors, "p4b5_mask_point_support", "Formal internal point lies outside its instance mask", observation_id=row.get("point_track_2d_observation_id"))
    for row in point_3d:
        if str(row.get("source_point_track_2d_id")) not in source_2d_ids:
            _error(errors, "p4b5_point_provenance", "3D point lacks a valid 2D source", observation_id=row.get("point_track_3d_observation_id"))
        if row.get("point_3d_world") not in (None, "", "null"):
            _error(errors, "p4b5_world_geometry", "P4-B.5 must not fabricate world points", observation_id=row.get("point_track_3d_observation_id"))
        if row.get("geometry_mode") == "unavailable" and row.get("valid"):
            _error(errors, "p4b5_unavailable_geometry", "Unavailable clip contains valid 3D trajectory evidence", observation_id=row.get("point_track_3d_observation_id"))

    evidence_rows = []
    for level in ("point", "edge", "object", "frame"):
        evidence_rows.extend(tables.get(f"evidence/{level}_evidence.parquet", []))
    owned_frame_ids = {
        str(row["frame_id"])
        for row in tables.get("manifests/frames.parquet", [])
        if row.get("is_owned_frame")
    }
    evidence_keys = []
    for row in evidence_rows:
        if row.get("frame_id") and str(row["frame_id"]) not in owned_frame_ids:
            _error(errors, "p4b5_context_evidence", "Context frame entered final dynamic evidence", evidence_id=row.get("evidence_id"))
        if row.get("valid") and not _nested(row.get("source_evidence_ids"), []):
            _error(errors, "p4b5_evidence_provenance", "Valid evidence has no traceable source IDs", evidence_id=row.get("evidence_id"))
        evidence_keys.append((
            row.get("branch_name"), row.get("frame_id"), row.get("object_track_id"),
            row.get("point_id"), row.get("edge_id"),
        ))
    duplicates = [key for key, count in Counter(evidence_keys).items() if count > 1]
    if duplicates:
        _error(errors, "p4b5_owned_evidence_unique", "Owned-frame evidence is duplicated", count=len(duplicates))

    handoffs = tables.get("observations/clip_track_handoffs.parquet", [])
    for row in handoffs:
        if bool(row.get("allows_cross_clip_3d")) and not row.get("alignment_id"):
            _error(errors, "p4b5_handoff_alignment", "Identity handoff authorizes 3D without alignment", handoff_id=row.get("handoff_id"))

    for row in tables.get("observations/structure_graphs.parquet", []):
        point_ids = set(_nested(row.get("point_ids"), []))
        edges = _nested(row.get("edges"), [])
        if row.get("valid") and not bool(row.get("fixed_topology")):
            _error(errors, "p4b5_fixed_graph", "Valid structure graph is not fixed", graph_id=row.get("structure_graph_id"))
        if any(len(edge) != 2 or not set(edge).issubset(point_ids) for edge in edges):
            _error(errors, "p4b5_fixed_graph", "Structure edge references unknown fixed point IDs", graph_id=row.get("structure_graph_id"))
        if row.get("graph_type") == "formal_mask_internal_fixed_graph" and row.get("point_source") != "formal_instance_mask_internal":
            _error(errors, "p4b5_formal_mask_graph", "Ordinary formal graph does not use formal-mask points", graph_id=row.get("structure_graph_id"))

    keypoints = tables.get("observations/keypoints.parquet", [])
    for row in keypoints:
        metadata = _nested(row.get("metadata"), {})
        if metadata.get("migrated_coverage_only") is not False or row.get("source_version") != P4B5_PIPELINE_VERSION:
            _error(errors, "p4b5_keypoint_source", "Legacy 8-frame coverage is presented as full inference", observation_id=row.get("keypoint_observation_id"))

    strict_hashes = manifest.get("strict_prior_hashes", {})
    for name in ("scale_priors_strict_v1.yaml", "scale_priors_strict_v2.yaml"):
        path = dataset_root.parents[2] / "configs" / name
        # Dataset roots may be moved; fall back to the configured project layout.
        if not path.exists():
            path = Path(__file__).resolve().parents[3] / "configs" / name
        if path.exists() and strict_hashes.get(name) != sha256_file(path):
            _error(errors, "p4b5_strict_prior_hash", "Frozen strict-prior hash changed", prior=name)

    coverage_names = {
        str(row.get("metric_name"))
        for row in tables.get("reports/coverage_metrics.parquet", [])
        if row.get("scope_type") == "dataset"
    }
    expected = {
        "frame_depth_coverage", "frame_shared_3d_coverage",
        "sequence_depth_aligned_coverage", "dynamic_3d_ready_coverage",
        "formal_dynamic_evidence_coverage",
    }
    if coverage_names != expected:
        _error(errors, "p4b5_coverage_semantics", "Five P4-B.5 coverage meanings are not all present", found=sorted(coverage_names))


def _group(rows: Iterable[Mapping[str, Any]], key: str) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key, ""))].append(row)
    return grouped


def _all_evidence(tables: Mapping[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for name in ("point", "edge", "object", "frame", "clip"):
        output.extend(tables.get(f"evidence/{name}_evidence.parquet", []))
    return output


def _coverage_by_video(tables: Mapping[str, list[dict[str, Any]]], videos: set[str]) -> list[dict[str, Any]]:
    frames = tables.get("manifests/frames.parquet", [])
    shared = tables.get("observations/shared_3d_frames.parquet", [])
    evidence = _all_evidence(tables)
    result = []
    for video_id in sorted(videos):
        unique_frames = {row["frame_id"] for row in frames if row.get("video_id") == video_id}
        owned = {row["frame_id"] for row in frames if row.get("video_id") == video_id and row.get("is_owned_frame")}
        valid_shared = {row["frame_id"] for row in shared if row.get("video_id") == video_id and row.get("valid")}
        result.append({
            "video_id": video_id,
            "frame_count": len(unique_frames),
            "owned_frame_count": len(owned),
            "shared_3d_valid_count": len(valid_shared),
            "shared_3d_owned_ratio": len(valid_shared) / len(owned) if owned else math.nan,
            "valid_evidence_count": sum(1 for row in evidence if row.get("video_id") == video_id and row.get("valid")),
        })
    return result


def _coverage_by_branch(tables: Mapping[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, str]] = Counter()
    valids: Counter[tuple[str, str]] = Counter()
    for row in _all_evidence(tables):
        key = (str(row.get("branch_name")), str(row.get("applicability")))
        counts[key] += 1
        valids[key] += int(bool(row.get("valid")))
    return [{"branch_name": key[0], "applicability": key[1], "count": count, "valid_count": valids[key]} for key, count in sorted(counts.items())]


def _failure_reasons(tables: Mapping[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows = []
    for source_table, values in tables.items():
        counts = Counter(str(row.get("missing_reason")) for row in values if row.get("missing_reason"))
        rows.extend({"source_table": source_table, "missing_reason": reason, "count": count} for reason, count in sorted(counts.items()))
    return rows


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], columns: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})
    temporary.replace(path)
