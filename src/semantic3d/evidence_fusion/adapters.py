"""Offline adapters from persisted M3/M4/M5 artifacts to unified M6 evidence."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .contracts import EvidenceBranchGroup, UnifiedEvidence


def _rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _float(value: Any, default: float = float("nan")) -> float:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return default
    return output


def _json(value: Any, default: Any) -> Any:
    if not isinstance(value, str) or not value.strip():
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def _aliased_clip(row: Mapping[str, str], aliases: Mapping[str, str]) -> str:
    clip_id = str(row.get("clip_id", ""))
    return str(aliases.get(clip_id, clip_id))


def _unified(
    *,
    evidence_id: str,
    value: float,
    valid: bool,
    applicable: bool,
    confidence: float,
    uncertainty: float,
    provider_status: str,
    failure_reason: str,
    branch_name: str,
    branch_group: EvidenceBranchGroup,
    row: Mapping[str, str],
    clip_aliases: Mapping[str, str],
    source_path: Path,
    frame_index: int,
    spatial_reference: Mapping[str, Any],
    temporal_reference: Mapping[str, Any],
) -> UnifiedEvidence:
    identity = {
        "object_id": str(row.get("object_id", "")),
        "track_id": str(row.get("track_id", row.get("object_track_id", ""))),
        "frame_id": str(row.get("frame_id", f"frame_{frame_index:06d}")),
        "video_id": str(row.get("video_id", "")),
        "clip_id": _aliased_clip(row, clip_aliases),
        "frame_index": frame_index,
        "spatial_reference": dict(spatial_reference),
        "temporal_reference": dict(temporal_reference),
        "provenance": {
            "source_artifact": str(source_path),
            "source_stage": source_path.parent.name,
            "authenticity_label_used": False,
            "provider_failure_used_as_risk": False,
        },
    }
    if valid and math.isfinite(value):
        return UnifiedEvidence.observed(
            evidence_id=evidence_id,
            residual_value=value,
            confidence=float(max(0.0, min(1.0, confidence))),
            uncertainty=uncertainty,
            provider_status=provider_status or "executed_valid",
            branch_name=branch_name,
            branch_group=branch_group,
            **identity,
        )
    return UnifiedEvidence.unavailable(
        evidence_id=evidence_id,
        applicable=applicable,
        provider_status=provider_status or "executed_invalid",
        failure_reason=failure_reason or "residual_unavailable",
        branch_name=branch_name,
        branch_group=branch_group,
        confidence=float(max(0.0, min(1.0, confidence))),
        uncertainty=uncertainty,
        **identity,
    )


def _adapt_metric_scale(
    path: Path,
    aliases: Mapping[str, str],
) -> list[UnifiedEvidence]:
    output = []
    for index, row in enumerate(_rows(path)):
        valid = _bool(row.get("valid"))
        applicable = _bool(row.get("applicable"))
        frame_index = int(row["frame_index"])
        output.append(
            _unified(
                evidence_id=f"m3:metric_scale:{index}:{row['video_id']}:{frame_index}",
                value=_float(row.get("combined_residual")),
                valid=valid,
                applicable=applicable,
                confidence=_float(row.get("quality_score"), 0.0),
                uncertainty=_float(row.get("uncertainty")),
                provider_status=str(row.get("provider_status", "")),
                failure_reason=str(row.get("failure_reason", "")),
                branch_name="metric_single_object_scale",
                branch_group=EvidenceBranchGroup.STATIC_METRIC_GEOMETRY,
                row=row,
                clip_aliases=aliases,
                source_path=path,
                frame_index=frame_index,
                spatial_reference={
                    "kind": "reference_only",
                    "object_id": row.get("object_id", ""),
                    "coordinate_frame": row.get("coordinate_frame", ""),
                },
                temporal_reference={"frame_index": frame_index},
            )
        )
    return output


def _adapt_temporal_scale(
    path: Path,
    aliases: Mapping[str, str],
) -> list[UnifiedEvidence]:
    output = []
    for index, row in enumerate(_rows(path)):
        valid = _bool(row.get("valid"))
        applicable = _bool(row.get("applicable"))
        frame_index = int(row["frame_index"])
        output.append(
            _unified(
                evidence_id=f"m3:temporal_scale:{index}:{row['track_id']}:{frame_index}",
                value=_float(row.get("residual")),
                valid=valid,
                applicable=applicable,
                confidence=_float(row.get("quality_score"), 0.0),
                uncertainty=_float(row.get("uncertainty")),
                provider_status=str(row.get("provider_status", "")),
                failure_reason=str(row.get("failure_reason", "")),
                branch_name=f"temporal_same_object_scale:{row.get('reference_method', '')}",
                branch_group=EvidenceBranchGroup.TEMPORAL_SCALE,
                row=row,
                clip_aliases=aliases,
                source_path=path,
                frame_index=frame_index,
                spatial_reference={
                    "kind": "reference_only",
                    "track_id": row.get("track_id", ""),
                },
                temporal_reference={
                    "frame_index": frame_index,
                    "reference_method": row.get("reference_method", ""),
                    "dimension_type": row.get("dimension_type", ""),
                },
            )
        )
    return output


def _d2_spatial_reference(
    evidence_type: str,
    metadata: Mapping[str, Any],
    *,
    point_id: str,
    source_image_shape: tuple[int, int],
) -> Mapping[str, Any]:
    predicted = metadata.get("predicted_uv")
    if evidence_type == "point" and predicted is not None:
        return {
            "kind": "point",
            "xy": predicted,
            "point_id": point_id,
            "source_image_shape": source_image_shape,
        }
    if evidence_type == "boundary" and predicted is not None:
        return {
            "kind": "boundary",
            "xy": predicted,
            "point_id": point_id,
            "source_image_shape": source_image_shape,
        }
    return {"kind": "reference_only", "point_id": point_id}


def _source_image_shape(
    row: Mapping[str, str],
    *,
    source_image_shapes: Mapping[str, tuple[int, int]],
    fallback: tuple[int, int],
) -> tuple[int, int]:
    """Resolve the source image shape without assuming one video orientation."""

    video_id = str(row.get("video_id", ""))
    return source_image_shapes.get(video_id, fallback)


def _adapt_d2(
    paths: Mapping[str, Path],
    aliases: Mapping[str, str],
    *,
    source_image_shapes: Mapping[str, tuple[int, int]],
    fallback_source_image_shape: tuple[int, int],
) -> list[UnifiedEvidence]:
    output: list[UnifiedEvidence] = []
    boundary_support: dict[tuple[str, str, int, str], list[tuple[float, float]]] = {}
    for evidence_type in ("point", "boundary"):
        path = paths[evidence_type]
        for row in _rows(path):
            if not _bool(row.get("valid")):
                continue
            metadata = _json(row.get("metadata"), {})
            predicted = metadata.get("predicted_uv")
            if evidence_type == "boundary" and predicted is not None:
                key = (
                    row["video_id"],
                    _aliased_clip(row, aliases),
                    int(row["frame_t1"]),
                    row.get("track_id", ""),
                )
                boundary_support.setdefault(key, []).append(
                    (float(predicted[0]), float(predicted[1]))
                )
    fields = {
        "point": "point_reprojection_residual",
        "boundary": "boundary_reprojection_residual",
        "object": "object_reprojection_residual",
    }
    for evidence_type, path in paths.items():
        for index, row in enumerate(_rows(path)):
            valid = _bool(row.get("valid"))
            failure = str(row.get("failure_reason", ""))
            visibility = str(row.get("visibility_status", ""))
            applicable = valid or not (
                visibility in {"out_of_frame", "occluded", "depth_conflict"}
            )
            frame_index = int(row["frame_t1"])
            metadata = _json(row.get("metadata"), {})
            source_image_shape = _source_image_shape(
                row,
                source_image_shapes=source_image_shapes,
                fallback=fallback_source_image_shape,
            )
            spatial = _d2_spatial_reference(
                evidence_type,
                metadata,
                point_id=str(row.get("point_id", "")),
                source_image_shape=source_image_shape,
            )
            if evidence_type == "object":
                key = (
                    row["video_id"],
                    _aliased_clip(row, aliases),
                    frame_index,
                    row.get("track_id", ""),
                )
                support = boundary_support.get(key, [])
                spatial = {
                    "kind": "object_support_points" if support else "reference_only",
                    "xy_points": support,
                    "source_image_shape": source_image_shape,
                    "support_is_visible_boundary_not_mask": True,
                }
            confidence = min(
                _float(row.get("pose_confidence"), 0.0),
                _float(row.get("point_confidence"), 0.0),
            )
            output.append(
                _unified(
                    evidence_id=str(
                        row.get("evidence_id", f"m4:d2:{evidence_type}:{index}")
                    ),
                    value=_float(row.get(fields[evidence_type])),
                    valid=valid,
                    applicable=applicable,
                    confidence=confidence,
                    uncertainty=(1.0 - confidence if valid else float("nan")),
                    provider_status=str(row.get("provider_status", "")),
                    failure_reason=failure or (
                        f"d2_{visibility}_not_applicable" if not valid else ""
                    ),
                    branch_name=f"D2_{evidence_type}_reprojection",
                    branch_group=EvidenceBranchGroup.D2_POSE_REPROJECTION,
                    row=row,
                    clip_aliases=aliases,
                    source_path=path,
                    frame_index=frame_index,
                    spatial_reference=spatial,
                    temporal_reference={
                        "frame_t": int(row["frame_t"]),
                        "frame_t1": frame_index,
                    },
                )
            )
    return output


def _d3_group(residual_name: str) -> EvidenceBranchGroup:
    if residual_name == "R_object_boundary_relation":
        return EvidenceBranchGroup.BOUNDARY_STRUCTURE
    if residual_name in {"R_edge_length", "R_local_rigidity"}:
        return EvidenceBranchGroup.INTERNAL_STRUCTURE
    return EvidenceBranchGroup.D3_STRUCTURAL_RELATION


def _adapt_d3(path: Path, aliases: Mapping[str, str]) -> list[UnifiedEvidence]:
    output = []
    for row in _rows(path):
        if row.get("source_kind") != "persisted_video":
            continue
        valid = _bool(row.get("valid"))
        frame_index = int(row["frame_t1"])
        localization = _json(row.get("localization_reference"), {})
        localization = {
            "kind": "reference_only",
            **dict(localization),
        }
        output.append(
            _unified(
                evidence_id=row["residual_id"],
                value=_float(row.get("value")),
                valid=valid,
                applicable=True,
                confidence=_float(row.get("confidence"), 0.0),
                uncertainty=(
                    1.0 - _float(row.get("confidence"), 0.0)
                    if valid
                    else float("nan")
                ),
                provider_status="executed_valid" if valid else "blocked_by_input",
                failure_reason=str(row.get("failure_reason", "")),
                branch_name=row["residual_name"],
                branch_group=_d3_group(row["residual_name"]),
                row=row,
                clip_aliases=aliases,
                source_path=path,
                frame_index=frame_index,
                spatial_reference=localization,
                temporal_reference={
                    "frame_t": int(row["frame_t"]),
                    "frame_t1": frame_index,
                    "source_edge": row.get("source_edge", ""),
                },
            )
        )
    return output


def _adapt_occlusion_events(
    path: Path,
    aliases: Mapping[str, str],
) -> list[UnifiedEvidence]:
    output = []
    for row in _rows(path):
        if row.get("source_kind") != "persisted_video":
            continue
        frame_index = int(row["frame_index"])
        status = str(row.get("status", ""))
        reason = str(row.get("failure_reason", "")) or "occlusion_residual_unavailable"
        output.append(
            _unified(
                evidence_id=row["event_id"],
                value=float("nan"),
                valid=False,
                applicable=status != "not_applicable",
                confidence=0.0,
                uncertainty=float("nan"),
                provider_status=status or "executed_invalid",
                failure_reason=reason,
                branch_name="occlusion_or_visibility_event",
                branch_group=EvidenceBranchGroup.OCCLUSION_REAPPEARANCE,
                row=row,
                clip_aliases=aliases,
                source_path=path,
                frame_index=frame_index,
                spatial_reference={
                    "kind": "reference_only",
                    **dict(_json(row.get("localization_reference"), {})),
                },
                temporal_reference={"frame_index": frame_index},
            )
        )
    return output


def load_persisted_unified_evidence(
    project_root: str | Path,
    config: Mapping[str, Any],
) -> tuple[tuple[UnifiedEvidence, ...], Mapping[str, Any]]:
    """Load existing M3/M4/M5 artifacts without running any provider."""

    root = Path(project_root).resolve()
    inputs = config["inputs"]
    aliases = {str(key): str(value) for key, value in config["clip_aliases"].items()}
    localization = config["localization"]
    fallback_image_shape = tuple(
        int(value) for value in localization["source_image_shape"]
    )
    source_image_shapes = {
        str(video_id): tuple(int(value) for value in shape)
        for video_id, shape in localization.get("source_image_shapes", {}).items()
    }
    paths = {
        name: root / relative
        for name, relative in inputs.items()
    }
    evidence: list[UnifiedEvidence] = []
    evidence.extend(_adapt_metric_scale(paths["metric_scale"], aliases))
    evidence.extend(_adapt_temporal_scale(paths["temporal_scale"], aliases))
    evidence.extend(
        _adapt_d2(
            {
                "point": paths["d2_point"],
                "boundary": paths["d2_boundary"],
                "object": paths["d2_object"],
            },
            aliases,
            source_image_shapes=source_image_shapes,
            fallback_source_image_shape=fallback_image_shape,
        )
    )
    evidence.extend(_adapt_d3(paths["d3_relations"], aliases))
    evidence.extend(_adapt_occlusion_events(paths["occlusion_events"], aliases))
    return tuple(evidence), {
        "input_paths": {name: str(path) for name, path in sorted(paths.items())},
        "input_exists": {name: path.exists() for name, path in sorted(paths.items())},
        "loaded_evidence_count": len(evidence),
        "provider_inference_executed": False,
        "authenticity_labels_read": False,
        "source_image_shapes": {
            video_id: list(shape)
            for video_id, shape in sorted(source_image_shapes.items())
        },
        "fallback_source_image_shape": list(fallback_image_shape),
    }
