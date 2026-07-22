"""Five-level traceable aggregation from localized evidence to one clip."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .aggregation import aggregate_evidence_v2
from .applicability import AggregationEvidence, EvidenceApplicability
from .contracts import (
    ClipEvidenceAggregate,
    EdgeEvidenceAggregate,
    FrameEvidenceAggregate,
    ObjectEvidenceAggregate,
    PointEvidenceAggregate,
    _EvidenceAggregate,
)
from .temporal_localization import TemporalInterval, localize_temporal_intervals


def _status_from_aggregate(item: _EvidenceAggregate) -> EvidenceApplicability:
    try:
        return EvidenceApplicability(item.missing_reason)
    except ValueError:
        return EvidenceApplicability.OBSERVATION_MISSING


def _aggregate_as_evidence(item: _EvidenceAggregate, *, source_id: str) -> AggregationEvidence:
    branch = (
        item.contributing_branch_names[0]
        if len(item.contributing_branch_names) == 1 else "multibranch_aggregate"
    )
    frame_index = getattr(item, "frame_index", None)
    object_track_id = str(getattr(item, "object_track_id", ""))
    point_or_edge_id = str(getattr(item, "point_id", "") or getattr(item, "edge_id", ""))
    metadata = {
        "aggregate_level": type(item).__name__,
        "contributing_branch_names": item.contributing_branch_names,
        "coverage": item.coverage,
    }
    if item.valid:
        return AggregationEvidence.observed(
            item.value, quality=item.quality, branch_name=branch,
            source_id=source_id, frame_index=frame_index,
            object_track_id=object_track_id, point_or_edge_id=point_or_edge_id,
            metadata=metadata,
        )
    return AggregationEvidence.unavailable(
        applicability=_status_from_aggregate(item), reason=item.missing_reason,
        branch_name=branch, source_id=source_id, frame_index=frame_index,
        object_track_id=object_track_id, point_or_edge_id=point_or_edge_id,
        metadata=metadata,
    )


def _group_records(
    records: Sequence[AggregationEvidence],
    *,
    include_identity: bool,
) -> dict[tuple[Any, ...], list[AggregationEvidence]]:
    grouped: dict[tuple[Any, ...], list[AggregationEvidence]] = {}
    for item in records:
        video_id = str(item.metadata.get("video_id", ""))
        key = (video_id, item.frame_index, item.object_track_id)
        if include_identity:
            key += (item.point_or_edge_id,)
        grouped.setdefault(key, []).append(item)
    return grouped


def aggregate_point_evidence(
    evidences: Sequence[AggregationEvidence],
    *,
    method: str = "quality_weighted_top_k",
    top_k: int = 2,
    quality_floor: float = 0.0,
) -> tuple[PointEvidenceAggregate, ...]:
    """Aggregate branches attached to the same point in the same frame."""

    output = []
    for (video_id, frame_index, track_id, point_id), rows in sorted(
        _group_records(evidences, include_identity=True).items(), key=lambda item: str(item[0])
    ):
        output.append(aggregate_evidence_v2(
            rows, level="point", method=method, top_k=top_k,
            quality_floor=quality_floor,
            identity={
                "video_id": video_id, "frame_index": -1 if frame_index is None else frame_index,
                "object_track_id": track_id, "point_id": point_id,
            },
        ))
    return tuple(output)


def aggregate_edge_evidence(
    evidences: Sequence[AggregationEvidence],
    *,
    method: str = "quality_weighted_top_k",
    top_k: int = 2,
    quality_floor: float = 0.0,
) -> tuple[EdgeEvidenceAggregate, ...]:
    """Aggregate branches attached to the same fixed edge and frame."""

    output = []
    for (video_id, frame_index, track_id, edge_id), rows in sorted(
        _group_records(evidences, include_identity=True).items(), key=lambda item: str(item[0])
    ):
        output.append(aggregate_evidence_v2(
            rows, level="edge", method=method, top_k=top_k,
            quality_floor=quality_floor,
            identity={
                "video_id": video_id, "frame_index": -1 if frame_index is None else frame_index,
                "object_track_id": track_id, "edge_id": edge_id,
            },
        ))
    return tuple(output)


def _branch_scores(items: Sequence[_EvidenceAggregate]) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for item in items:
        for contributor in item.top_contributors:
            if math.isfinite(float(contributor["value"])):
                grouped.setdefault(str(contributor["branch_name"]), []).append(float(contributor["value"]))
    return {name: float(np.mean(values)) for name, values in sorted(grouped.items())}


def aggregate_object_evidence(
    points: Sequence[PointEvidenceAggregate],
    edges: Sequence[EdgeEvidenceAggregate],
    *,
    object_metadata: Mapping[tuple[str, int, str], Mapping[str, Any]] | None = None,
    method: str = "hybrid_median_top_k",
    top_k: int = 3,
    quality_floor: float = 0.0,
) -> tuple[ObjectEvidenceAggregate, ...]:
    """Aggregate localized point/edge evidence into object-frame scores."""

    grouped_points: dict[tuple[str, int, str], list[PointEvidenceAggregate]] = {}
    grouped_edges: dict[tuple[str, int, str], list[EdgeEvidenceAggregate]] = {}
    for item in points:
        grouped_points.setdefault((item.video_id, item.frame_index, item.object_track_id), []).append(item)
    for item in edges:
        grouped_edges.setdefault((item.video_id, item.frame_index, item.object_track_id), []).append(item)
    output = []
    metadata_lookup = dict(object_metadata or {})
    for key in sorted(set(grouped_points) | set(grouped_edges)):
        point_rows, edge_rows = grouped_points.get(key, []), grouped_edges.get(key, [])
        rows: list[_EvidenceAggregate] = [*point_rows, *edge_rows]
        adapted = [_aggregate_as_evidence(item, source_id=f"{type(item).__name__}:{index}") for index, item in enumerate(rows)]
        valid_points = [item for item in point_rows if item.valid]
        valid_edges = [item for item in edge_rows if item.valid]
        top_points = tuple(item.point_id for item in sorted(valid_points, key=lambda row: row.value * row.quality, reverse=True)[:top_k])
        top_edges = tuple(item.edge_id for item in sorted(valid_edges, key=lambda row: row.value * row.quality, reverse=True)[:top_k])
        info = dict(metadata_lookup.get(key, {}))
        bbox = info.get("localization_bbox")
        if bbox is not None:
            bbox = tuple(float(value) for value in bbox)
        spatial_count = len(point_rows) + len(edge_rows)
        spatial_valid = len(valid_points) + len(valid_edges)
        output.append(aggregate_evidence_v2(
            adapted, level="object", method=method, top_k=top_k,
            quality_floor=quality_floor,
            coverage_dimensions={
                "branch_coverage": len({name for item in rows for name in item.contributing_branch_names}) / max(1, len({name for item in rows for name in item.contributing_branch_names})),
                "object_coverage": 1.0 if adapted else 0.0,
                "temporal_coverage": 1.0,
                "spatial_coverage": spatial_valid / spatial_count if spatial_count else 0.0,
            },
            identity={
                "video_id": key[0], "frame_index": key[1], "object_track_id": key[2],
                "semantic_label": str(info.get("semantic_label", "")),
                "branch_scores": _branch_scores(rows),
                "valid_point_ratio": len(valid_points) / len(point_rows) if point_rows else float("nan"),
                "valid_edge_ratio": len(valid_edges) / len(edge_rows) if edge_rows else float("nan"),
                "top_anomalous_point_ids": top_points,
                "top_anomalous_edge_ids": top_edges,
                "localization_bbox": bbox,
                "localization_mask_reference": str(info.get("localization_mask_reference", "")),
            },
        ))
    return tuple(output)


def aggregate_frame_evidence(
    objects: Sequence[ObjectEvidenceAggregate],
    *,
    method: str = "hybrid_median_top_k",
    top_k: int = 2,
    quality_floor: float = 0.0,
) -> tuple[FrameEvidenceAggregate, ...]:
    """Aggregate object-frame evidence and retain spatial contributors."""

    grouped: dict[tuple[str, int], list[ObjectEvidenceAggregate]] = {}
    for item in objects:
        grouped.setdefault((item.video_id, item.frame_index), []).append(item)
    output = []
    for (video_id, frame_index), rows in sorted(grouped.items()):
        adapted = [_aggregate_as_evidence(item, source_id=f"object:{item.object_track_id}:{frame_index}") for item in rows]
        valid = sorted((item for item in rows if item.valid), key=lambda item: item.value * item.quality, reverse=True)
        branches = sorted({branch for item in rows for branch in item.contributing_branch_names})
        branch_coverage = {
            branch: sum(item.valid and branch in item.contributing_branch_names for item in rows) / len(rows)
            for branch in branches
        }
        output.append(aggregate_evidence_v2(
            adapted, level="frame", method=method, top_k=top_k,
            quality_floor=quality_floor,
            coverage_dimensions={
                "branch_coverage": float(np.mean(list(branch_coverage.values()))) if branch_coverage else 0.0,
                "object_coverage": len(valid) / len(rows) if rows else 0.0,
                "temporal_coverage": 1.0,
                "spatial_coverage": float(np.mean([item.coverage_dimensions.get("spatial_coverage", item.coverage) for item in rows])) if rows else 0.0,
            },
            identity={
                "video_id": video_id, "frame_index": frame_index,
                "object_scores": {item.object_track_id: item.value for item in rows},
                "active_branches": tuple(branches), "branch_coverage": branch_coverage,
                "top_object_ids": tuple(item.object_track_id for item in valid[:top_k]),
                "top_point_ids": tuple(dict.fromkeys(point for item in valid for point in item.top_anomalous_point_ids)),
                "top_edge_ids": tuple(dict.fromkeys(edge for item in valid for edge in item.top_anomalous_edge_ids)),
            },
        ))
    return tuple(output)


def aggregate_clip_evidence(
    frames: Sequence[FrameEvidenceAggregate],
    *,
    video_id: str,
    clip_id: str,
    method: str = "hybrid_median_top_k",
    top_k: int = 3,
    quality_floor: float = 0.0,
    intervals: Sequence[TemporalInterval] = (),
) -> ClipEvidenceAggregate:
    """Aggregate frames while preserving raw scores and temporal intervals."""

    ordered = tuple(sorted(frames, key=lambda item: item.frame_index))
    adapted = [_aggregate_as_evidence(item, source_id=f"frame:{item.frame_index}") for item in ordered]
    valid = [item for item in ordered if item.valid]
    branches = sorted({branch for item in ordered for branch in item.contributing_branch_names})
    branch_coverage = {
        branch: sum(item.valid and branch in item.contributing_branch_names for item in ordered) / len(ordered)
        for branch in branches
    } if ordered else {}
    values = np.asarray([item.value for item in valid], dtype=float)
    top_count = min(top_k, len(values))
    interval_score = float(np.mean([item.score for item in intervals])) if intervals else float("nan")
    top_objects = tuple(dict.fromkeys(obj for item in valid for obj in item.top_object_ids))
    regions = tuple(f"mask:{obj}" for obj in top_objects)
    return aggregate_evidence_v2(
        adapted, level="clip", method=method, top_k=top_k,
        quality_floor=quality_floor,
        coverage_dimensions={
            "branch_coverage": float(np.mean(list(branch_coverage.values()))) if branch_coverage else 0.0,
            "object_coverage": float(np.mean([item.coverage_dimensions.get("object_coverage", item.coverage) for item in ordered])) if ordered else 0.0,
            "temporal_coverage": len(valid) / len(ordered) if ordered else 0.0,
            "spatial_coverage": float(np.mean([item.coverage_dimensions.get("spatial_coverage", item.coverage) for item in ordered])) if ordered else 0.0,
        },
        identity={
            "video_id": video_id, "clip_id": clip_id,
            "frame_score_sequence": tuple(item.value for item in ordered),
            "frame_indices": tuple(item.frame_index for item in ordered),
            "peak_score": float(np.max(values)) if len(values) else float("nan"),
            "top_k_frame_mean": float(np.mean(np.sort(values)[-top_count:])) if top_count else float("nan"),
            "persistent_interval_score": interval_score,
            "candidate_intervals": tuple(asdict(item) for item in intervals),
            "valid_frame_ratio": len(valid) / len(ordered) if ordered else 0.0,
            "branch_coverage": branch_coverage,
            "top_objects": top_objects[:top_k],
            "top_spatial_regions": regions[:top_k],
        },
    )


@dataclass(frozen=True)
class MultilevelAggregationResult:
    point_aggregates: tuple[PointEvidenceAggregate, ...]
    edge_aggregates: tuple[EdgeEvidenceAggregate, ...]
    object_aggregates: tuple[ObjectEvidenceAggregate, ...]
    frame_aggregates: tuple[FrameEvidenceAggregate, ...]
    clip_aggregate: ClipEvidenceAggregate
    smoothed_frame_scores: tuple[float, ...]
    intervals: tuple[TemporalInterval, ...]


def aggregate_multilevel_evidence(
    *,
    point_evidences: Sequence[AggregationEvidence],
    edge_evidences: Sequence[AggregationEvidence],
    video_id: str,
    clip_id: str,
    object_metadata: Mapping[tuple[str, int, str], Mapping[str, Any]] | None = None,
    method: str = "hybrid_median_top_k",
    top_k: int = 3,
    quality_floor: float = 0.0,
    temporal_threshold: float | None = None,
    moving_median_window: int = 3,
    max_gap: int = 1,
    minimum_duration: int = 2,
) -> MultilevelAggregationResult:
    """Run point, edge, object, frame, clip and temporal aggregation."""

    points = aggregate_point_evidence(point_evidences, method=method, top_k=top_k, quality_floor=quality_floor)
    edges = aggregate_edge_evidence(edge_evidences, method=method, top_k=top_k, quality_floor=quality_floor)
    objects = aggregate_object_evidence(points, edges, object_metadata=object_metadata, method=method, top_k=top_k, quality_floor=quality_floor)
    frames = aggregate_frame_evidence(objects, method=method, top_k=top_k, quality_floor=quality_floor)
    if temporal_threshold is None:
        smoothed = np.asarray([item.value for item in frames], dtype=float)
        intervals: tuple[TemporalInterval, ...] = ()
    else:
        smoothed, intervals = localize_temporal_intervals(
            [item.frame_index for item in frames], [item.value for item in frames],
            threshold=temporal_threshold, moving_median_window=moving_median_window,
            max_gap=max_gap, minimum_duration=minimum_duration,
            qualities=[item.quality for item in frames],
        )
    clip = aggregate_clip_evidence(
        frames, video_id=video_id, clip_id=clip_id, method=method,
        top_k=top_k, quality_floor=quality_floor, intervals=intervals,
    )
    return MultilevelAggregationResult(
        points, edges, objects, frames, clip,
        tuple(float(value) for value in smoothed), intervals,
    )
