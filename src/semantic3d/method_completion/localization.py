"""Threshold-free mapping from residual evidence to spatial/temporal support."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from ..validity import ResidualEvidence


@dataclass(frozen=True)
class PointResidualLocation:
    evidence: ResidualEvidence
    frame_index: int
    xy: tuple[float, float]
    point_id: str
    track_id: str = ""


@dataclass(frozen=True)
class TrackResidualLocation:
    evidence: ResidualEvidence
    track_id: str
    points_by_frame: Mapping[int, tuple[float, float]]


@dataclass(frozen=True)
class ObjectResidualLocation:
    evidence: ResidualEvidence
    frame_index: int
    object_id: str
    mask: np.ndarray


@dataclass(frozen=True)
class BoundaryResidualLocation:
    evidence: ResidualEvidence
    frame_index: int
    object_id: str
    boundary_xy: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class PairResidualLocation:
    """One pair residual mapped to both object masks and their relation edge."""

    evidence: ResidualEvidence
    frame_index: int
    pair_id: str
    object_a_id: str
    object_b_id: str
    object_a_mask: np.ndarray
    object_b_mask: np.ndarray
    object_a_center: tuple[float, float]
    object_b_center: tuple[float, float]


@dataclass(frozen=True)
class LocalizationEvidenceBundle:
    """Spatial and temporal evidence products without anomaly decisions."""

    frame_residual_map: Mapping[int, np.ndarray]
    object_scores: Mapping[str, float]
    track_scores: Mapping[str, float]
    spatial_evidence_map: Mapping[int, np.ndarray]
    temporal_evidence_sequence: Mapping[int, float]
    valid_source_ids: tuple[str, ...]
    skipped_source_reasons: Mapping[str, int]
    relation_edges: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _place(array: np.ndarray, x: float, y: float, value: float) -> bool:
    column, row = int(round(float(x))), int(round(float(y)))
    if row < 0 or column < 0 or row >= array.shape[0] or column >= array.shape[1]:
        return False
    current = array[row, column]
    array[row, column] = value if math.isnan(float(current)) else max(float(current), value)
    return True


def _score(mapping: dict[str, float], key: str, value: float) -> None:
    mapping[key] = max(mapping.get(key, -math.inf), value)


def map_residual_evidence(
    *,
    image_shape: tuple[int, int],
    frame_indices: Sequence[int],
    point_residuals: Sequence[PointResidualLocation] = (),
    track_residuals: Sequence[TrackResidualLocation] = (),
    object_residuals: Sequence[ObjectResidualLocation] = (),
    boundary_residuals: Sequence[BoundaryResidualLocation] = (),
    pair_residuals: Sequence[PairResidualLocation] = (),
) -> LocalizationEvidenceBundle:
    """Map valid residuals to their native supports using max evidence fusion.

    Missing and provider-failed evidence is skipped and recorded.  Empty frame
    pixels and frames without evidence remain NaN, never zero.
    """

    height, width = (int(image_shape[0]), int(image_shape[1]))
    if height <= 0 or width <= 0:
        raise ValueError("image_shape must be positive.")
    ordered_frames = tuple(sorted({int(frame) for frame in frame_indices}))
    if not ordered_frames:
        raise ValueError("frame_indices must not be empty.")
    maps = {
        frame: np.full((height, width), np.nan, dtype=np.float32)
        for frame in ordered_frames
    }
    object_scores: dict[str, float] = {}
    track_scores: dict[str, float] = {}
    temporal: dict[int, float] = {frame: float("nan") for frame in ordered_frames}
    valid_sources: list[str] = []
    skipped: dict[str, int] = {}
    relation_edges: dict[str, Mapping[str, Any]] = {}

    def accept(evidence: ResidualEvidence) -> bool:
        if evidence.valid:
            valid_sources.extend(evidence.source_ids or (evidence.name,))
            return True
        reason = evidence.missing_reason or "missing_residual_evidence"
        skipped[reason] = skipped.get(reason, 0) + 1
        return False

    def update_temporal(frame: int, value: float) -> None:
        current = temporal[frame]
        temporal[frame] = value if math.isnan(current) else max(current, value)

    for item in point_residuals:
        if item.frame_index not in maps or not accept(item.evidence):
            continue
        if _place(maps[item.frame_index], *item.xy, item.evidence.value):
            update_temporal(item.frame_index, item.evidence.value)
            if item.track_id:
                _score(track_scores, item.track_id, item.evidence.value)

    for item in track_residuals:
        if not accept(item.evidence):
            continue
        placed = False
        for frame, xy in sorted(item.points_by_frame.items()):
            if frame in maps and _place(maps[frame], *xy, item.evidence.value):
                update_temporal(frame, item.evidence.value)
                placed = True
        if placed:
            _score(track_scores, item.track_id, item.evidence.value)

    for item in object_residuals:
        if item.frame_index not in maps or not accept(item.evidence):
            continue
        mask = np.asarray(item.mask, dtype=bool)
        if mask.shape != (height, width):
            raise ValueError("Object mask shape must match image_shape.")
        if not np.any(mask):
            skipped["empty_object_mask"] = skipped.get("empty_object_mask", 0) + 1
            continue
        target = maps[item.frame_index]
        target[mask] = np.fmax(target[mask], item.evidence.value)
        _score(object_scores, item.object_id, item.evidence.value)
        update_temporal(item.frame_index, item.evidence.value)

    for item in boundary_residuals:
        if item.frame_index not in maps or not accept(item.evidence):
            continue
        placed = False
        for x, y in item.boundary_xy:
            placed = _place(
                maps[item.frame_index], x, y, item.evidence.value
            ) or placed
        if placed:
            _score(object_scores, item.object_id, item.evidence.value)
            update_temporal(item.frame_index, item.evidence.value)

    for item in pair_residuals:
        if item.frame_index not in maps or not accept(item.evidence):
            continue
        mask_a = np.asarray(item.object_a_mask, dtype=bool)
        mask_b = np.asarray(item.object_b_mask, dtype=bool)
        if mask_a.shape != (height, width) or mask_b.shape != (height, width):
            raise ValueError("Pair object masks must match image_shape.")
        if not np.any(mask_a) or not np.any(mask_b):
            skipped["empty_pair_object_mask"] = skipped.get("empty_pair_object_mask", 0) + 1
            continue
        target = maps[item.frame_index]
        target[mask_a] = np.fmax(target[mask_a], item.evidence.value)
        target[mask_b] = np.fmax(target[mask_b], item.evidence.value)
        x1, y1 = item.object_a_center
        x2, y2 = item.object_b_center
        sample_count = max(2, int(round(math.hypot(x2 - x1, y2 - y1))) + 1)
        edge_points = []
        for x, y in zip(np.linspace(x1, x2, sample_count), np.linspace(y1, y2, sample_count)):
            if _place(target, x, y, item.evidence.value):
                edge_points.append((float(x), float(y)))
        _score(object_scores, item.object_a_id, item.evidence.value)
        _score(object_scores, item.object_b_id, item.evidence.value)
        update_temporal(item.frame_index, item.evidence.value)
        relation_edges[item.pair_id] = {
            "frame_index": item.frame_index,
            "object_a_id": item.object_a_id,
            "object_b_id": item.object_b_id,
            "residual": item.evidence.value,
            "edge_points": edge_points,
        }

    return LocalizationEvidenceBundle(
        frame_residual_map=maps,
        object_scores=dict(sorted(object_scores.items())),
        track_scores=dict(sorted(track_scores.items())),
        spatial_evidence_map=maps,
        temporal_evidence_sequence=temporal,
        valid_source_ids=tuple(valid_sources),
        skipped_source_reasons=dict(sorted(skipped.items())),
        relation_edges=dict(sorted(relation_edges.items())),
        metadata={
            "aggregation": "per-support maximum without learned threshold",
            "missing_pixels_are_nan": True,
            "provider_failure_used_as_anomaly": False,
            "final_anomaly_decision": False,
        },
    )


def rank_object_scale_evidence(
    object_scores: Mapping[str, float],
) -> tuple[tuple[str, float], ...]:
    """Rank finite object evidence without choosing an anomaly threshold."""

    return tuple(
        sorted(
            (
                (str(object_id), float(score))
                for object_id, score in object_scores.items()
                if math.isfinite(float(score))
            ),
            key=lambda item: (-item[1], item[0]),
        )
    )
