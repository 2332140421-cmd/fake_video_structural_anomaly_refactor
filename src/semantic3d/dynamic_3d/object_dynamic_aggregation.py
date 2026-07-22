"""Validity-aware object aggregation for point and fixed-edge residuals."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from ..validity import ResidualEvidence


@dataclass(frozen=True)
class ObjectDynamicAggregate:
    """Robust object-level summary that retains localization identities."""

    object_track_id: str
    point_residuals: tuple[ResidualEvidence, ...]
    edge_residuals: tuple[ResidualEvidence, ...]
    median: ResidualEvidence
    trimmed_mean: ResidualEvidence
    topk_mean: ResidualEvidence
    valid_point_ratio: float
    valid_edge_ratio: float
    top_anomalous_points: tuple[str, ...]
    top_anomalous_edges: tuple[str, ...]
    quality: float
    valid: bool
    missing_reason: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for value in (self.valid_point_ratio, self.valid_edge_ratio, self.quality):
            if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise ValueError("Aggregate ratios and quality must be in [0, 1].")
        if self.valid and (not self.median.valid or self.missing_reason):
            raise ValueError("Valid aggregate requires valid robust statistics.")
        if not self.valid and not self.missing_reason:
            raise ValueError("Invalid aggregate requires missing_reason.")
        object.__setattr__(self, "metadata", dict(self.metadata))


def _trimmed_mean(values: np.ndarray, trim_fraction: float) -> float:
    ordered = np.sort(values)
    count = int(math.floor(len(ordered) * trim_fraction))
    if count and len(ordered) > 2 * count:
        ordered = ordered[count:-count]
    return float(np.mean(ordered))


def aggregate_object_dynamic_evidence(
    object_track_id: str,
    point_evidence_by_id: Sequence[tuple[str, ResidualEvidence]],
    edge_evidence_by_id: Sequence[tuple[str, ResidualEvidence]],
    *,
    trim_fraction: float = 0.10,
    topk: int = 3,
) -> ObjectDynamicAggregate:
    """Aggregate only valid evidence with median, trimmed mean, and top-k mean."""

    if not 0.0 <= trim_fraction < 0.5 or topk <= 0:
        raise ValueError("Invalid robust aggregation parameters.")
    point_items = tuple(point_evidence_by_id)
    edge_items = tuple(edge_evidence_by_id)
    valid_points = [(identity, evidence) for identity, evidence in point_items if evidence.valid]
    valid_edges = [(identity, evidence) for identity, evidence in edge_items if evidence.valid]
    combined = valid_points + valid_edges
    if not combined:
        missing = ResidualEvidence.missing("r_object_dynamic", "no_valid_dynamic_evidence", source_ids=(object_track_id,))
        return ObjectDynamicAggregate(
            object_track_id, tuple(item[1] for item in point_items), tuple(item[1] for item in edge_items),
            missing, ResidualEvidence.missing("r_object_dynamic_trimmed_mean", "no_valid_dynamic_evidence"),
            ResidualEvidence.missing("r_object_dynamic_topk_mean", "no_valid_dynamic_evidence"),
            0.0 if point_items else 0.0, 0.0 if edge_items else 0.0, (), (), 0.0, False,
            "no_valid_dynamic_evidence",
        )
    values = np.asarray([item.value for _, item in combined], dtype=float)
    qualities = [item.quality for _, item in combined]
    quality = float(np.mean(qualities))
    top_count = min(topk, len(values))
    top_indices = np.argsort(values)[-top_count:]
    median = ResidualEvidence.observed("r_object_dynamic_median", float(np.median(values)), quality=quality, source_ids=(object_track_id,))
    trimmed = ResidualEvidence.observed("r_object_dynamic_trimmed_mean", _trimmed_mean(values, trim_fraction), quality=quality, source_ids=(object_track_id,))
    top = ResidualEvidence.observed("r_object_dynamic_topk_mean", float(np.mean(values[top_indices])), quality=quality, source_ids=(object_track_id,))
    sorted_points = tuple(identity for identity, _ in sorted(valid_points, key=lambda item: item[1].value, reverse=True)[:topk])
    sorted_edges = tuple(identity for identity, _ in sorted(valid_edges, key=lambda item: item[1].value, reverse=True)[:topk])
    return ObjectDynamicAggregate(
        object_track_id=object_track_id,
        point_residuals=tuple(item[1] for item in point_items),
        edge_residuals=tuple(item[1] for item in edge_items),
        median=median,
        trimmed_mean=trimmed,
        topk_mean=top,
        valid_point_ratio=len(valid_points) / len(point_items) if point_items else 0.0,
        valid_edge_ratio=len(valid_edges) / len(edge_items) if edge_items else 0.0,
        top_anomalous_points=sorted_points,
        top_anomalous_edges=sorted_edges,
        quality=quality,
        valid=True,
        metadata={"not_video_classification_score": True, "trim_fraction": trim_fraction, "topk": topk},
    )
