"""Robust applicability- and quality-aware P4 evidence aggregation."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from ..validity import ResidualEvidence
from .applicability import AggregationEvidence, EvidenceApplicability, from_residual_evidence
from .contracts import (
    ClipEvidenceAggregate,
    EdgeEvidenceAggregate,
    FrameEvidenceAggregate,
    ObjectEvidenceAggregate,
    PointEvidenceAggregate,
    _EvidenceAggregate,
)


AGGREGATE_TYPES = {
    "point": PointEvidenceAggregate,
    "edge": EdgeEvidenceAggregate,
    "object": ObjectEvidenceAggregate,
    "frame": FrameEvidenceAggregate,
    "clip": ClipEvidenceAggregate,
}


def _as_item(item: AggregationEvidence | ResidualEvidence, index: int) -> AggregationEvidence:
    if isinstance(item, AggregationEvidence):
        return item
    if isinstance(item, ResidualEvidence):
        return from_residual_evidence(item, source_id=(item.source_ids[0] if item.source_ids else f"legacy:{index}"))
    raise TypeError(f"Unsupported evidence type: {type(item).__name__}")


def _aggregate_values(
    values: np.ndarray,
    qualities: np.ndarray,
    *,
    method: str,
    top_k: int,
    trim_fraction: float,
) -> float:
    if top_k < 1:
        raise ValueError("top_k must be positive.")
    if not 0.0 <= trim_fraction < 0.5:
        raise ValueError("trim_fraction must be in [0, 0.5).")
    normalized_method = {"topk_mean": "legacy_topk_mean"}.get(method, method)
    count = min(top_k, len(values))
    weights = np.maximum(qualities, 1e-12)
    if normalized_method == "median":
        return float(np.median(values))
    if normalized_method == "trimmed_mean":
        ordered = np.sort(values)
        trim = int(math.floor(len(values) * trim_fraction))
        selected = ordered[trim:len(ordered) - trim] if trim else ordered
        return float(np.mean(selected))
    if normalized_method == "top_k_mean":
        return float(np.mean(values[np.argsort(values)[-count:]]))
    if normalized_method == "legacy_topk_mean":
        selected = np.argsort(values)[-count:]
        return float(np.average(values[selected], weights=weights[selected]))
    if normalized_method == "quality_weighted_mean":
        return float(np.average(values, weights=weights))
    if normalized_method == "quality_weighted_top_k":
        selected = np.argsort(values * qualities)[-count:]
        return float(np.average(values[selected], weights=weights[selected]))
    if normalized_method == "hybrid_median_top_k":
        selected = np.argsort(values * qualities)[-count:]
        top_value = float(np.average(values[selected], weights=weights[selected]))
        return 0.5 * float(np.median(values)) + 0.5 * top_value
    raise ValueError(
        "method must be median, trimmed_mean, top_k_mean, topk_mean, "
        "quality_weighted_mean, quality_weighted_top_k, or hybrid_median_top_k."
    )


def _invalid_reason(items: Sequence[AggregationEvidence]) -> str:
    statuses = {item.applicability for item in items}
    if statuses and statuses == {EvidenceApplicability.NOT_APPLICABLE}:
        return EvidenceApplicability.NOT_APPLICABLE.value
    for status in (
        EvidenceApplicability.OBSERVATION_MISSING,
        EvidenceApplicability.INVALID_GEOMETRY,
        EvidenceApplicability.UNSUPPORTED_MODE,
    ):
        if status in statuses:
            return status.value
    return "no_valid_evidence"


def aggregate_evidence_v2(
    evidences: Sequence[AggregationEvidence | ResidualEvidence],
    *,
    level: str,
    method: str = "topk_mean",
    top_k: int = 3,
    trim_fraction: float = 0.1,
    quality_floor: float = 0.0,
    identity: Mapping[str, Any] | None = None,
    coverage_dimensions: Mapping[str, float] | None = None,
) -> _EvidenceAggregate:
    """Aggregate only valid applicable evidence and retain full provenance."""

    if level not in AGGREGATE_TYPES:
        raise ValueError(f"Unsupported aggregate level: {level!r}.")
    if not 0.0 <= quality_floor <= 1.0:
        raise ValueError("quality_floor must be in [0, 1].")
    aggregate_type = AGGREGATE_TYPES[level]
    items = tuple(_as_item(item, index) for index, item in enumerate(evidences))
    applicable = tuple(item for item in items if item.applicability == EvidenceApplicability.APPLICABLE)
    accepted = tuple(item for item in applicable if item.valid and item.quality >= quality_floor and math.isfinite(item.value))
    missing = sum(item.applicability == EvidenceApplicability.OBSERVATION_MISSING for item in items)
    not_applicable = sum(item.applicability == EvidenceApplicability.NOT_APPLICABLE for item in items)
    invalid_geometry = sum(item.applicability == EvidenceApplicability.INVALID_GEOMETRY for item in items)
    unsupported = sum(item.applicability == EvidenceApplicability.UNSUPPORTED_MODE for item in items)
    denominator = len(applicable) + missing + invalid_geometry + unsupported
    coverage = len(accepted) / denominator if denominator else 1.0 if not_applicable else 0.0
    quality = float(np.mean([item.quality for item in accepted])) if accepted else 1.0 if items and not_applicable == len(items) else 0.0
    source_ids = tuple(item.source_id for item in accepted)
    branches = tuple(item.branch_name for item in accepted)
    ranked = sorted(accepted, key=lambda item: item.value * item.quality, reverse=True)
    top_contributors = tuple({
        "source_id": item.source_id,
        "branch_name": item.branch_name,
        "value": item.value,
        "quality": item.quality,
        "frame_index": item.frame_index,
        "object_track_id": item.object_track_id,
        "point_or_edge_id": item.point_or_edge_id,
    } for item in ranked[:max(1, top_k)])
    dimensions = {
        "branch_coverage": coverage,
        "object_coverage": coverage,
        "temporal_coverage": coverage,
        "spatial_coverage": coverage,
        **dict(coverage_dimensions or {}),
    }
    common: dict[str, Any] = {
        "quality": quality,
        "coverage": coverage,
        "applicable_count": len(applicable),
        "valid_count": len(accepted),
        "observation_missing_count": missing,
        "not_applicable_count": not_applicable,
        "invalid_geometry_count": invalid_geometry,
        "unsupported_mode_count": unsupported,
        "contributing_source_ids": source_ids,
        "contributing_branch_names": branches,
        "top_contributors": top_contributors,
        "coverage_dimensions": dimensions,
        "metadata": {
            "aggregation_method": method,
            "top_k": top_k,
            "trim_fraction": trim_fraction,
            "quality_floor": quality_floor,
            "input_count": len(items),
            "valid_input_count": len(accepted),
            "quality_rejected_count": len(applicable) - len(accepted),
            "classification_output": False,
        },
        **dict(identity or {}),
    }
    if not accepted:
        return aggregate_type(value=float("nan"), valid=False, missing_reason=_invalid_reason(items), **common)
    values = np.asarray([item.value for item in accepted], dtype=float)
    qualities = np.asarray([item.quality for item in accepted], dtype=float)
    return aggregate_type(
        value=_aggregate_values(values, qualities, method=method, top_k=top_k, trim_fraction=trim_fraction),
        valid=True, missing_reason="", **common,
    )
