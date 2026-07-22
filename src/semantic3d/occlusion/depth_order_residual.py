"""Depth ordering evidence at an established occlusion relation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from ..validity import ResidualEvidence
from .occlusion_graph import OcclusionRelation


@dataclass(frozen=True)
class OcclusionDepthOrderResidual:
    """Expected foreground-smaller-Z residual with source-quality metadata."""

    foreground_object_id: str
    background_object_id: str
    frame_index: int
    foreground_depth: float
    background_depth: float
    depth_uncertainty: float
    depth_source: str
    evidence: ResidualEvidence
    valid: bool
    missing_reason: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


def compute_occlusion_depth_order_residual(
    relation: OcclusionRelation,
    *,
    observed_foreground_depth: Optional[float] = None,
    observed_background_depth: Optional[float] = None,
    depth_uncertainty: float = 0.05,
    depth_source: str = "object_center_depth",
) -> OcclusionDepthOrderResidual:
    """Return positive evidence only when a confident relation reverses Z order."""

    foreground = float(relation.foreground_depth if observed_foreground_depth is None else observed_foreground_depth)
    background = float(relation.background_depth if observed_background_depth is None else observed_background_depth)
    source_ids = (relation.foreground_object_id, relation.background_object_id, str(relation.frame_index))
    reason = ""
    if not relation.valid:
        reason = relation.missing_reason or "invalid_occlusion_relation"
    elif not math.isfinite(foreground) or not math.isfinite(background):
        reason = "invalid_occlusion_depth"
    elif abs(foreground - background) <= depth_uncertainty:
        reason = "depth_order_within_uncertainty"
    if reason:
        evidence = ResidualEvidence.missing("r_occlusion_depth_order", reason, source_ids=source_ids)
        return OcclusionDepthOrderResidual(relation.foreground_object_id, relation.background_object_id, relation.frame_index, foreground, background, depth_uncertainty, depth_source, evidence, False, reason)
    scale = max(abs(foreground), abs(background), depth_uncertainty)
    residual = max(0.0, foreground - background + depth_uncertainty) / scale
    quality = relation.occlusion_confidence * (0.5 if depth_source == "object_center_depth" else 1.0)
    evidence = ResidualEvidence.observed("r_occlusion_depth_order", residual, quality=quality, source_ids=source_ids, metadata={"expected_order": "foreground_Z < background_Z", "depth_source": depth_source, "center_depth_low_quality": depth_source == "object_center_depth"})
    return OcclusionDepthOrderResidual(relation.foreground_object_id, relation.background_object_id, relation.frame_index, foreground, background, depth_uncertainty, depth_source, evidence, True, metadata={"boundary_neighbourhood_preferred": True})
