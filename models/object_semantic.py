"""Unary metric object semantics and same-track metric size stability."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from data.schemas import ClipObservation, ResidualEvidence
from .geometry import build_metric_object_surface


@dataclass(frozen=True)
class MetricPrior:
    category: str
    dimension: str
    min_meters: float
    max_meters: float
    orientation_requirement: str
    minimum_observability: float
    source_note: str


def load_metric_priors(path: str | Path) -> dict[str, MetricPrior]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return {
        str(row["category"]): MetricPrior(
            category=str(row["category"]),
            dimension=str(row["dimension"]),
            min_meters=float(row["min_meters"]),
            max_meters=float(row["max_meters"]),
            orientation_requirement=str(row.get("orientation_requirement", "")),
            minimum_observability=float(row.get("minimum_observability", 0.0)),
            source_note=str(row.get("source_note", "")),
        )
        for row in payload.get("metric_scale_priors", ())
    }


def _axis_extent(cloud: Any, dimension: str, viewpoint: str) -> float:
    if dimension == "height":
        return cloud.y_extent_m
    if dimension == "width":
        return cloud.x_extent_m
    if dimension == "length":
        return cloud.x_extent_m if viewpoint in {"side", "oblique"} else cloud.z_extent_m
    if dimension == "diameter":
        return max(cloud.x_extent_m, cloud.y_extent_m)
    if dimension == "category_major_dimension":
        return max(cloud.x_extent_m, cloud.y_extent_m, cloud.z_extent_m)
    raise ValueError(f"Unsupported metric prior dimension: {dimension}.")


def _log_interval_distance(value: float, low: float, high: float) -> float:
    return max(math.log(low) - math.log(value), 0.0, math.log(value) - math.log(high))


def compute_object_semantic_residuals(
    clip: ClipObservation,
    *,
    prior_path: str | Path,
    min_depth_coverage: float = 0.5,
    max_occlusion_ratio: float = 0.5,
    min_mask_quality: float = 0.3,
) -> list[ResidualEvidence]:
    priors = load_metric_priors(prior_path)
    output: list[ResidualEvidence] = []
    history: dict[tuple[str, str], list[tuple[int, float, float]]] = {}
    for frame in clip.frames:
        for obj in frame.objects:
            support = {
                "kind": "object_mask",
                "mask": obj.instance_mask,
                "frame_index": frame.frame_index,
                "object_id": obj.object_id,
                "track_id": obj.track_id,
            }
            base_meta = {"category": obj.category, "coordinate_frame": "camera_frame_metric"}
            reason = ""
            prior = priors.get(obj.category)
            if prior is None:
                reason = "missing_category_metric_prior"
            elif obj.instance_mask is None or not np.any(obj.instance_mask):
                reason = "instance_mask_unavailable"
            elif obj.truncated:
                reason = "severe_object_truncation"
            elif obj.occlusion_ratio > max_occlusion_ratio:
                reason = "severe_object_occlusion"
            elif obj.mask_quality < min_mask_quality:
                reason = "insufficient_mask_quality"
            elif not obj.track_identity_stable:
                reason = "unstable_track_identity"
            elif (
                prior.orientation_requirement
                and "unknown" not in prior.orientation_requirement
                and obj.viewpoint not in prior.orientation_requirement.split("_or_")
            ):
                reason = "dimension_not_observable_from_current_view"
            cloud = None if reason else build_metric_object_surface(frame, obj)
            if not reason and (cloud is None or not cloud.valid):
                reason = "metric_object_surface_unavailable"
            if (
                not reason
                and prior is not None
                and cloud.valid_point_ratio
                < max(min_depth_coverage, prior.minimum_observability)
            ):
                reason = "insufficient_valid_metric_depth_ratio"
            if reason:
                output.append(
                    ResidualEvidence.unavailable(
                        "semantic_metric_prior",
                        "object",
                        reason,
                        spatial_support=support,
                        temporal_support={"frame_index": frame.frame_index},
                        metadata=base_meta,
                    )
                )
                continue
            assert prior is not None and cloud is not None
            estimate = _axis_extent(cloud, prior.dimension, obj.viewpoint)
            residual = _log_interval_distance(estimate, prior.min_meters, prior.max_meters)
            confidence = min(
                obj.confidence,
                obj.mask_quality,
                cloud.depth_quality,
                cloud.valid_point_ratio,
            )
            metadata = {
                **base_meta,
                "dimension": prior.dimension,
                "observable": True,
                "observability_reason": "visible_metric_surface_supported",
                "estimated_size_m": estimate,
                "prior_min_m": prior.min_meters,
                "prior_max_m": prior.max_meters,
                "source_note": prior.source_note,
                "visible_surface_only": True,
                "world_frame_claimed": False,
            }
            output.append(
                ResidualEvidence.observed(
                    "semantic_metric_prior",
                    "object",
                    residual,
                    confidence=confidence,
                    spatial_support=support,
                    temporal_support={"frame_index": frame.frame_index},
                    metadata=metadata,
                )
            )
            key = (obj.track_id, prior.dimension)
            earlier = history.setdefault(key, [])
            if earlier:
                reference = float(np.median([value for _, value, _ in earlier[-5:]]))
                temporal = abs(math.log(estimate) - math.log(reference))
                output.append(
                    ResidualEvidence.observed(
                        "semantic_metric_temporal",
                        "track",
                        temporal,
                        confidence=min(confidence, float(np.median([q for _, _, q in earlier[-5:]]))),
                        spatial_support=support,
                        temporal_support={
                            "frame_index": frame.frame_index,
                            "history_frames": [index for index, _, _ in earlier[-5:]],
                        },
                        metadata={"dimension": prior.dimension, "reference_size_m": reference},
                    )
                )
            else:
                output.append(
                    ResidualEvidence.unavailable(
                        "semantic_metric_temporal",
                        "track",
                        "insufficient_same_track_metric_history",
                        spatial_support=support,
                        temporal_support={"frame_index": frame.frame_index},
                    )
                )
            earlier.append((frame.frame_index, estimate, confidence))
    return output
