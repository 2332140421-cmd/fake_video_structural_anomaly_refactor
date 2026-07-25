"""Unary metric object semantics and same-track metric size stability."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from data.schemas import ClipObservation, ResidualEvidence
from semantic3d.method_completion.view_observability import (
    ObjectViewInput,
    PoseEstimateStatus,
    ViewpointClass,
    evaluate_object_view,
)
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


def _axis_extent(
    cloud: Any,
    dimension: str,
    viewpoint: str,
    axis_diagnostics: Mapping[str, Any],
) -> float:
    if dimension == "height":
        return float(axis_diagnostics["estimated_axis_extent_m"])
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


def _container_axis_evidence(cloud: Any) -> tuple[bool, dict[str, Any]]:
    """Find a stable container axis in the camera-frame XY visible surface."""

    covariance = np.asarray(cloud.robust_covariance, dtype=float)[:2, :2]
    if covariance.shape != (2, 2) or not np.isfinite(covariance).all():
        return False, {"axis_source": "unavailable", "reason": "invalid_robust_covariance"}
    values, vectors = np.linalg.eigh(covariance)
    major = vectors[:, int(np.argmax(values))]
    ordered = np.sort(np.maximum(values, 0.0))
    dominance = float(ordered[-1] / max(ordered[-2], 1e-12))
    points_xy = np.asarray(
        [[point.x_m, point.y_m] for point in cloud.points if point.valid],
        dtype=float,
    )
    if points_xy.ndim != 2 or points_xy.shape[0] < 3:
        return False, {"axis_source": "unavailable", "reason": "insufficient_axis_points"}
    projected = points_xy @ major
    low, high = np.quantile(
        projected,
        [float(cloud.quantile_low), float(cloud.quantile_high)],
    )
    extent = float(high - low)
    reliable = dominance >= 1.25 and math.isfinite(extent) and extent > 0.0
    return reliable, {
        "axis_source": "robust_metric_surface_principal_axis_xy",
        "axis_xy": [float(major[0]), float(major[1])],
        "camera_y_alignment": float(abs(major[1])),
        "axis_eigenvalue_ratio": dominance,
        "estimated_axis_extent_m": extent,
        "extent_quantiles": [float(cloud.quantile_low), float(cloud.quantile_high)],
        "threshold_role": "fixed_geometry_gate_not_fitted_on_video_labels",
    }


def _viewpoint(value: str) -> ViewpointClass:
    return {
        "front": ViewpointClass.FRONTAL,
        "frontal": ViewpointClass.FRONTAL,
        "side": ViewpointClass.LATERAL,
        "lateral": ViewpointClass.LATERAL,
        "oblique": ViewpointClass.OBLIQUE,
        "top_down": ViewpointClass.TOP_DOWN,
        "bottom_up": ViewpointClass.BOTTOM_UP,
    }.get(value, ViewpointClass.UNKNOWN)


def _dimension_observability(frame: Any, obj: Any, cloud: Any) -> dict[str, dict[str, Any]]:
    container_axis, axis_diagnostics = _container_axis_evidence(cloud)
    viewpoint = _viewpoint(obj.viewpoint)
    explicit_pose = str(obj.metadata.get("pose_estimate_status", "unavailable"))
    if explicit_pose not in {item.value for item in PoseEstimateStatus}:
        explicit_pose = PoseEstimateStatus.UNAVAILABLE.value
    if (
        explicit_pose == PoseEstimateStatus.UNAVAILABLE.value
        and container_axis
        and obj.category in {"cup", "bottle", "vase"}
    ):
        explicit_pose = PoseEstimateStatus.UPRIGHT_SHAPE_COMPATIBLE.value
    x1, y1, x2, y2 = obj.bbox_xyxy
    view = evaluate_object_view(
        ObjectViewInput(
            object_id=obj.object_id,
            track_id=obj.track_id,
            class_name=obj.category,
            bbox=obj.bbox_xyxy,
            image_width=frame.image.shape[1],
            image_height=frame.image.shape[0],
            detection_confidence=obj.confidence,
            mask_area=float(np.count_nonzero(obj.instance_mask)),
            bbox_area=max((x2 - x1) * (y2 - y1), 1.0),
            occlusion_ratio=obj.occlusion_ratio,
            viewpoint_hint=viewpoint,
            pose_estimate_status=explicit_pose,
            view_confidence=(
                float(obj.metadata.get("view_confidence", 1.0))
                if viewpoint != ViewpointClass.UNKNOWN
                else float("nan")
            ),
            metadata={
                "mask_completeness": float(np.count_nonzero(obj.instance_mask))
                / max((x2 - x1) * (y2 - y1), 1.0),
                "mask_is_visible_not_amodal": True,
                "depth_extent_supported": False,
            },
        )
    )
    height_observable = bool(view.height_observable and container_axis)
    flags = {
        "height": height_observable,
        "width": bool(view.width_observable),
        "length": bool(view.length_observable),
    }
    axis_sources = {
        "height": (
            "robust_metric_surface_principal_axis_xy"
            if container_axis
            else ""
        ),
        "width": (
            "camera_x_visible_extent_with_frontal_view"
            if view.width_observable
            else ""
        ),
        "length": (
            "camera_x_visible_extent_with_lateral_view"
            if view.length_observable
            else ""
        ),
    }
    reason_keys = {"height": "height_m", "width": "width_m", "length": "length_m"}
    records: dict[str, dict[str, Any]] = {}
    for dimension in ("height", "width", "length"):
        reasons = list(view.dimension_reasons.get(reason_keys[dimension], ()))
        if dimension == "height" and not container_axis:
            reasons.append("no_reliable_dimension_aligned_metric_axis")
        records[dimension] = {
            "dimension": dimension,
            "observable": flags[dimension],
            "reason": "" if flags[dimension] else "|".join(dict.fromkeys(reasons)),
            "axis_source": axis_sources[dimension],
            "estimated_size_m": (
                _axis_extent(cloud, dimension, obj.viewpoint, axis_diagnostics)
                if flags[dimension]
                else float("nan")
            ),
            "prior_min_m": float("nan"),
            "prior_max_m": float("nan"),
            "residual": float("nan"),
            "confidence": 0.0,
            "viewpoint_evidence": (
                view.viewpoint_class != ViewpointClass.UNKNOWN
                or view.pose_estimate_status != PoseEstimateStatus.UNAVAILABLE
            ),
            "viewpoint_class": view.viewpoint_class.value,
            "pose_estimate_status": view.pose_estimate_status.value,
            "visible_surface_only": True,
            "axis_diagnostics": axis_diagnostics if dimension == "height" else {},
        }
    return records


def _unobservable_reason(record: Mapping[str, Any]) -> str:
    reason = str(record.get("reason", ""))
    if "border_contact" in reason:
        return "dimension_truncated"
    if "no_reliable" in reason or "axis_not_verified" in reason:
        return "no_reliable_dimension_axis"
    if "viewpoint" in reason or "foreshortening" in reason:
        return "viewpoint_unavailable"
    return "dimension_not_observable"


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
    history: dict[tuple[str, str], list[tuple[int, float, float, str]]] = {}
    for frame in clip.frames:
        for obj in frame.objects:
            support = {
                "kind": "object_mask",
                "mask": obj.instance_mask,
                "clip_id": clip.clip_id,
                "frame_index": frame.frame_index,
                "object_id": obj.object_id,
                "track_id": obj.track_id,
            }
            base_meta: dict[str, Any] = {
                "category": obj.category,
                "coordinate_frame": "camera_frame_metric",
                "dimension_observability": [],
                "old_object_pair_rsd_used": False,
                "authenticity_label_used": False,
            }
            reason = ""
            prior = priors.get(obj.category)
            if prior is None:
                reason = "missing_category_metric_prior"
            elif obj.instance_mask is None or not np.any(obj.instance_mask):
                reason = "instance_mask_unavailable"
            elif obj.occlusion_ratio > max_occlusion_ratio:
                reason = "severe_object_occlusion"
            elif obj.mask_quality < min_mask_quality:
                reason = "insufficient_mask_quality"
            elif not obj.track_identity_stable:
                reason = "unstable_track_identity"
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
            dimensions = (
                {} if cloud is None or not cloud.valid else _dimension_observability(frame, obj, cloud)
            )
            if prior is not None and prior.dimension in dimensions:
                selected = dimensions[prior.dimension]
                selected["prior_min_m"] = prior.min_meters
                selected["prior_max_m"] = prior.max_meters
                if not reason and not selected["observable"]:
                    reason = _unobservable_reason(selected)
                base_meta.update(
                    {
                        "dimension": prior.dimension,
                        "axis_source": selected["axis_source"],
                        "observability_reason": selected["reason"],
                    }
                )
            base_meta["dimension_observability"] = list(dimensions.values())
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
            selected = dimensions[prior.dimension]
            estimate = float(selected["estimated_size_m"])
            residual = _log_interval_distance(estimate, prior.min_meters, prior.max_meters)
            confidence = min(
                obj.confidence,
                obj.mask_quality,
                cloud.depth_quality,
                cloud.valid_point_ratio,
            )
            selected.update(
                {
                    "prior_min_m": prior.min_meters,
                    "prior_max_m": prior.max_meters,
                    "residual": residual,
                    "confidence": confidence,
                }
            )
            metadata = {
                **base_meta,
                "dimension": prior.dimension,
                "observable": True,
                "observability_reason": "dimension_axis_and_visible_metric_surface_supported",
                "axis_source": selected["axis_source"],
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
            comparable = [item for item in earlier if item[3] == selected["axis_source"]]
            if comparable:
                reference = float(np.median([value for _, value, _, _ in comparable[-5:]]))
                temporal = abs(math.log(estimate) - math.log(reference))
                output.append(
                    ResidualEvidence.observed(
                        "semantic_metric_temporal",
                        "track",
                        temporal,
                        confidence=min(
                            confidence,
                            float(np.median([q for _, _, q, _ in comparable[-5:]])),
                        ),
                        spatial_support=support,
                        temporal_support={
                            "frame_index": frame.frame_index,
                            "history_frames": [index for index, _, _, _ in comparable[-5:]],
                        },
                        metadata={
                            "dimension": prior.dimension,
                            "axis_source": selected["axis_source"],
                            "reference_size_m": reference,
                            "same_clip_only": True,
                        },
                    )
                )
            else:
                missing_reason = (
                    "incompatible_dimension_axis_history"
                    if earlier
                    else "insufficient_same_track_metric_history"
                )
                output.append(
                    ResidualEvidence.unavailable(
                        "semantic_metric_temporal",
                        "track",
                        missing_reason,
                        spatial_support=support,
                        temporal_support={"frame_index": frame.frame_index},
                    )
                )
            earlier.append(
                (frame.frame_index, estimate, confidence, str(selected["axis_source"]))
            )
    return output
