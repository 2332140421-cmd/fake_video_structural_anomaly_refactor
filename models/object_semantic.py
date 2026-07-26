"""Unary metric object semantics and same-track metric size stability."""

from __future__ import annotations

import csv
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
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
    entry_id: str
    category: str
    dimension: str
    min_meters: float
    max_meters: float
    dimension_definition: str
    applicable_scope: str
    excluded_scope: str
    confidence: str
    minimum_observability: float
    derivation_id: str


@dataclass(frozen=True)
class ScalePriorRegistry:
    schema_version: str
    prior_sha256: str
    source_table_sha256: str
    priors_by_label: Mapping[str, MetricPrior]
    unsupported_by_label: Mapping[str, str]

    def resolve(self, category: str) -> MetricPrior | None:
        return self.priors_by_label.get(category)

    def unsupported_reason(self, category: str) -> str:
        return self.unsupported_by_label.get(category, "")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _exact_labels(row: Mapping[str, Any]) -> tuple[str, ...]:
    class_name = str(row.get("class_name", ""))
    aliases = row.get("aliases", ())
    if not class_name or class_name != class_name.strip():
        raise ValueError("Scale-prior class_name must be non-empty exact text.")
    if not isinstance(aliases, list):
        raise ValueError(f"aliases for {class_name!r} must be a YAML list.")
    labels = (class_name, *(str(alias) for alias in aliases))
    if any(not label or label != label.strip() for label in labels):
        raise ValueError(f"Aliases for {class_name!r} must be non-empty exact text.")
    if len(labels) != len(set(labels)):
        raise ValueError(f"Duplicate exact class/alias mapping for {class_name!r}.")
    return labels


def load_scale_prior_registry(path: str | Path) -> ScalePriorRegistry:
    prior_path = Path(path)
    payload = yaml.safe_load(prior_path.read_text(encoding="utf-8")) or {}
    if payload.get("schema_version") != "paper_core_scale_priors_v1":
        raise ValueError("Unsupported scale-prior schema; formal v1 is required.")
    if payload.get("unit") != "meter":
        raise ValueError("Scale-prior unit must be exactly 'meter'.")
    source_name = payload.get("source_table")
    if not isinstance(source_name, str) or not source_name:
        raise ValueError("Formal scale priors require a source_table.")
    source_path = prior_path.parent / source_name
    with source_path.open(newline="", encoding="utf-8") as handle:
        source_reader = csv.DictReader(handle)
        source_fields = set(source_reader.fieldnames or ())
        source_rows = list(source_reader)
    required_source_fields = {
        "derivation_id",
        "source_type",
        "source_title",
        "publisher",
        "source_identifier",
        "source_version",
        "accessed_at",
        "sample_count",
        "raw_measurements_or_range",
        "derivation_method",
        "review_status",
    }
    if not required_source_fields <= source_fields:
        raise ValueError("Scale-prior source table is missing required columns.")
    sources: dict[str, Mapping[str, str]] = {}
    for row in source_rows:
        derivation_id = str(row["derivation_id"])
        if not derivation_id or derivation_id in sources:
            raise ValueError("Source derivation_id values must be non-empty and unique.")
        if row["review_status"] != "APPROVED_SOURCE_BACKED":
            raise ValueError(f"Derivation {derivation_id!r} is not source-approved.")
        if int(row["sample_count"]) <= 0:
            raise ValueError(f"Derivation {derivation_id!r} has no source samples.")
        if any(not str(row[field]).strip() for field in required_source_fields):
            raise ValueError(f"Derivation {derivation_id!r} has incomplete provenance.")
        sources[derivation_id] = row

    priors: dict[str, MetricPrior] = {}
    unsupported: dict[str, str] = {}
    entry_ids: set[str] = set()
    for row in payload.get("priors", ()):
        labels = _exact_labels(row)
        entry_id = str(row.get("entry_id", ""))
        derivation_id = str(row.get("derivation_id", ""))
        dimension = str(row.get("supported_dimension", ""))
        if not entry_id or entry_id in entry_ids:
            raise ValueError("Scale-prior entry_id values must be non-empty and unique.")
        if derivation_id not in sources:
            raise ValueError(f"Scale prior {entry_id!r} has no approved source derivation.")
        if dimension not in {"height", "width", "length"}:
            raise ValueError(f"Scale prior {entry_id!r} uses an unsupported dimension.")
        minimum = float(row["min_m"])
        maximum = float(row["max_m"])
        if not (math.isfinite(minimum) and math.isfinite(maximum) and 0 < minimum < maximum):
            raise ValueError(f"Scale prior {entry_id!r} must satisfy 0 < min_m < max_m.")
        confidence = str(row.get("confidence", ""))
        if confidence not in {"low", "medium", "high"}:
            raise ValueError(f"Scale prior {entry_id!r} has invalid confidence.")
        prior = MetricPrior(
            entry_id=entry_id,
            category=labels[0],
            dimension=dimension,
            min_meters=minimum,
            max_meters=maximum,
            dimension_definition=str(row.get("dimension_definition", "")),
            applicable_scope=str(row.get("applicable_scope", "")),
            excluded_scope=str(row.get("excluded_scope", "")),
            confidence=confidence,
            minimum_observability=float(row.get("minimum_observability", 0.0)),
            derivation_id=derivation_id,
        )
        if not all(
            (
                prior.dimension_definition,
                prior.applicable_scope,
                prior.excluded_scope,
            )
        ):
            raise ValueError(f"Scale prior {entry_id!r} has incomplete scope definition.")
        for label in labels:
            if label in priors or label in unsupported:
                raise ValueError(f"Ambiguous exact scale-prior mapping for {label!r}.")
            priors[label] = prior
        entry_ids.add(entry_id)

    for row in payload.get("unsupported_classes", ()):
        labels = _exact_labels(row)
        reason = str(row.get("reason", ""))
        if reason != "CATEGORY_TOO_BROAD_WITHOUT_SUBTYPE":
            raise ValueError("Unsupported-class policy must use an explicit approved reason.")
        for label in labels:
            if label in priors or label in unsupported:
                raise ValueError(f"Ambiguous exact scale-prior mapping for {label!r}.")
            unsupported[label] = reason
    return ScalePriorRegistry(
        schema_version=str(payload["schema_version"]),
        prior_sha256=_sha256(prior_path),
        source_table_sha256=_sha256(source_path),
        priors_by_label=MappingProxyType(priors),
        unsupported_by_label=MappingProxyType(unsupported),
    )


def load_metric_priors(path: str | Path) -> dict[str, MetricPrior]:
    """Compatibility view containing only exact and explicit-alias mappings."""

    return dict(load_scale_prior_registry(path).priors_by_label)


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


def _mask_completeness_support(
    mask: Any,
    bbox_xyxy: Any,
    frame_shape: tuple[int, ...],
) -> dict[str, Any]:
    """Measure visible-mask fill inside a clipped, discrete bbox support."""

    height, width = (int(frame_shape[0]), int(frame_shape[1]))
    array = np.asarray(mask)
    unavailable = {
        "valid": False,
        "mask_area_total": float("nan"),
        "mask_area_inside_bbox": float("nan"),
        "bbox_area_raw": float("nan"),
        "bbox_area_clipped": float("nan"),
        "mask_spill_area": float("nan"),
        "mask_spill_ratio": float("nan"),
        "legacy_total_mask_over_raw_bbox_ratio": float("nan"),
        "mask_completeness": float("nan"),
        "mask_shape": list(array.shape),
        "frame_shape": [height, width],
        "mask_completeness_definition": (
            "area(bool_mask_intersection_clipped_bbox)"
            "/area(clipped_bbox_discrete_half_open)"
        ),
    }
    if array.ndim != 2 or array.shape != (height, width):
        return {**unavailable, "reason": "mask_shape_mismatch"}
    try:
        x1, y1, x2, y2 = (float(value) for value in bbox_xyxy)
    except (TypeError, ValueError):
        return {**unavailable, "reason": "invalid_bbox"}
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
        return {**unavailable, "reason": "invalid_bbox"}

    # xyxy is discretized as a half-open pixel slice.  floor(left/top) and
    # ceil(right/bottom) include every pixel touched by the continuous bbox;
    # the resulting integer support is then clipped to the image.
    left = min(width, max(0, math.floor(x1)))
    top = min(height, max(0, math.floor(y1)))
    right = min(width, max(0, math.ceil(x2)))
    bottom = min(height, max(0, math.ceil(y2)))
    clipped_bbox = [left, top, right, bottom]
    bbox_area = (right - left) * (bottom - top)
    if x2 <= x1 or y2 <= y1 or bbox_area <= 0:
        return {
            **unavailable,
            "reason": "invalid_clipped_bbox",
            "bbox_clipped_xyxy": clipped_bbox,
        }

    boolean_mask = array.astype(bool, copy=False)
    total_area = int(np.count_nonzero(boolean_mask))
    inside_area = int(np.count_nonzero(boolean_mask[top:bottom, left:right]))
    spill_area = total_area - inside_area
    raw_bbox_area = (x2 - x1) * (y2 - y1)
    completeness = inside_area / bbox_area
    if not math.isfinite(completeness):
        return {
            **unavailable,
            "reason": "non_finite_mask_completeness",
            "bbox_clipped_xyxy": clipped_bbox,
        }
    if not -1e-12 <= completeness <= 1.0 + 1e-12:
        raise ValueError("mask completeness support calculation escaped [0, 1].")
    completeness = min(1.0, max(0.0, completeness))
    return {
        "valid": True,
        "reason": "",
        "mask_area_total": total_area,
        "mask_area_inside_bbox": inside_area,
        "bbox_area_raw": raw_bbox_area,
        "bbox_area_clipped": bbox_area,
        "mask_spill_area": spill_area,
        "mask_spill_ratio": spill_area / total_area if total_area else 0.0,
        "legacy_total_mask_over_raw_bbox_ratio": (
            total_area / max(raw_bbox_area, 1.0)
        ),
        "mask_completeness": completeness,
        "bbox_clipped_xyxy": clipped_bbox,
        "mask_shape": list(boolean_mask.shape),
        "frame_shape": [height, width],
        "mask_completeness_definition": (
            "area(bool_mask_intersection_clipped_bbox)"
            "/area(clipped_bbox_discrete_half_open)"
        ),
    }


def _dimension_observability(
    frame: Any,
    obj: Any,
    cloud: Any,
    mask_support: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    container_axis, axis_diagnostics = _container_axis_evidence(cloud)
    viewpoint = _viewpoint(obj.viewpoint)
    explicit_pose = str(obj.metadata.get("pose_estimate_status", "unavailable"))
    if explicit_pose not in {item.value for item in PoseEstimateStatus}:
        explicit_pose = PoseEstimateStatus.UNAVAILABLE.value
    view = evaluate_object_view(
        ObjectViewInput(
            object_id=obj.object_id,
            track_id=obj.track_id,
            class_name=obj.category,
            bbox=obj.bbox_xyxy,
            image_width=frame.image.shape[1],
            image_height=frame.image.shape[0],
            detection_confidence=obj.confidence,
            mask_area=float(mask_support["mask_area_inside_bbox"]),
            bbox_area=float(mask_support["bbox_area_clipped"]),
            occlusion_ratio=obj.occlusion_ratio,
            viewpoint_hint=viewpoint,
            pose_estimate_status=explicit_pose,
            view_confidence=(
                float(obj.metadata.get("view_confidence", 1.0))
                if viewpoint != ViewpointClass.UNKNOWN
                else float("nan")
            ),
            metadata={
                "mask_completeness": float(mask_support["mask_completeness"]),
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
            "mask_support": dict(mask_support),
        }
    principal_observable = bool(
        container_axis
        and view.valid
        and math.isfinite(view.border_contact_ratio)
        and view.border_contact_ratio == 0.0
    )
    records["principal_extent"] = {
        "dimension": "principal_extent",
        "observable": principal_observable,
        "reason": (
            ""
            if principal_observable
            else (
                "border_contact"
                if math.isfinite(view.border_contact_ratio)
                and view.border_contact_ratio > 0.0
                else str(axis_diagnostics.get("reason", "dimension_axis_unavailable"))
            )
        ),
        "axis_source": (
            "robust_metric_surface_principal_axis_xy"
            if principal_observable
            else ""
        ),
        "estimated_size_m": (
            float(axis_diagnostics["estimated_axis_extent_m"])
            if principal_observable
            else float("nan")
        ),
        "prior_min_m": float("nan"),
        "prior_max_m": float("nan"),
        "residual": float("nan"),
        "confidence": 0.0,
        "viewpoint_evidence": False,
        "viewpoint_class": view.viewpoint_class.value,
        "pose_estimate_status": view.pose_estimate_status.value,
        "visible_surface_only": True,
        "axis_diagnostics": axis_diagnostics,
        "mask_support": dict(mask_support),
        "category_prior_required": False,
        "physical_dimension_definition": (
            "dominant visible metric surface extent; not asserted to be canonical "
            "height, width, or length"
        ),
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
    registry = load_scale_prior_registry(prior_path)
    output: list[ResidualEvidence] = []
    history: dict[tuple[str, str], list[tuple[int, float, float, str]]] = {}
    for frame in clip.frames:
        for obj in frame.objects:
            mask_support = (
                {
                    "valid": False,
                    "reason": "instance_mask_unavailable",
                    "mask_completeness": float("nan"),
                }
                if obj.instance_mask is None
                else _mask_completeness_support(
                    obj.instance_mask,
                    obj.bbox_xyxy,
                    frame.image.shape,
                )
            )
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
                "mask_support": dict(mask_support),
                "class_id": obj.metadata.get("class_id"),
                "scale_prior_schema_version": registry.schema_version,
                "scale_prior_sha256": registry.prior_sha256,
                "scale_prior_source_table_sha256": registry.source_table_sha256,
                "scale_prior_entry_id": "",
                "scale_prior_confidence": "",
                "legacy_container_upright_fallback_used": False,
            }
            prior = registry.resolve(obj.category)
            if prior is not None:
                base_meta.update(
                    {
                        "scale_prior_entry_id": prior.entry_id,
                        "scale_prior_confidence": prior.confidence,
                        "scale_prior_derivation_id": prior.derivation_id,
                    }
                )
            unsupported = registry.unsupported_reason(obj.category)

            common_reason = ""
            if obj.instance_mask is None or not np.any(obj.instance_mask):
                common_reason = "instance_mask_unavailable"
            elif not bool(mask_support["valid"]):
                common_reason = str(mask_support["reason"])
            elif obj.occlusion_ratio > max_occlusion_ratio:
                common_reason = "severe_object_occlusion"
            elif obj.mask_quality < min_mask_quality:
                common_reason = "insufficient_mask_quality"
            elif not obj.track_identity_stable:
                common_reason = "unstable_track_identity"
            cloud = None if common_reason else build_metric_object_surface(frame, obj)
            if not common_reason and (cloud is None or not cloud.valid):
                common_reason = "metric_object_surface_unavailable"
            if (
                not common_reason
                and cloud is not None
                and cloud.valid_point_ratio
                < min_depth_coverage
            ):
                common_reason = "insufficient_valid_metric_depth_ratio"
            dimensions = (
                {}
                if cloud is None or not cloud.valid
                else _dimension_observability(frame, obj, cloud, mask_support)
            )
            base_meta.update(
                {
                    "metric_object_surface_valid": bool(cloud is not None and cloud.valid),
                    "valid_metric_depth_ratio": (
                        float(cloud.valid_point_ratio)
                        if cloud is not None and cloud.valid
                        else float("nan")
                    ),
                }
            )

            prior_reason = common_reason
            if prior is None:
                prior_reason = (
                    "category_too_broad_without_subtype"
                    if unsupported
                    else "missing_category_metric_prior"
                )
            elif (
                not prior_reason
                and cloud is not None
                and cloud.valid_point_ratio
                < max(min_depth_coverage, prior.minimum_observability)
            ):
                prior_reason = "insufficient_valid_metric_depth_ratio"
            if prior is not None and prior.dimension in dimensions:
                selected = dimensions[prior.dimension]
                selected["prior_min_m"] = prior.min_meters
                selected["prior_max_m"] = prior.max_meters
                if not prior_reason and not selected["observable"]:
                    prior_reason = _unobservable_reason(selected)
                base_meta.update(
                    {
                        "dimension": prior.dimension,
                        "axis_source": selected["axis_source"],
                        "observability_reason": selected["reason"],
                    }
                )
            base_meta["dimension_observability"] = list(dimensions.values())
            if prior_reason:
                output.append(
                    ResidualEvidence.unavailable(
                        "semantic_metric_prior",
                        "object",
                        prior_reason,
                        spatial_support=support,
                        temporal_support={"frame_index": frame.frame_index},
                        metadata=base_meta,
                    )
                )
            else:
                assert prior is not None and cloud is not None
                selected = dimensions[prior.dimension]
                estimate = float(selected["estimated_size_m"])
                residual = _log_interval_distance(
                    estimate,
                    prior.min_meters,
                    prior.max_meters,
                )
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
                    "observability_reason": (
                        "dimension_axis_and_visible_metric_surface_supported"
                    ),
                    "axis_source": selected["axis_source"],
                    "estimated_size_m": estimate,
                    "prior_min_m": prior.min_meters,
                    "prior_max_m": prior.max_meters,
                    "dimension_definition": prior.dimension_definition,
                    "applicable_scope": prior.applicable_scope,
                    "excluded_scope": prior.excluded_scope,
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

            temporal_reason = common_reason
            temporal_dimension = ""
            temporal_record: Mapping[str, Any] | None = None
            if not temporal_reason:
                temporal_record = next(
                    (
                        dimensions[name]
                        for name in ("height", "width", "length", "principal_extent")
                        if bool(dimensions[name]["observable"])
                    ),
                    None,
                )
                if temporal_record is None:
                    temporal_reason = (
                        _unobservable_reason(dimensions["height"])
                        if dimensions
                        else "dimension_not_observable"
                    )
                else:
                    temporal_dimension = str(temporal_record["dimension"])
            temporal_meta = {
                **base_meta,
                "dimension": temporal_dimension,
                "axis_source": (
                    str(temporal_record["axis_source"])
                    if temporal_record is not None
                    else ""
                ),
                "category_prior_required": False,
                "same_clip_only": True,
            }
            if temporal_reason:
                output.append(
                    ResidualEvidence.unavailable(
                        "semantic_metric_temporal",
                        "track",
                        temporal_reason,
                        spatial_support=support,
                        temporal_support={"frame_index": frame.frame_index},
                        metadata=temporal_meta,
                    )
                )
                continue

            assert cloud is not None and temporal_record is not None
            estimate = float(temporal_record["estimated_size_m"])
            axis_source = str(temporal_record["axis_source"])
            confidence = min(
                obj.confidence,
                obj.mask_quality,
                cloud.depth_quality,
                cloud.valid_point_ratio,
            )
            key = (obj.track_id, temporal_dimension)
            earlier = history.setdefault(key, [])
            comparable = [item for item in earlier if item[3] == axis_source]
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
                            **temporal_meta,
                            "reference_size_m": reference,
                            "estimated_size_m": estimate,
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
                        metadata=temporal_meta,
                    )
                )
            earlier.append(
                (frame.frame_index, estimate, confidence, axis_source)
            )
    return output
