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


@dataclass(frozen=True)
class CanonicalAxisRegistry:
    schema_version: str
    config_sha256: str
    thresholds: Mapping[str, float]
    bottle_source_runtime_dimension_match: bool
    bottle_source_field: str
    bottle_source_dimension_definition: str
    bottle_runtime_dimension_definition: str


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


_CANONICAL_THRESHOLD_NAMES = frozenset(
    {
        "bottle_eigenvalue_ratio_min",
        "bottle_extent_ratio_min",
        "bottle_point_support_min",
        "bottle_depth_coverage_min",
        "bottle_axis_stability_min",
        "bottle_track_axis_compatibility_min",
        "bottle_track_extent_log_delta_max",
        "bottle_mask_quality_min",
        "truncation_threshold",
        "occlusion_threshold",
        "robust_projection_low_quantile",
        "robust_projection_high_quantile",
    }
)


def load_canonical_axis_registry(path: str | Path) -> CanonicalAxisRegistry:
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if payload.get("schema_version") != "paper_core_canonical_axis_v1":
        raise ValueError("Unsupported canonical-axis schema; formal v1 is required.")
    if payload.get("unit_system") != "meter":
        raise ValueError("Canonical-axis unit_system must be exactly 'meter'.")
    if payload.get("frozen_before_real_video_run") is not True:
        raise ValueError("Canonical-axis thresholds must be frozen before real-video use.")
    if payload.get("post_result_threshold_tuning") is not False:
        raise ValueError("Post-result canonical-axis threshold tuning is forbidden.")
    if payload.get("authenticity_label_used") is not False:
        raise ValueError("Canonical-axis configuration must be label blind.")
    if payload.get("universal_fallback") is not False:
        raise ValueError("Canonical-axis configuration cannot enable a universal fallback.")

    raw_thresholds = payload.get("thresholds")
    if not isinstance(raw_thresholds, Mapping):
        raise ValueError("Canonical-axis thresholds must be a mapping.")
    missing = sorted(_CANONICAL_THRESHOLD_NAMES - set(raw_thresholds))
    extra = sorted(set(raw_thresholds) - _CANONICAL_THRESHOLD_NAMES)
    if missing or extra:
        raise ValueError(
            f"Canonical-axis threshold keys mismatch: missing={missing}, extra={extra}."
        )
    thresholds = {name: float(raw_thresholds[name]) for name in raw_thresholds}
    positive = _CANONICAL_THRESHOLD_NAMES - {"truncation_threshold"}
    if any(
        not math.isfinite(thresholds[name]) or thresholds[name] <= 0.0
        for name in positive
    ):
        raise ValueError("Canonical-axis positive thresholds must be finite and > 0.")
    if (
        not math.isfinite(thresholds["truncation_threshold"])
        or thresholds["truncation_threshold"] < 0.0
    ):
        raise ValueError("truncation_threshold must be finite and non-negative.")
    for name in (
        "bottle_depth_coverage_min",
        "bottle_axis_stability_min",
        "bottle_track_axis_compatibility_min",
        "bottle_mask_quality_min",
        "truncation_threshold",
        "occlusion_threshold",
        "robust_projection_low_quantile",
        "robust_projection_high_quantile",
    ):
        if thresholds[name] > 1.0:
            raise ValueError(f"Canonical-axis threshold {name!r} must be <= 1.")
    if not (
        0.0
        <= thresholds["robust_projection_low_quantile"]
        < thresholds["robust_projection_high_quantile"]
        <= 1.0
    ):
        raise ValueError("Canonical-axis robust projection quantiles are invalid.")
    metadata = payload.get("threshold_metadata")
    if not isinstance(metadata, Mapping) or set(metadata) != _CANONICAL_THRESHOLD_NAMES:
        raise ValueError("Every canonical-axis threshold requires metadata.")

    bottle = (payload.get("source_runtime_dimension_match") or {}).get("bottle")
    if not isinstance(bottle, Mapping):
        raise ValueError("Bottle source/runtime dimension audit is required.")
    source_field = str(bottle.get("source_field", ""))
    source_definition = str(bottle.get("source_dimension_definition", ""))
    runtime_definition = str(bottle.get("runtime_dimension_definition", ""))
    required_text = (
        source_field,
        str(bottle.get("source_coordinate_convention", "")),
        str(bottle.get("source_dimension_axis", "")),
        source_definition,
        str(bottle.get("runtime_axis_source", "")),
        runtime_definition,
        str(bottle.get("match_evidence", "")),
    )
    if any(not value.strip() for value in required_text):
        raise ValueError("Bottle source/runtime dimension provenance is incomplete.")
    return CanonicalAxisRegistry(
        schema_version=str(payload["schema_version"]),
        config_sha256=_sha256(config_path),
        thresholds=MappingProxyType(thresholds),
        bottle_source_runtime_dimension_match=bottle.get("ready") is True,
        bottle_source_field=source_field,
        bottle_source_dimension_definition=source_definition,
        bottle_runtime_dimension_definition=runtime_definition,
    )


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


def _axis_config(config: CanonicalAxisRegistry, name: str) -> float:
    return float(config.thresholds[name])


def _visible_surface_xyz(cloud: Any) -> np.ndarray:
    points = np.asarray(
        [[point.x_m, point.y_m, point.z_m] for point in cloud.points if point.valid],
        dtype=float,
    )
    if points.ndim != 2 or points.shape[1:] != (3,):
        return np.empty((0, 3), dtype=float)
    return points[np.isfinite(points).all(axis=1)]


def _stable_vector(vector: np.ndarray) -> np.ndarray:
    normalized = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(normalized))
    if not math.isfinite(norm) or norm <= 0.0:
        return np.full(3, float("nan"))
    normalized = normalized / norm
    pivot = int(np.argmax(np.abs(normalized)))
    return normalized if normalized[pivot] >= 0.0 else -normalized


def _pca3d_axis(cloud: Any, config: CanonicalAxisRegistry) -> dict[str, Any]:
    """Describe all three robust visible-surface PCA axes without naming one canonical."""

    points = _visible_surface_xyz(cloud)
    if len(points) < 3:
        return {
            "valid": False,
            "reason": "BOTTLE_AXIS_POINT_SUPPORT_INSUFFICIENT",
            "axis_point_support": int(len(points)),
        }
    center = np.median(points, axis=0)
    covariance = np.cov(points - center, rowvar=False)
    if covariance.shape != (3, 3) or not np.isfinite(covariance).all():
        return {
            "valid": False,
            "reason": "BOTTLE_AXIS_NUMERICALLY_UNSTABLE",
            "axis_point_support": int(len(points)),
        }
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    eigenvalues = np.maximum(values[order], 0.0)
    axes = np.asarray([_stable_vector(vectors[:, index]) for index in order])
    if not np.isfinite(axes).all():
        return {
            "valid": False,
            "reason": "BOTTLE_AXIS_NUMERICALLY_UNSTABLE",
            "axis_point_support": int(len(points)),
        }
    projections = (points - center) @ axes.T
    quantile_low = _axis_config(config, "robust_projection_low_quantile")
    quantile_high = _axis_config(config, "robust_projection_high_quantile")
    lows, highs = np.quantile(
        projections,
        [quantile_low, quantile_high],
        axis=0,
    )
    extents = highs - lows
    if not np.isfinite(extents).all() or extents[0] <= 0.0:
        return {
            "valid": False,
            "reason": "BOTTLE_AXIS_NUMERICALLY_UNSTABLE",
            "axis_point_support": int(len(points)),
        }
    eigenvalue_ratio = float(eigenvalues[0] / max(eigenvalues[1], 1e-12))
    extent_ratio = float(extents[0] / max(extents[1], extents[2], 1e-12))
    return {
        "valid": True,
        "reason": "",
        "axis_source": "robust_visible_surface_pca_3d_primary_axis",
        "axis_vector": [float(value) for value in axes[0]],
        "axis_vectors": [[float(value) for value in axis] for axis in axes],
        "axis_eigenvalues": [float(value) for value in eigenvalues],
        "axis_extents_m": [float(value) for value in extents],
        "principal_extent": float(extents[0]),
        "secondary_extent": float(extents[1]),
        "tertiary_extent": float(extents[2]),
        "axis_eigenvalue_ratio": eigenvalue_ratio,
        "axis_extent_ratio": extent_ratio,
        "eigenvalue_ratio": eigenvalue_ratio,
        "extent_ratio": extent_ratio,
        "elongation_ratio": min(eigenvalue_ratio, extent_ratio),
        "axis_point_support": int(len(points)),
        "extent_quantiles": [quantile_low, quantile_high],
        "visible_surface_only": True,
    }


def _bottle_height_axis(
    cloud: Any,
    config: CanonicalAxisRegistry,
    history: list[tuple[np.ndarray, float]],
) -> dict[str, Any]:
    if not config.bottle_source_runtime_dimension_match:
        return {
            "valid": False,
            "reason": "BOTTLE_SOURCE_RUNTIME_DIMENSION_MISMATCH",
        }
    diagnostics = _pca3d_axis(cloud, config)
    if not diagnostics.get("valid"):
        return diagnostics
    if int(diagnostics["axis_point_support"]) < int(
        _axis_config(config, "bottle_point_support_min")
    ):
        return {
            **diagnostics,
            "valid": False,
            "reason": "BOTTLE_AXIS_POINT_SUPPORT_INSUFFICIENT",
        }
    if float(diagnostics["axis_eigenvalue_ratio"]) < _axis_config(
        config, "bottle_eigenvalue_ratio_min"
    ) or float(diagnostics["axis_extent_ratio"]) < _axis_config(
        config, "bottle_extent_ratio_min"
    ):
        return {
            **diagnostics,
            "valid": False,
            "reason": "BOTTLE_NOT_ELONGATED_IN_3D",
        }
    axis = np.asarray(diagnostics["axis_vector"], dtype=float)
    extent = float(diagnostics["principal_extent"])
    axis_stability = min(
        float(diagnostics["axis_eigenvalue_ratio"])
        / (float(diagnostics["axis_eigenvalue_ratio"]) + 1.0),
        float(diagnostics["axis_extent_ratio"])
        / (float(diagnostics["axis_extent_ratio"]) + 1.0),
    )
    diagnostics["axis_stability"] = axis_stability
    if axis_stability < _axis_config(config, "bottle_axis_stability_min"):
        return {
            **diagnostics,
            "valid": False,
            "reason": "BOTTLE_AXIS_UNSTABLE",
        }
    if history:
        previous_axis, previous_extent = history[-1]
        compatibility = abs(float(np.dot(axis, previous_axis)))
        diagnostics["track_axis_compatibility"] = compatibility
        if compatibility < _axis_config(
            config, "bottle_track_axis_compatibility_min"
        ):
            return {
                **diagnostics,
                "valid": False,
                "reason": "BOTTLE_TRACK_AXIS_INCOMPATIBLE",
            }
        if abs(math.log(extent) - math.log(previous_extent)) > _axis_config(
            config, "bottle_track_extent_log_delta_max"
        ):
            return {
                **diagnostics,
                "valid": False,
                "reason": "BOTTLE_TRACK_SIZE_INCOMPATIBLE",
            }
    history.append((axis, extent))
    return {
        **diagnostics,
        "valid": True,
        "axis_quality": min(
            1.0,
            axis_stability,
            float(cloud.valid_point_ratio),
            float(cloud.depth_quality),
        ),
        "estimated_size_m": extent,
        "canonical_mapping_rule": "exact_bottle_dominant_visible_surface_pca3d_to_height_v1",
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
    canonical_axis_config: CanonicalAxisRegistry,
    bottle_history: list[tuple[np.ndarray, float]],
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
    records = {
        dimension: {
            "dimension": dimension,
            "canonical_dimension": dimension,
            "observable": False,
            "reason": "CANONICAL_MAPPING_UNAVAILABLE",
            "axis_source": "",
            "axis_vector": [],
            "axis_quality": 0.0,
            "canonical_mapping_rule": "",
            "estimated_size_m": float("nan"),
            "prior_min_m": float("nan"),
            "prior_max_m": float("nan"),
            "residual": float("nan"),
            "confidence": 0.0,
            "viewpoint_evidence": False,
            "viewpoint_class": view.viewpoint_class.value,
            "pose_estimate_status": view.pose_estimate_status.value,
            "visible_surface_only": True,
            "axis_diagnostics": {},
            "mask_support": dict(mask_support),
        }
        for dimension in ("height", "width", "length")
    }
    if obj.category == "person":
        records["height"].update(
            {
                "reason": "PERSON_POSE_PROVIDER_MISSING",
                "axis_diagnostics": {
                    "valid": False,
                    "reason": "PERSON_POSE_PROVIDER_MISSING",
                    "active_path_keypoint_provider": False,
                    "bbox_height_used": False,
                    "mask_image_axis_used": False,
                    "human_ratio_extrapolation_used": False,
                },
                "pose_axis_source": "",
                "pose_keypoint_support": {},
                "upright_status": "unavailable",
            }
        )
    elif obj.category == "bottle":
        if (
            obj.truncated
            or view.border_contact_ratio
            > _axis_config(canonical_axis_config, "truncation_threshold")
        ):
            bottle = {"valid": False, "reason": "BOTTLE_TRUNCATED"}
        else:
            bottle = _bottle_height_axis(
                cloud,
                canonical_axis_config,
                bottle_history,
            )
        records["height"].update(
            {
                "observable": bool(bottle.get("valid")),
                "reason": "" if bottle.get("valid") else str(bottle["reason"]),
                "axis_source": str(bottle.get("axis_source", "")),
                "axis_vector": list(bottle.get("axis_vector", ())),
                "axis_quality": float(bottle.get("axis_quality", 0.0)),
                "canonical_mapping_rule": str(
                    bottle.get("canonical_mapping_rule", "")
                ),
                "estimated_size_m": float(
                    bottle.get("estimated_size_m", float("nan"))
                ),
                "viewpoint_evidence": bool(bottle.get("valid")),
                "axis_diagnostics": dict(bottle),
                "principal_extent": float(
                    bottle.get("principal_extent", float("nan"))
                ),
                "secondary_extent": float(
                    bottle.get("secondary_extent", float("nan"))
                ),
                "eigenvalue_ratio": float(
                    bottle.get("eigenvalue_ratio", float("nan"))
                ),
                "extent_ratio": float(
                    bottle.get("extent_ratio", float("nan"))
                ),
                "axis_point_support": int(
                    bottle.get("axis_point_support", 0)
                ),
                "axis_stability": float(
                    bottle.get("axis_stability", float("nan"))
                ),
                "elongation_ratio": float(
                    bottle.get("elongation_ratio", float("nan"))
                ),
            }
        )
    elif obj.category == "car":
        records["length"]["reason"] = "CAR_INDEPENDENT_VIEWPOINT_AXIS_UNAVAILABLE"
        records["length"]["canonical_mapping_rule"] = (
            "car_length_requires_independent_longitudinal_lateral_vertical_axes_v1"
        )
    principal_observable = bool(
        container_axis
        and view.valid
        and math.isfinite(view.border_contact_ratio)
        and view.border_contact_ratio == 0.0
    )
    records["principal_extent"] = {
        "dimension": "principal_extent",
        "canonical_dimension": "",
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
        "axis_vector": (
            [*axis_diagnostics.get("axis_xy", ()), 0.0]
            if principal_observable
            else []
        ),
        "axis_quality": (
            min(1.0, float(axis_diagnostics["axis_eigenvalue_ratio"]) / 1.25)
            if principal_observable
            else 0.0
        ),
        "canonical_mapping_rule": "none_generic_temporal_extent_only",
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
    if reason and reason.upper() == reason:
        return reason
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
    canonical_axis_path: str | Path,
    min_depth_coverage: float = 0.5,
    max_occlusion_ratio: float = 0.5,
    min_mask_quality: float = 0.3,
) -> list[ResidualEvidence]:
    registry = load_scale_prior_registry(prior_path)
    canonical_axis = load_canonical_axis_registry(canonical_axis_path)
    output: list[ResidualEvidence] = []
    history: dict[tuple[str, str], list[tuple[int, float, float, str]]] = {}
    bottle_histories: dict[str, list[tuple[np.ndarray, float]]] = {}
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
                "class_name": obj.category,
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
                "canonical_axis_schema_version": canonical_axis.schema_version,
                "canonical_threshold_config_sha256": canonical_axis.config_sha256,
                "source_runtime_dimension_match": (
                    canonical_axis.bottle_source_runtime_dimension_match
                    if obj.category == "bottle"
                    else None
                ),
                "objectron_source_field": (
                    canonical_axis.bottle_source_field
                    if obj.category == "bottle"
                    else ""
                ),
                "legacy_container_upright_fallback_used": False,
                "canonical_dimension": "",
                "axis_source": "",
                "axis_vector": [],
                "axis_quality": 0.0,
                "canonical_mapping_rule": "",
                "estimated_size_m": float("nan"),
                "prior_min_m": float("nan"),
                "prior_max_m": float("nan"),
                "residual": float("nan"),
                "confidence": 0.0,
                "visible_surface_only": True,
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
            object_min_depth_coverage = (
                _axis_config(canonical_axis, "bottle_depth_coverage_min")
                if obj.category == "bottle"
                else min_depth_coverage
            )
            object_max_occlusion_ratio = (
                _axis_config(canonical_axis, "occlusion_threshold")
                if obj.category == "bottle"
                else max_occlusion_ratio
            )
            object_min_mask_quality = (
                _axis_config(canonical_axis, "bottle_mask_quality_min")
                if obj.category == "bottle"
                else min_mask_quality
            )

            common_reason = ""
            if obj.instance_mask is None or not np.any(obj.instance_mask):
                common_reason = "instance_mask_unavailable"
            elif not bool(mask_support["valid"]):
                common_reason = str(mask_support["reason"])
            elif obj.occlusion_ratio > object_max_occlusion_ratio:
                common_reason = "severe_object_occlusion"
            elif obj.mask_quality < object_min_mask_quality:
                common_reason = "insufficient_mask_quality"
            elif not obj.track_identity_stable:
                common_reason = "unstable_track_identity"
            cloud = (
                None
                if common_reason
                else build_metric_object_surface(
                    frame,
                    obj,
                    quantile_low=_axis_config(
                        canonical_axis, "robust_projection_low_quantile"
                    ),
                    quantile_high=_axis_config(
                        canonical_axis, "robust_projection_high_quantile"
                    ),
                )
            )
            if not common_reason and (cloud is None or not cloud.valid):
                common_reason = "metric_object_surface_unavailable"
            if (
                not common_reason
                and cloud is not None
                and cloud.valid_point_ratio
                < object_min_depth_coverage
            ):
                common_reason = "insufficient_valid_metric_depth_ratio"
            dimensions = (
                {}
                if cloud is None or not cloud.valid
                else _dimension_observability(
                    frame,
                    obj,
                    cloud,
                    mask_support,
                    canonical_axis,
                    bottle_histories.setdefault(obj.track_id, []),
                )
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
            if prior is not None and obj.category == "person":
                prior_reason = "PERSON_POSE_PROVIDER_MISSING"
            elif prior is None:
                prior_reason = (
                    "category_too_broad_without_subtype"
                    if unsupported
                    else "missing_category_metric_prior"
                )
            elif (
                not prior_reason
                and cloud is not None
                and cloud.valid_point_ratio
                < max(object_min_depth_coverage, prior.minimum_observability)
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
                        "canonical_dimension": prior.dimension,
                        "axis_source": selected["axis_source"],
                        "axis_vector": list(selected.get("axis_vector", ())),
                        "axis_quality": float(selected.get("axis_quality", 0.0)),
                        "canonical_mapping_rule": str(
                            selected.get("canonical_mapping_rule", "")
                        ),
                        "estimated_size_m": float(
                            selected.get("estimated_size_m", float("nan"))
                        ),
                        "prior_min_m": prior.min_meters,
                        "prior_max_m": prior.max_meters,
                        "observability_reason": selected["reason"],
                    }
                )
                for name in (
                    "pose_axis_source",
                    "pose_keypoint_support",
                    "upright_status",
                    "principal_extent",
                    "secondary_extent",
                    "eigenvalue_ratio",
                    "extent_ratio",
                    "axis_point_support",
                    "axis_stability",
                    "elongation_ratio",
                ):
                    if name in selected:
                        base_meta[name] = selected[name]
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
                    float(selected["axis_quality"]),
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
                    "canonical_dimension": prior.dimension,
                    "observable": True,
                    "observability_reason": (
                        "dimension_axis_and_visible_metric_surface_supported"
                    ),
                    "axis_source": selected["axis_source"],
                    "axis_vector": list(selected["axis_vector"]),
                    "axis_quality": float(selected["axis_quality"]),
                    "canonical_mapping_rule": selected["canonical_mapping_rule"],
                    "estimated_size_m": estimate,
                    "prior_min_m": prior.min_meters,
                    "prior_max_m": prior.max_meters,
                    "dimension_definition": prior.dimension_definition,
                    "applicable_scope": prior.applicable_scope,
                    "excluded_scope": prior.excluded_scope,
                    "visible_surface_only": True,
                    "world_frame_claimed": False,
                    "residual": residual,
                    "confidence": confidence,
                }
                for name in (
                    "pose_axis_source",
                    "pose_keypoint_support",
                    "upright_status",
                    "principal_extent",
                    "secondary_extent",
                    "eigenvalue_ratio",
                    "extent_ratio",
                    "axis_point_support",
                    "axis_stability",
                    "elongation_ratio",
                ):
                    if name in selected:
                        metadata[name] = selected[name]
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
                        for name in ("principal_extent", "height", "width", "length")
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
