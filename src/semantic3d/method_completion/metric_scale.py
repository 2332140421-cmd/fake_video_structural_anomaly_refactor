"""Metric single-object scale recovery and physical-prior residuals.

This module is the primary P4-C3A-MD2 scale route. It accepts only explicit
metric or sensor-metric depth in meters and never promotes monocular relative
depth to a physical size.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional

import numpy as np

from ..depth_provider import DepthObservation, DepthRepresentation, DepthScaleStatus
from ..geometry.camera import CameraObservation
from .multi_interval_prior import (
    DimensionScalePrior,
    MultiIntervalScalePriorRegistry,
    log_distance_to_interval_union,
)
from .scale_evidence import (
    ProviderStatus,
    ScaleBranchName,
    ScaleEvidenceRole,
    ScaleGeometryEvidence,
)


class MetricDepthType(str, Enum):
    """Origin of metric depth values."""

    METRIC = "metric"
    SENSOR_METRIC = "sensor_metric"
    RELATIVE = "relative"
    UNKNOWN = "unknown"


class MetricScaleStatus(str, Enum):
    """How metric scale was established."""

    MODEL_PREDICTED = "model_predicted"
    SENSOR_GROUND_TRUTH = "sensor_ground_truth"
    CALIBRATED = "calibrated"
    UNAVAILABLE = "unavailable"
    INVALID = "invalid"


class MetricDepthDefinition(str, Enum):
    """Geometric meaning of a positive depth sample."""

    Z_DEPTH = "z_depth"
    RAY_DISTANCE = "ray_distance"
    UNKNOWN = "unknown"


class ExtentEstimator(str, Enum):
    """Supported object-size estimators."""

    PROJECTED_EXTENT = "projected_extent"
    MASK_POINTCLOUD_EXTENT = "mask_pointcloud_extent"


@dataclass(frozen=True)
class MetricDepthEvidence:
    """Metric depth map with explicit unit, definition, and provider state."""

    depth_map: Optional[np.ndarray]
    valid_mask: Optional[np.ndarray]
    confidence_map: Optional[np.ndarray]
    depth_type: MetricDepthType | str
    depth_unit: str
    scale_status: MetricScaleStatus | str
    depth_definition: MetricDepthDefinition | str
    provider_name: str
    provider_status: ProviderStatus | str
    quality: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        depth_type = MetricDepthType(self.depth_type)
        scale_status = MetricScaleStatus(self.scale_status)
        definition = MetricDepthDefinition(self.depth_definition)
        provider_status = ProviderStatus(self.provider_status)
        depth = None if self.depth_map is None else np.asarray(self.depth_map, dtype=float)
        valid = None if self.valid_mask is None else np.asarray(self.valid_mask, dtype=bool)
        confidence = (
            None if self.confidence_map is None else np.asarray(self.confidence_map, dtype=float)
        )
        if depth is not None and depth.ndim != 2:
            raise ValueError("Metric depth_map must be HxW.")
        if valid is not None and depth is not None and valid.shape != depth.shape:
            raise ValueError("valid_mask must match depth_map.")
        if confidence is not None and depth is not None and confidence.shape != depth.shape:
            raise ValueError("confidence_map must match depth_map.")
        quality = float(self.quality)
        if not math.isfinite(quality) or not 0.0 <= quality <= 1.0:
            raise ValueError("MetricDepthEvidence.quality must be in [0, 1].")
        object.__setattr__(self, "depth_map", depth)
        object.__setattr__(self, "valid_mask", valid)
        object.__setattr__(self, "confidence_map", confidence)
        object.__setattr__(self, "depth_type", depth_type)
        object.__setattr__(self, "scale_status", scale_status)
        object.__setattr__(self, "depth_definition", definition)
        object.__setattr__(self, "provider_status", provider_status)
        object.__setattr__(self, "quality", quality)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def from_depth_observation(cls, observation: DepthObservation) -> "MetricDepthEvidence":
        """Adapt canonical depth only when metric semantics are explicit."""

        metadata = dict(observation.metadata)
        metric_status = str(metadata.get("metric_scale_status", "unavailable"))
        depth_type = str(metadata.get("depth_type", "unknown"))
        definition = str(metadata.get("depth_definition", "unknown"))
        if observation.depth_representation == DepthRepresentation.METRIC_DEPTH:
            depth_type = depth_type if depth_type in {"metric", "sensor_metric"} else "metric"
        if observation.scale_status != DepthScaleStatus.METRIC_CALIBRATED:
            metric_status = "unavailable"
        return cls(
            depth_map=observation.depth_map,
            valid_mask=observation.valid_mask,
            confidence_map=observation.confidence_map,
            depth_type=depth_type,
            depth_unit=str(metadata.get("depth_unit", "unknown")),
            scale_status=metric_status,
            depth_definition=definition,
            provider_name=observation.provider_name,
            provider_status=(ProviderStatus.OK if observation.valid else ProviderStatus.PROVIDER_FAILED),
            quality=observation.quality,
            metadata=metadata,
        )


@dataclass(frozen=True)
class MetricObjectRegion:
    """Object region and observation-quality data used by the metric route."""

    video_id: str
    clip_id: str
    frame_id: str
    object_id: str
    track_id: str
    class_name: str
    bbox: tuple[float, float, float, float]
    image_shape: tuple[int, int]
    mask: Optional[np.ndarray] = None
    detection_confidence: float = 1.0
    provider_status: ProviderStatus | str = ProviderStatus.OK
    border_contacts: frozenset[str] = frozenset()
    severe_truncation: bool = False
    out_of_frame_ratio: float = 0.0
    occlusion_ratio: float = 0.0
    pose_status: str = "resolved"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        shape = tuple(int(item) for item in self.image_shape)
        if len(shape) != 2 or min(shape) <= 0:
            raise ValueError("image_shape must be positive (height, width).")
        bbox = tuple(float(item) for item in self.bbox)
        if len(bbox) != 4:
            raise ValueError("bbox must be [x1, y1, x2, y2].")
        mask = None if self.mask is None else np.asarray(self.mask, dtype=bool)
        if mask is not None and mask.shape != shape:
            raise ValueError("Object mask must match image_shape.")
        confidence = float(self.detection_confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("detection_confidence must be in [0, 1].")
        for name, value in (
            ("out_of_frame_ratio", self.out_of_frame_ratio),
            ("occlusion_ratio", self.occlusion_ratio),
        ):
            if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1].")
        object.__setattr__(self, "bbox", bbox)
        object.__setattr__(self, "image_shape", shape)
        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "provider_status", ProviderStatus(self.provider_status))
        object.__setattr__(self, "border_contacts", frozenset(self.border_contacts))
        object.__setattr__(self, "detection_confidence", confidence)
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class SizeObservability:
    """Independent observability decisions for object physical dimensions."""

    height_observable: bool
    width_observable: bool
    depth_extent_observable: bool
    reasons: Mapping[str, tuple[str, ...]]
    length_observable: bool = False

    @property
    def none_observable(self) -> bool:
        return not (
            self.height_observable
            or self.width_observable
            or self.length_observable
            or self.depth_extent_observable
        )


@dataclass(frozen=True)
class MetricExtentEstimate:
    """Robust metric extent result in the OpenCV camera coordinate system."""

    x_extent_m: float
    y_extent_m: float
    z_extent_m: float
    projected_width_px: float
    projected_height_px: float
    depth_m: float
    depth_confidence: float
    point_count: int
    valid_point_ratio: float
    extent_estimator: str
    extent_uncertainty: float
    valid: bool
    failure_reason: str = ""


@dataclass(frozen=True)
class MetricScaleThresholds:
    """Transparent engineering gates for metric scale recovery."""

    min_detection_confidence: float = 0.3
    min_depth_confidence: float = 0.3
    min_valid_depth_ratio: float = 0.5
    max_out_of_frame_ratio: float = 0.1
    max_occlusion_ratio: float = 0.5
    min_point_count: int = 20
    quantile_low: float = 0.05
    quantile_high: float = 0.95
    allow_approximated_intrinsics: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.quantile_low < self.quantile_high <= 1.0:
            raise ValueError("Extent quantiles must satisfy 0 <= low < high <= 1.")
        if self.min_point_count < 1:
            raise ValueError("min_point_count must be positive.")


@dataclass(frozen=True)
class MetricSingleObjectScaleResult:
    """Primary evidence plus all dimension-level metric diagnostics."""

    evidence: ScaleGeometryEvidence
    observability: SizeObservability
    extent: Optional[MetricExtentEstimate]
    dimension_residuals: Mapping[str, float]
    dimension_intervals: Mapping[str, tuple[tuple[float, float], ...]]
    estimated_dimensions_m: Mapping[str, float]


def _clipped_bbox(
    bbox: tuple[float, float, float, float], image_shape: tuple[int, int]
) -> tuple[int, int, int, int]:
    height, width = image_shape
    x1, y1, x2, y2 = bbox
    return (
        max(0, min(width, int(math.floor(x1)))),
        max(0, min(height, int(math.floor(y1)))),
        max(0, min(width, int(math.ceil(x2)))),
        max(0, min(height, int(math.ceil(y2)))),
    )


def _region_mask(obj: MetricObjectRegion) -> tuple[Optional[np.ndarray], str]:
    if obj.mask is not None and np.any(obj.mask):
        return obj.mask.copy(), "mask"
    x1, y1, x2, y2 = _clipped_bbox(obj.bbox, obj.image_shape)
    if x2 <= x1 or y2 <= y1:
        return None, "invalid_bbox"
    region = np.zeros(obj.image_shape, dtype=bool)
    region[y1:y2, x1:x2] = True
    return region, "bbox"


def ray_distance_to_z_depth(
    ray_distance: np.ndarray, rows: np.ndarray, columns: np.ndarray, K: np.ndarray
) -> np.ndarray:
    """Convert Euclidean camera-ray distance to optical-axis Z depth."""

    matrix = np.asarray(K, dtype=float)
    fx, fy, cx, cy = matrix[0, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2]
    ray_norm = np.sqrt(((columns - cx) / fx) ** 2 + ((rows - cy) / fy) ** 2 + 1.0)
    return np.asarray(ray_distance, dtype=float) / ray_norm


def evaluate_size_observability(
    obj: MetricObjectRegion, *, for_pointcloud: bool = False
) -> SizeObservability:
    """Evaluate height, width, length, and visible depth range independently."""

    reasons: dict[str, list[str]] = {
        "height": [],
        "width": [],
        "length": [],
        "depth_extent": [],
    }
    height = width = length = depth = True
    if obj.severe_truncation or obj.out_of_frame_ratio > 0.1:
        height = width = length = depth = False
        for values in reasons.values():
            values.append("severe_truncation_or_out_of_frame")
    if {"top", "bottom"} & obj.border_contacts:
        height = False
        reasons["height"].append("vertical_border_contact")
    if {"left", "right"} & obj.border_contacts:
        width = False
        reasons["width"].append("horizontal_border_contact")
        length = False
        reasons["length"].append("horizontal_border_contact")
    pose = obj.pose_status.lower()
    if obj.class_name.lower() == "person" and pose in {"sitting", "bending", "kneeling"}:
        height = False
        reasons["height"].append("person_full_height_pose_not_applicable")
    if pose in {"oblique", "foreshortened", "unresolved"}:
        if bool(obj.metadata.get("width_pose_sensitive", True)):
            width = False
            reasons["width"].append("width_pose_not_observable")
        if bool(obj.metadata.get("length_pose_sensitive", True)):
            length = False
            reasons["length"].append("length_pose_not_observable")
    if obj.occlusion_ratio > 0.5:
        depth = False
        reasons["depth_extent"].append("heavy_occlusion")
        if bool(obj.metadata.get("height_occluded", False)):
            height = False
            reasons["height"].append("height_occluded")
        if bool(obj.metadata.get("width_occluded", False)):
            width = False
            reasons["width"].append("width_occluded")
    for name, current in (
        ("height", height),
        ("width", width),
        ("length", length),
        ("depth_extent", depth),
    ):
        override = obj.metadata.get(f"{name}_observable")
        if override is not None:
            if name == "height":
                height = bool(override)
            elif name == "width":
                width = bool(override)
            elif name == "length":
                length = bool(override)
            else:
                depth = bool(override)
            if not bool(override):
                reasons[name].append("explicit_observability_gate")
    if not for_pointcloud:
        depth = False
        reasons["depth_extent"].append("projected_extent_has_no_3d_depth_extent")
    return SizeObservability(
        height_observable=height,
        width_observable=width,
        depth_extent_observable=depth,
        reasons={key: tuple(value) for key, value in reasons.items()},
        length_observable=length,
    )


def _valid_region_samples(
    obj: MetricObjectRegion,
    depth: MetricDepthEvidence,
    camera: CameraObservation,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, str]:
    region, source = _region_mask(obj)
    if region is None or depth.depth_map is None:
        raise ValueError("invalid_object_region")
    if depth.depth_map.shape != obj.image_shape:
        raise ValueError("depth_image_shape_mismatch")
    valid = region & np.isfinite(depth.depth_map) & (depth.depth_map > 0.0)
    if depth.valid_mask is not None:
        valid &= depth.valid_mask
    region_count = int(np.count_nonzero(region))
    valid_count = int(np.count_nonzero(valid))
    ratio = valid_count / max(region_count, 1)
    if not valid_count:
        raise ValueError("no_valid_metric_depth_in_object")
    rows, columns = np.nonzero(valid)
    values = depth.depth_map[rows, columns]
    if depth.depth_definition == MetricDepthDefinition.RAY_DISTANCE:
        assert camera.K is not None
        values = ray_distance_to_z_depth(values, rows, columns, camera.K)
    confidence_values = (
        np.ones(valid_count, dtype=float)
        if depth.confidence_map is None
        else depth.confidence_map[rows, columns]
    )
    finite_confidence = confidence_values[np.isfinite(confidence_values)]
    confidence = float(np.median(finite_confidence)) if finite_confidence.size else depth.quality
    return rows, columns, values, confidence_values, ratio, confidence, source


def estimate_projected_extent(
    obj: MetricObjectRegion,
    depth: MetricDepthEvidence,
    camera: CameraObservation,
) -> MetricExtentEstimate:
    """Recover projected metric height/width with H=h_px*Z/fy and W=w_px*Z/fx."""

    try:
        rows, columns, z_values, _, ratio, confidence, source = _valid_region_samples(
            obj, depth, camera
        )
    except ValueError as exc:
        return _invalid_extent(ExtentEstimator.PROJECTED_EXTENT, str(exc))
    assert camera.K is not None
    h_px = float(rows.max() - rows.min() + 1)
    w_px = float(columns.max() - columns.min() + 1)
    z = float(np.median(z_values))
    fx, fy = float(camera.K[0, 0]), float(camera.K[1, 1])
    z_q05, z_q95 = np.quantile(z_values, [0.05, 0.95])
    uncertainty = float(max(0.0, z_q95 - z_q05) / max(z, 1e-12))
    return MetricExtentEstimate(
        x_extent_m=w_px * z / fx,
        y_extent_m=h_px * z / fy,
        z_extent_m=float(z_q95 - z_q05),
        projected_width_px=w_px,
        projected_height_px=h_px,
        depth_m=z,
        depth_confidence=confidence,
        point_count=int(z_values.size),
        valid_point_ratio=ratio,
        extent_estimator=f"projected_extent:{source}",
        extent_uncertainty=uncertainty,
        valid=True,
    )


def estimate_mask_pointcloud_extent(
    obj: MetricObjectRegion,
    depth: MetricDepthEvidence,
    camera: CameraObservation,
    *,
    quantile_low: float = 0.05,
    quantile_high: float = 0.95,
) -> MetricExtentEstimate:
    """Back-project mask pixels and estimate robust quantile 3D extents."""

    if obj.mask is None or not np.any(obj.mask):
        return _invalid_extent(ExtentEstimator.MASK_POINTCLOUD_EXTENT, "formal_mask_required")
    try:
        rows, columns, z_values, _, ratio, confidence, _ = _valid_region_samples(
            obj, depth, camera
        )
    except ValueError as exc:
        return _invalid_extent(ExtentEstimator.MASK_POINTCLOUD_EXTENT, str(exc))
    assert camera.K is not None
    fx, fy, cx, cy = (
        float(camera.K[0, 0]),
        float(camera.K[1, 1]),
        float(camera.K[0, 2]),
        float(camera.K[1, 2]),
    )
    x = (columns.astype(float) - cx) * z_values / fx
    y = (rows.astype(float) - cy) * z_values / fy
    points = np.column_stack((x, y, z_values))
    low = np.quantile(points, quantile_low, axis=0)
    high = np.quantile(points, quantile_high, axis=0)
    extents = high - low
    median = np.median(points, axis=0)
    mad = np.median(np.abs(points - median), axis=0)
    uncertainty = float(np.linalg.norm(mad) / max(np.linalg.norm(extents), 1e-12))
    return MetricExtentEstimate(
        x_extent_m=float(extents[0]),
        y_extent_m=float(extents[1]),
        z_extent_m=float(extents[2]),
        projected_width_px=float(columns.max() - columns.min() + 1),
        projected_height_px=float(rows.max() - rows.min() + 1),
        depth_m=float(np.median(z_values)),
        depth_confidence=confidence,
        point_count=int(points.shape[0]),
        valid_point_ratio=ratio,
        extent_estimator=f"mask_pointcloud_quantile_{quantile_low:g}_{quantile_high:g}",
        extent_uncertainty=uncertainty,
        valid=True,
    )


def _invalid_extent(estimator: ExtentEstimator, reason: str) -> MetricExtentEstimate:
    return MetricExtentEstimate(
        x_extent_m=float("nan"),
        y_extent_m=float("nan"),
        z_extent_m=float("nan"),
        projected_width_px=float("nan"),
        projected_height_px=float("nan"),
        depth_m=float("nan"),
        depth_confidence=0.0,
        point_count=0,
        valid_point_ratio=0.0,
        extent_estimator=estimator.value,
        extent_uncertainty=float("nan"),
        valid=False,
        failure_reason=reason,
    )


class MetricSingleObjectScaleBranch:
    """Priority-1 unary metric physical-scale residual branch."""

    def __init__(
        self,
        prior_registry: MultiIntervalScalePriorRegistry,
        *,
        estimator: ExtentEstimator | str = ExtentEstimator.MASK_POINTCLOUD_EXTENT,
        thresholds: MetricScaleThresholds = MetricScaleThresholds(),
        config_sha256: str = "",
        software_commit: str = "",
    ) -> None:
        self.prior_registry = prior_registry
        self.estimator = ExtentEstimator(estimator)
        self.thresholds = thresholds
        self.config_sha256 = config_sha256
        self.software_commit = software_commit

    def evaluate(
        self,
        obj: MetricObjectRegion,
        depth: MetricDepthEvidence,
        camera: CameraObservation,
    ) -> MetricSingleObjectScaleResult:
        """Recover dimensions, gate each dimension, and compare to interval unions."""

        observability = evaluate_size_observability(
            obj, for_pointcloud=self.estimator == ExtentEstimator.MASK_POINTCLOUD_EXTENT
        )
        base = self._evidence_base(obj, depth, camera)
        failure, provider_status = self._gate(obj, depth, camera)
        if failure:
            return MetricSingleObjectScaleResult(
                evidence=ScaleGeometryEvidence.missing(
                    failure_reason=failure,
                    provider_status=provider_status,
                    **base,
                ),
                observability=observability,
                extent=None,
                dimension_residuals={},
                dimension_intervals={},
                estimated_dimensions_m={},
            )
        resolved = self.prior_registry.resolve(obj.class_name)
        if resolved.prior is None:
            return MetricSingleObjectScaleResult(
                evidence=ScaleGeometryEvidence.missing(
                    failure_reason="missing_physical_scale_prior", **base
                ),
                observability=observability,
                extent=None,
                dimension_residuals={},
                dimension_intervals={},
                estimated_dimensions_m={},
            )
        extent = (
            estimate_mask_pointcloud_extent(
                obj,
                depth,
                camera,
                quantile_low=self.thresholds.quantile_low,
                quantile_high=self.thresholds.quantile_high,
            )
            if self.estimator == ExtentEstimator.MASK_POINTCLOUD_EXTENT
            else estimate_projected_extent(obj, depth, camera)
        )
        if not extent.valid:
            return MetricSingleObjectScaleResult(
                evidence=ScaleGeometryEvidence.missing(
                    failure_reason=extent.failure_reason, **base
                ),
                observability=observability,
                extent=extent,
                dimension_residuals={},
                dimension_intervals={},
                estimated_dimensions_m={},
            )
        if extent.valid_point_ratio < self.thresholds.min_valid_depth_ratio:
            return self._blocked_result(
                base, observability, extent, "insufficient_valid_metric_depth_ratio"
            )
        if extent.depth_confidence < self.thresholds.min_depth_confidence:
            return self._blocked_result(base, observability, extent, "low_metric_depth_confidence")
        if (
            self.estimator == ExtentEstimator.MASK_POINTCLOUD_EXTENT
            and extent.point_count < self.thresholds.min_point_count
        ):
            return self._blocked_result(base, observability, extent, "insufficient_pointcloud_support")

        values = {
            "height_m": extent.y_extent_m,
            "width_m": extent.x_extent_m,
            "length_m": extent.z_extent_m,
            "extent_m": max(extent.x_extent_m, extent.y_extent_m, extent.z_extent_m),
        }
        observable = {
            "height_m": observability.height_observable,
            "width_m": observability.width_observable,
            "length_m": observability.length_observable,
            "extent_m": not observability.none_observable,
        }
        residuals: dict[str, float] = {}
        intervals: dict[str, tuple[tuple[float, float], ...]] = {}
        estimates: dict[str, float] = {}
        priors = resolved.prior.dimensions
        for dimension, prior in priors.items():
            if dimension not in values or not observable.get(dimension, False):
                continue
            value = float(values[dimension])
            if not math.isfinite(value) or value <= 0.0:
                continue
            residuals[dimension] = log_distance_to_interval_union(value, prior.intervals)
            intervals[dimension] = tuple((item.low, item.high) for item in prior.intervals)
            estimates[dimension] = value
        if not residuals:
            return self._blocked_result(
                base, observability, extent, "no_observable_dimension_with_physical_prior"
            )
        combined = max(residuals.values())
        prior_confidence = min(
            priors[name].confidence for name in residuals if isinstance(priors[name], DimensionScalePrior)
        )
        quality = min(
            obj.detection_confidence,
            depth.quality,
            camera.quality,
            extent.depth_confidence,
            prior_confidence,
        )
        provenance = {
            "extent_estimator": extent.extent_estimator,
            "estimated_dimensions_m": estimates,
            "dimension_residuals": residuals,
            "dimension_prior_intervals_m": intervals,
            "prior_resolution": resolved.resolution,
            "resolved_class_name": resolved.resolved_label,
            "R_metric_height": residuals.get("height_m", float("nan")),
            "R_metric_width": residuals.get("width_m", float("nan")),
            "R_metric_length": residuals.get("length_m", float("nan")),
            "R_metric_extent": residuals.get("extent_m", float("nan")),
            "aggregation": "max_over_valid_dimension_log_distances",
            "provider_disagreement_is_anomaly": False,
            "metric_depth_is_sensor_ground_truth": depth.scale_status
            == MetricScaleStatus.SENSOR_GROUND_TRUTH,
        }
        evidence = ScaleGeometryEvidence.observed(
            residual_value=combined,
            confidence=quality,
            uncertainty=extent.extent_uncertainty,
            provenance=provenance,
            **{key: value for key, value in base.items() if key != "provenance"},
        )
        return MetricSingleObjectScaleResult(
            evidence=evidence,
            observability=observability,
            extent=extent,
            dimension_residuals=residuals,
            dimension_intervals=intervals,
            estimated_dimensions_m=estimates,
        )

    def _gate(
        self,
        obj: MetricObjectRegion,
        depth: MetricDepthEvidence,
        camera: CameraObservation,
    ) -> tuple[str, ProviderStatus]:
        if obj.provider_status == ProviderStatus.PROVIDER_FAILED:
            return "object_provider_failed", ProviderStatus.PROVIDER_FAILED
        if depth.provider_status == ProviderStatus.PROVIDER_FAILED:
            return "metric_depth_provider_failed", ProviderStatus.PROVIDER_FAILED
        if obj.provider_status != ProviderStatus.OK or depth.provider_status != ProviderStatus.OK:
            return "provider_not_ready", ProviderStatus.BLOCKED
        if depth.depth_type not in {MetricDepthType.METRIC, MetricDepthType.SENSOR_METRIC}:
            return "metric_scale_unavailable", ProviderStatus.BLOCKED
        if depth.depth_unit not in {"m", "meter", "meters"}:
            return "metric_depth_unit_not_meter", ProviderStatus.BLOCKED
        if depth.scale_status in {MetricScaleStatus.UNAVAILABLE, MetricScaleStatus.INVALID}:
            return "metric_scale_status_unavailable", ProviderStatus.BLOCKED
        if depth.depth_definition == MetricDepthDefinition.UNKNOWN:
            return "unknown_depth_definition", ProviderStatus.BLOCKED
        if depth.depth_map is None or depth.valid_mask is None:
            return "missing_metric_depth_map", ProviderStatus.BLOCKED
        if not camera.valid or camera.K is None:
            return "missing_camera_intrinsics", ProviderStatus.BLOCKED
        source = camera.intrinsics_source.strip().lower()
        if not self.thresholds.allow_approximated_intrinsics and (
            "approx" in source or "assumed" in source
        ):
            return "approximated_intrinsics_not_allowed", ProviderStatus.BLOCKED
        if obj.detection_confidence < self.thresholds.min_detection_confidence:
            return "low_object_detection_confidence", ProviderStatus.BLOCKED
        if obj.severe_truncation:
            return "severe_object_truncation", ProviderStatus.BLOCKED
        if obj.out_of_frame_ratio > self.thresholds.max_out_of_frame_ratio:
            return "object_largely_out_of_frame", ProviderStatus.BLOCKED
        if _region_mask(obj)[0] is None:
            return "invalid_object_region", ProviderStatus.BLOCKED
        return "", ProviderStatus.OK

    def _evidence_base(
        self,
        obj: MetricObjectRegion,
        depth: MetricDepthEvidence,
        camera: CameraObservation,
    ) -> dict[str, Any]:
        coordinate = (
            camera.coordinate_convention.value
            if hasattr(camera.coordinate_convention, "value")
            else str(camera.coordinate_convention)
        )
        return {
            "video_id": obj.video_id,
            "clip_id": obj.clip_id,
            "frame_id": obj.frame_id,
            "object_id": obj.object_id,
            "track_id": obj.track_id,
            "branch_name": ScaleBranchName.METRIC_SINGLE_OBJECT,
            "branch_priority": 1,
            "evidence_role": ScaleEvidenceRole.PRIMARY,
            "residual_name": "R_metric_abs",
            "depth_type": depth.depth_type.value,
            "depth_unit": depth.depth_unit,
            "depth_definition": depth.depth_definition.value,
            "coordinate_system": coordinate,
            "localization_reference": f"object_mask:{obj.object_id}",
            "provenance": {
                "depth_provider": depth.provider_name,
                "intrinsics_source": camera.intrinsics_source,
                "metric_scale_status": depth.scale_status.value,
            },
            "config_sha256": self.config_sha256,
            "software_commit": self.software_commit,
        }

    @staticmethod
    def _blocked_result(
        base: Mapping[str, Any],
        observability: SizeObservability,
        extent: MetricExtentEstimate,
        reason: str,
    ) -> MetricSingleObjectScaleResult:
        return MetricSingleObjectScaleResult(
            evidence=ScaleGeometryEvidence.missing(failure_reason=reason, **dict(base)),
            observability=observability,
            extent=extent,
            dimension_residuals={},
            dimension_intervals={},
            estimated_dimensions_m={},
        )


# Compatibility alias: old public code may keep importing the route concept.
MetricAbsoluteScaleBranch = MetricSingleObjectScaleBranch
