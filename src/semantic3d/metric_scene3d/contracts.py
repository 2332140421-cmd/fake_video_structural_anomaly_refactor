"""Auditable contracts for single-frame metric visible-surface reconstruction."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional

import numpy as np

from ..shared_3d_observation import CoordinateFrame


class MetricPointType(str, Enum):
    """Semantic role of a point in the single-frame structure."""

    SCENE_SURFACE_POINT = "scene_surface_point"
    DENSE_OBJECT_SURFACE_POINT = "dense_object_surface_point"
    BOUNDARY_POINT = "boundary_point"
    GEOMETRIC_TRACK_POINT = "geometric_track_point"
    SEMANTIC_KEYPOINT = "semantic_keypoint"


class Visibility(str, Enum):
    """Visibility represented by the source observation."""

    VISIBLE = "visible"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class MetricSurfacePoint:
    """One metric 3D point with image location and complete provenance."""

    point_id: str
    point_type: MetricPointType | str
    frame_id: str
    object_id: Optional[str]
    track_id: Optional[str]
    u: float
    v: float
    x_m: float
    y_m: float
    z_m: float
    depth_confidence: float
    confidence: float
    uncertainty: float
    uncertainty_definition: str
    visibility: Visibility | str
    valid: bool
    failure_reason: str
    coordinate_frame: CoordinateFrame | str
    depth_unit: str
    depth_definition: str
    intrinsics_source: str
    pose_source: str
    provider_name: str
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        point_type = MetricPointType(self.point_type)
        visibility = Visibility(self.visibility)
        coordinate_frame = CoordinateFrame(self.coordinate_frame)
        if coordinate_frame != CoordinateFrame.CAMERA_FRAME_METRIC:
            raise ValueError("M2 metric surface points require camera_frame_metric.")
        confidence = float(self.confidence)
        depth_confidence = float(self.depth_confidence)
        if any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0
            for value in (confidence, depth_confidence)
        ):
            raise ValueError("Point confidence values must be finite and in [0, 1].")
        xyz = np.asarray([self.x_m, self.y_m, self.z_m], dtype=float)
        if self.valid:
            if not np.isfinite(xyz).all() or self.z_m <= 0.0:
                raise ValueError("Valid metric points require finite xyz and positive z.")
            if self.failure_reason:
                raise ValueError("Valid points cannot have failure_reason.")
            if self.depth_unit != "meter" or self.depth_definition != "z_depth":
                raise ValueError("Metric points require meter z_depth.")
            if not self.intrinsics_source or not self.provider_name:
                raise ValueError("Valid points require intrinsics and provider provenance.")
        else:
            if not np.isnan(xyz).all():
                raise ValueError("Invalid metric points must preserve missing xyz as NaN.")
            if not self.failure_reason:
                raise ValueError("Invalid metric points require failure_reason.")
        object.__setattr__(self, "point_type", point_type)
        object.__setattr__(self, "visibility", visibility)
        object.__setattr__(self, "coordinate_frame", coordinate_frame)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "depth_confidence", depth_confidence)
        object.__setattr__(self, "uncertainty", float(self.uncertainty))
        object.__setattr__(self, "provenance", dict(self.provenance))

    @classmethod
    def missing(
        cls,
        *,
        point_id: str,
        point_type: MetricPointType | str,
        frame_id: str,
        u: float,
        v: float,
        reason: str,
        object_id: Optional[str] = None,
        track_id: Optional[str] = None,
        provider_name: str = "unavailable",
        provenance: Optional[Mapping[str, Any]] = None,
    ) -> "MetricSurfacePoint":
        """Create an invalid point without fabricating zero coordinates."""

        return cls(
            point_id=point_id,
            point_type=point_type,
            frame_id=frame_id,
            object_id=object_id,
            track_id=track_id,
            u=float(u),
            v=float(v),
            x_m=float("nan"),
            y_m=float("nan"),
            z_m=float("nan"),
            depth_confidence=0.0,
            confidence=0.0,
            uncertainty=float("nan"),
            uncertainty_definition="unavailable",
            visibility=Visibility.UNKNOWN,
            valid=False,
            failure_reason=reason,
            coordinate_frame=CoordinateFrame.CAMERA_FRAME_METRIC,
            depth_unit="meter",
            depth_definition="z_depth",
            intrinsics_source="unavailable",
            pose_source="unavailable",
            provider_name=provider_name,
            provenance=dict(provenance or {}),
        )

    def xyz(self) -> np.ndarray:
        """Return xyz for valid points only."""

        if not self.valid:
            raise ValueError(f"Point {self.point_id!r} is invalid: {self.failure_reason}.")
        return np.asarray([self.x_m, self.y_m, self.z_m], dtype=float)


@dataclass(frozen=True)
class BoundaryMetricPoint:
    """A visible-mask boundary point with foreground/background depth evidence."""

    point: MetricSurfacePoint
    foreground_depth_m: float
    background_depth_m: float
    boundary_depth_jump_m: float
    boundary_order: int

    def __post_init__(self) -> None:
        if self.point.point_type != MetricPointType.BOUNDARY_POINT:
            raise ValueError("BoundaryMetricPoint requires point_type=boundary_point.")
        if self.point.valid and not math.isfinite(float(self.foreground_depth_m)):
            raise ValueError("Valid boundary points require foreground depth.")


@dataclass(frozen=True)
class ObjectSurfacePointCloud:
    """Visible object surface and robust metric extent in one camera frame."""

    object_id: str
    track_id: Optional[str]
    class_name: str
    frame_id: str
    points: tuple[MetricSurfacePoint, ...]
    point_count: int
    valid_point_ratio: float
    x_extent_m: float
    y_extent_m: float
    z_extent_m: float
    robust_centroid_m: tuple[float, float, float]
    robust_covariance: tuple[tuple[float, float, float], ...]
    point_uncertainty: float
    mask_quality: float
    depth_quality: float
    quantile_low: float
    quantile_high: float
    valid: bool
    failure_reason: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.valid_point_ratio <= 1.0:
            raise ValueError("valid_point_ratio must be in [0, 1].")
        if self.valid:
            values = (
                self.x_extent_m,
                self.y_extent_m,
                self.z_extent_m,
                *self.robust_centroid_m,
            )
            if not all(math.isfinite(float(value)) for value in values):
                raise ValueError("Valid point clouds require finite robust geometry.")
            if min(self.x_extent_m, self.y_extent_m, self.z_extent_m) < 0.0:
                raise ValueError("Robust extents cannot be negative.")
            if self.point_count <= 0 or self.failure_reason:
                raise ValueError("Valid point clouds require points and no failure reason.")
        elif not self.failure_reason:
            raise ValueError("Invalid point clouds require failure_reason.")
        object.__setattr__(self, "points", tuple(self.points))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class StructureEdge3D:
    """One metric relationship in a single-frame object structure graph."""

    edge_id: str
    source_point_id: str
    target_point_id: str
    edge_length_m: float
    relative_depth_m: float
    direction_vector: tuple[float, float, float]
    edge_type: str
    confidence: float
    valid: bool
    failure_reason: str = ""

    def __post_init__(self) -> None:
        if self.valid:
            values = (self.edge_length_m, self.relative_depth_m, *self.direction_vector)
            if not all(math.isfinite(float(value)) for value in values):
                raise ValueError("Valid graph edges require finite metric geometry.")
            if self.edge_length_m <= 0.0 or self.failure_reason:
                raise ValueError("Valid graph edges require positive length.")
        elif not self.failure_reason:
            raise ValueError("Invalid graph edges require failure_reason.")


@dataclass(frozen=True)
class SingleFrameStructureGraph:
    """Object-local graph built only from independently observed current-frame points."""

    graph_id: str
    frame_id: str
    object_id: str
    track_id: Optional[str]
    nodes: tuple[MetricSurfacePoint, ...]
    edges: tuple[StructureEdge3D, ...]
    coordinate_frame: CoordinateFrame | str
    valid: bool
    quality: float
    failure_reason: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        frame = CoordinateFrame(self.coordinate_frame)
        if frame != CoordinateFrame.CAMERA_FRAME_METRIC:
            raise ValueError("M2 structure graphs require camera_frame_metric.")
        if self.valid and (not self.nodes or not self.edges or self.failure_reason):
            raise ValueError("Valid structure graphs require nodes and edges.")
        if not self.valid and not self.failure_reason:
            raise ValueError("Invalid structure graphs require failure_reason.")
        object.__setattr__(self, "nodes", tuple(self.nodes))
        object.__setattr__(self, "edges", tuple(self.edges))
        object.__setattr__(self, "coordinate_frame", frame)
        object.__setattr__(self, "metadata", dict(self.metadata))
