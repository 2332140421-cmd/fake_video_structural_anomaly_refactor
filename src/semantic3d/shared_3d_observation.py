"""Shared object-centric 3D observation contracts.

These P0 dataclasses define how future static and dynamic branches will share
camera, depth, object, and point evidence. They do not perform back-projection
or fabricate 3D coordinates when reconstruction inputs are unavailable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .depth_provider import DepthObservation, DepthScaleStatus
from .geometry.camera import CameraObservation
from .validity import MissingReason


class GeometryScaleStatus(str, Enum):
    """Scale domain for a reconstructed 3D quantity."""

    METRIC_3D = "metric_3d"
    RELATIVE_3D = "relative_3d"
    NORMALIZED_SHAPE = "normalized_shape"
    UNKNOWN = "unknown"


class VisibilityStatus(str, Enum):
    """Object or point visibility in the source frame."""

    VISIBLE = "visible"
    PARTIALLY_VISIBLE = "partially_visible"
    OCCLUDED = "occluded"
    OUT_OF_FRAME = "out_of_frame"
    UNKNOWN = "unknown"


class GeometryScaleUnit(str, Enum):
    """Unit carried by an observed 3D scale."""

    METER = "meter"
    RELATIVE_UNIT = "relative_unit"
    UNITLESS = "unitless"
    UNKNOWN = "unknown"


class ReconstructionFrame(str, Enum):
    """Primary coordinate frame of an object reconstruction."""

    CAMERA = "camera"
    WORLD = "world"
    UNKNOWN = "unknown"


class CoordinateFrame(str, Enum):
    """Explicit coordinate frame carried by reconstructed 3D observations.

    ``camera`` and ``world`` remain accepted by compatibility helpers below,
    but new metric reconstruction code must use the more specific values.
    """

    CAMERA_FRAME_METRIC = "camera_frame_metric"
    CAMERA_FRAME_RELATIVE = "camera_frame_relative"
    CLIP_LOCAL_ALIGNED = "clip_local_aligned"
    WORLD_FRAME = "world_frame"
    UNKNOWN = "unknown"


def is_camera_coordinate_frame(value: str | CoordinateFrame) -> bool:
    """Return whether a frame is a legacy or explicit camera coordinate frame."""

    normalized = value.value if isinstance(value, CoordinateFrame) else str(value)
    return normalized in {
        "camera",
        CoordinateFrame.CAMERA_FRAME_METRIC.value,
        CoordinateFrame.CAMERA_FRAME_RELATIVE.value,
    }


def is_world_coordinate_frame(value: str | CoordinateFrame) -> bool:
    """Return whether a frame is a legacy or explicit world coordinate frame."""

    normalized = value.value if isinstance(value, CoordinateFrame) else str(value)
    return normalized in {"world", CoordinateFrame.WORLD_FRAME.value}


@dataclass(frozen=True)
class Point2DObservation:
    """One image-plane point with explicit validity and provenance."""

    point_id: str
    x: Optional[float]
    y: Optional[float]
    confidence: float
    valid: bool
    missing_reason: str = ""
    source: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        confidence = float(self.confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("Point2DObservation.confidence must be in [0, 1].")
        if self.valid:
            if self.x is None or self.y is None:
                raise ValueError("Valid Point2DObservation requires x and y.")
            if not math.isfinite(float(self.x)) or not math.isfinite(float(self.y)):
                raise ValueError("Valid Point2DObservation coordinates must be finite.")
            if self.missing_reason:
                raise ValueError("Valid Point2DObservation cannot have missing_reason.")
        elif not self.missing_reason:
            raise ValueError("Invalid Point2DObservation requires missing_reason.")
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class Point3DObservation:
    """One 3D point in an explicit coordinate and scale domain."""

    point_id: str
    x: Optional[float]
    y: Optional[float]
    z: Optional[float]
    coordinate_frame: str
    scale_status: GeometryScaleStatus | str
    confidence: float
    valid: bool
    missing_reason: str = ""
    source_point_2d_id: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        scale_status = GeometryScaleStatus(self.scale_status)
        confidence = float(self.confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("Point3DObservation.confidence must be in [0, 1].")
        if self.valid:
            if self.x is None or self.y is None or self.z is None:
                raise ValueError("Valid Point3DObservation requires x, y, and z.")
            if not all(math.isfinite(float(value)) for value in (self.x, self.y, self.z)):
                raise ValueError("Valid Point3DObservation coordinates must be finite.")
            if not self.coordinate_frame.strip():
                raise ValueError("Valid Point3DObservation requires coordinate_frame.")
            if scale_status == GeometryScaleStatus.UNKNOWN:
                raise ValueError("Valid Point3DObservation requires an explicit scale domain.")
            if self.missing_reason:
                raise ValueError("Valid Point3DObservation cannot have missing_reason.")
        elif not self.missing_reason:
            raise ValueError("Invalid Point3DObservation requires missing_reason.")
        object.__setattr__(self, "scale_status", scale_status)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def as_array(self) -> np.ndarray:
        """Return xyz only for a valid point; invalid points cannot become zeros."""

        if not self.valid:
            raise ValueError(f"Point {self.point_id!r} is invalid: {self.missing_reason}.")
        assert self.x is not None and self.y is not None and self.z is not None
        return np.asarray([self.x, self.y, self.z], dtype=float)


@dataclass(frozen=True)
class Object3DObservation:
    """Shared 3D representation for one tracked semantic object.

    ``observed_scale_3d`` stores an explicitly reconstructed scale. Normalized
    structure points describe shape only and are never used to infer physical
    size without a separate scale observation.
    """

    video_id: str
    frame_index: int
    track_id: Optional[str]
    semantic_label: str
    canonical_label: str
    center_3d: Optional[Point3DObservation]
    boundary_points_3d: tuple[Point3DObservation, ...]
    keypoints_3d: tuple[Point3DObservation, ...]
    structure_points_3d: tuple[Point3DObservation, ...]
    observed_scale_3d: Optional[float]
    normalized_structure_points: tuple[Point3DObservation, ...]
    scale_status: GeometryScaleStatus | str
    visibility: VisibilityStatus | str
    reconstruction_quality: float
    valid: bool
    missing_reason: str
    source_object_2d_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    center_3d_camera: Optional[Point3DObservation] = None
    center_3d_world: Optional[Point3DObservation] = None
    boundary_points_3d_world: Optional[tuple[Point3DObservation, ...]] = None
    keypoints_3d_world: Optional[tuple[Point3DObservation, ...]] = None
    structure_points_3d_world: Optional[tuple[Point3DObservation, ...]] = None
    scale_method: str = ""
    scale_unit: GeometryScaleUnit | str = GeometryScaleUnit.UNKNOWN
    reconstruction_frame: ReconstructionFrame | str = ReconstructionFrame.CAMERA
    scale_quality: Optional[float] = None
    scale_descriptors: Mapping[str, Any] = field(default_factory=dict)
    depth_scale_status: DepthScaleStatus | str = DepthScaleStatus.UNKNOWN

    def __post_init__(self) -> None:
        scale_status = GeometryScaleStatus(self.scale_status)
        visibility = VisibilityStatus(self.visibility)
        scale_unit = GeometryScaleUnit(self.scale_unit)
        reconstruction_frame = ReconstructionFrame(self.reconstruction_frame)
        depth_scale_status = DepthScaleStatus(self.depth_scale_status)
        quality = float(self.reconstruction_quality)
        if not math.isfinite(quality) or not 0.0 <= quality <= 1.0:
            raise ValueError("reconstruction_quality must be finite and in [0, 1].")
        if self.observed_scale_3d is not None:
            scale = float(self.observed_scale_3d)
            if not math.isfinite(scale) or scale <= 0.0:
                raise ValueError("observed_scale_3d must be finite and positive.")
            if scale_status not in {
                GeometryScaleStatus.METRIC_3D,
                GeometryScaleStatus.RELATIVE_3D,
            }:
                raise ValueError("Observed 3D scale requires metric_3d or relative_3d status.")
            expected_unit = (
                GeometryScaleUnit.METER
                if scale_status == GeometryScaleStatus.METRIC_3D
                else GeometryScaleUnit.RELATIVE_UNIT
            )
            if scale_unit == GeometryScaleUnit.UNKNOWN:
                scale_unit = expected_unit
            elif scale_unit != expected_unit:
                raise ValueError(
                    f"{scale_status.value} requires scale_unit={expected_unit.value}."
                )
            object.__setattr__(self, "observed_scale_3d", scale)
        elif scale_status in {GeometryScaleStatus.METRIC_3D, GeometryScaleStatus.RELATIVE_3D}:
            raise ValueError("metric_3d/relative_3d status requires observed_scale_3d.")

        normalized_points = tuple(self.normalized_structure_points)
        if any(
            point.scale_status != GeometryScaleStatus.NORMALIZED_SHAPE
            for point in normalized_points
        ):
            raise ValueError("normalized_structure_points must use normalized_shape status.")
        physical_points = tuple(
            point
            for point in (
                (self.center_3d,) if self.center_3d is not None else ()
            )
            + tuple(self.boundary_points_3d)
            + tuple(self.keypoints_3d)
            + tuple(self.structure_points_3d)
            if point.valid
        )
        if any(point.scale_status == GeometryScaleStatus.NORMALIZED_SHAPE for point in physical_points):
            raise ValueError("Physical 3D points cannot use normalized_shape status.")
        if self.observed_scale_3d is not None and any(
            point.scale_status != scale_status for point in physical_points
        ):
            raise ValueError("Object points and observed_scale_3d must share one scale domain.")

        center_camera = self.center_3d_camera
        center_world = self.center_3d_world
        if center_camera is None and self.center_3d is not None:
            if is_camera_coordinate_frame(self.center_3d.coordinate_frame):
                center_camera = self.center_3d
            elif is_world_coordinate_frame(self.center_3d.coordinate_frame):
                center_world = self.center_3d
        if center_camera is not None and not is_camera_coordinate_frame(
            center_camera.coordinate_frame
        ):
            raise ValueError("center_3d_camera must use a camera coordinate frame.")
        if center_world is not None and not is_world_coordinate_frame(
            center_world.coordinate_frame
        ):
            raise ValueError("center_3d_world must use a world coordinate frame.")
        world_point_groups = (
            None
            if self.boundary_points_3d_world is None
            else tuple(self.boundary_points_3d_world),
            None
            if self.keypoints_3d_world is None
            else tuple(self.keypoints_3d_world),
            None
            if self.structure_points_3d_world is None
            else tuple(self.structure_points_3d_world),
        )
        if any(
            not is_world_coordinate_frame(point.coordinate_frame)
            for group in world_point_groups
            if group is not None
            for point in group
            if point.valid
        ):
            raise ValueError("World 3D point fields must use a world coordinate frame.")

        if self.scale_quality is None:
            scale_quality = quality if self.observed_scale_3d is not None else 0.0
        else:
            scale_quality = float(self.scale_quality)
        if not math.isfinite(scale_quality) or not 0.0 <= scale_quality <= 1.0:
            raise ValueError("scale_quality must be finite and in [0, 1].")
        if scale_status == GeometryScaleStatus.METRIC_3D:
            if depth_scale_status not in {
                DepthScaleStatus.METRIC_CALIBRATED,
                DepthScaleStatus.UNKNOWN,
            }:
                raise ValueError("Metric 3D scale cannot come from relative depth.")
        if reconstruction_frame == ReconstructionFrame.CAMERA and center_camera is None and self.valid:
            raise ValueError("Camera-frame reconstruction requires center_3d_camera.")
        if reconstruction_frame == ReconstructionFrame.WORLD and center_world is None and self.valid:
            raise ValueError("World-frame reconstruction requires center_3d_world.")
        if self.valid:
            if self.center_3d is None or not self.center_3d.valid:
                raise ValueError("Valid Object3DObservation requires a valid center_3d.")
            if self.missing_reason:
                raise ValueError("Valid Object3DObservation cannot have missing_reason.")
        elif not self.missing_reason:
            raise ValueError("Invalid Object3DObservation requires missing_reason.")

        object.__setattr__(self, "boundary_points_3d", tuple(self.boundary_points_3d))
        object.__setattr__(self, "keypoints_3d", tuple(self.keypoints_3d))
        object.__setattr__(self, "structure_points_3d", tuple(self.structure_points_3d))
        object.__setattr__(self, "normalized_structure_points", normalized_points)
        object.__setattr__(self, "scale_status", scale_status)
        object.__setattr__(self, "scale_unit", scale_unit)
        object.__setattr__(self, "reconstruction_frame", reconstruction_frame)
        object.__setattr__(self, "scale_quality", scale_quality)
        object.__setattr__(self, "scale_descriptors", dict(self.scale_descriptors))
        object.__setattr__(self, "depth_scale_status", depth_scale_status)
        object.__setattr__(self, "visibility", visibility)
        object.__setattr__(self, "reconstruction_quality", quality)
        object.__setattr__(self, "center_3d_camera", center_camera)
        object.__setattr__(self, "center_3d_world", center_world)
        object.__setattr__(self, "boundary_points_3d_world", world_point_groups[0])
        object.__setattr__(self, "keypoints_3d_world", world_point_groups[1])
        object.__setattr__(self, "structure_points_3d_world", world_point_groups[2])
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def missing(
        cls,
        video_id: str,
        frame_index: int,
        semantic_label: str,
        source_object_2d_id: str,
        track_id: Optional[str] = None,
        canonical_label: Optional[str] = None,
        reason: MissingReason | str = MissingReason.MISSING_3D_RECONSTRUCTION,
        metadata: Mapping[str, Any] | None = None,
    ) -> "Object3DObservation":
        """Create an invalid object without fabricated zero-valued 3D points."""

        reason_value = reason.value if isinstance(reason, MissingReason) else str(reason)
        return cls(
            video_id=video_id,
            frame_index=frame_index,
            track_id=track_id,
            semantic_label=semantic_label,
            canonical_label=canonical_label or semantic_label,
            center_3d=None,
            boundary_points_3d=(),
            keypoints_3d=(),
            structure_points_3d=(),
            observed_scale_3d=None,
            normalized_structure_points=(),
            scale_status=GeometryScaleStatus.UNKNOWN,
            visibility=VisibilityStatus.UNKNOWN,
            reconstruction_quality=0.0,
            valid=False,
            missing_reason=reason_value,
            source_object_2d_id=source_object_2d_id,
            metadata=dict(metadata or {}),
        )

    def require_observed_scale(self, metric: bool = False) -> float:
        """Return explicit scale without deriving it from normalized points."""

        if self.observed_scale_3d is None:
            raise ValueError("No observed_scale_3d is available for this object.")
        required = GeometryScaleStatus.METRIC_3D if metric else self.scale_status
        if metric and self.scale_status != required:
            raise ValueError("Metric scale requested from a non-metric 3D observation.")
        return float(self.observed_scale_3d)

    def require_cross_frame_scale_comparable(self) -> float:
        """Return scale only when its calibration domain supports frame comparison."""

        scale = self.require_observed_scale()
        if self.depth_scale_status not in {
            DepthScaleStatus.METRIC_CALIBRATED,
            DepthScaleStatus.RELATIVE_SHARED_SEQUENCE,
        }:
            raise ValueError(
                "Cross-frame scale comparison requires metric_calibrated or "
                "relative_shared_sequence depth; relative_per_frame is not comparable."
            )
        return scale


@dataclass(frozen=True)
class Shared3DFrameObservation:
    """One frame shared by future static and dynamic 3D residual branches."""

    video_id: str
    frame_index: int
    image_width: int
    image_height: int
    camera: CameraObservation
    depth: DepthObservation
    objects: tuple[Object3DObservation, ...]
    valid: bool
    quality: float
    missing_reason: str = ""
    source_frame_id: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("Shared3D frame dimensions must be positive.")
        quality = float(self.quality)
        if not math.isfinite(quality) or not 0.0 <= quality <= 1.0:
            raise ValueError("Shared3DFrameObservation.quality must be in [0, 1].")
        if self.valid:
            if not self.camera.valid:
                raise ValueError("Valid Shared3DFrameObservation requires valid camera data.")
            self.depth.require_geometry_depth()
            if (
                any(obj.scale_status == GeometryScaleStatus.METRIC_3D for obj in self.objects)
                and self.depth.scale_status.value != "metric_calibrated"
            ):
                raise ValueError("Metric Object3D observations require metric-calibrated depth.")
            if self.missing_reason:
                raise ValueError("Valid Shared3DFrameObservation cannot have missing_reason.")
        elif not self.missing_reason:
            raise ValueError("Invalid Shared3DFrameObservation requires missing_reason.")
        object.__setattr__(self, "objects", tuple(self.objects))
        object.__setattr__(self, "quality", quality)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def missing(
        cls,
        video_id: str,
        frame_index: int,
        image_width: int,
        image_height: int,
        camera: CameraObservation,
        depth: DepthObservation,
        objects: Sequence[Object3DObservation] = (),
        reason: MissingReason | str = MissingReason.MISSING_3D_RECONSTRUCTION,
        source_frame_id: Optional[str] = None,
    ) -> "Shared3DFrameObservation":
        """Create a frame-level invalid record while preserving available evidence."""

        reason_value = reason.value if isinstance(reason, MissingReason) else str(reason)
        return cls(
            video_id=video_id,
            frame_index=frame_index,
            image_width=image_width,
            image_height=image_height,
            camera=camera,
            depth=depth,
            objects=tuple(objects),
            valid=False,
            quality=0.0,
            missing_reason=reason_value,
            source_frame_id=source_frame_id,
        )
