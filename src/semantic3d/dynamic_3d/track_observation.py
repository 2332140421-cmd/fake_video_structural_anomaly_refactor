"""Independent 2D point tracks and shared-geometry 3D track observations."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

from ..geometry.backprojection import backproject_pixel
from ..sequence_geometry import SequenceScaleStatus, Shared3DClipObservation
from ..shared_3d_observation import GeometryScaleStatus, VisibilityStatus
from .readiness import Dynamic3DReadiness, DynamicGeometryMode


def _optional_tuple(
    value: Optional[Sequence[float]], size: int, name: str
) -> Optional[tuple[float, ...]]:
    if value is None:
        return None
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain {size} finite values.")
    return tuple(float(item) for item in array)


@dataclass(frozen=True)
class PointTrack2DObservation:
    """One independently tracked image point at one global frame index."""

    point_id: str
    object_track_id: str
    frame_index: int
    pixel_uv: Optional[tuple[float, float]]
    visibility: VisibilityStatus | str
    occlusion_status: str
    tracking_confidence: float
    source_tracker: str
    valid: bool
    missing_reason: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        uv = _optional_tuple(self.pixel_uv, 2, "pixel_uv")
        visibility = VisibilityStatus(self.visibility)
        confidence = float(self.tracking_confidence)
        if not self.point_id.strip() or not self.object_track_id.strip():
            raise ValueError("Point and object track IDs must not be empty.")
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("tracking_confidence must be in [0, 1].")
        if self.valid:
            if uv is None or not self.source_tracker.strip():
                raise ValueError("Valid 2D track point requires pixel_uv and source_tracker.")
            if self.missing_reason:
                raise ValueError("Valid 2D track point cannot have missing_reason.")
            if not bool(self.metadata.get("independent_observation", False)):
                raise ValueError(
                    "Valid current-frame track points must be independent observations."
                )
            if bool(self.metadata.get("generated_from_projection", False)):
                raise ValueError("Projected points cannot masquerade as 2D observations.")
        else:
            if uv is not None or not self.missing_reason:
                raise ValueError("Invalid 2D track point requires no coordinates and a reason.")
        object.__setattr__(self, "pixel_uv", uv)
        object.__setattr__(self, "visibility", visibility)
        object.__setattr__(self, "tracking_confidence", confidence)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def missing(
        cls,
        *,
        point_id: str,
        object_track_id: str,
        frame_index: int,
        reason: str,
        source_tracker: str,
    ) -> "PointTrack2DObservation":
        """Create a missing 2D sample without zero pixel coordinates."""

        return cls(
            point_id=point_id,
            object_track_id=object_track_id,
            frame_index=frame_index,
            pixel_uv=None,
            visibility=VisibilityStatus.UNKNOWN,
            occlusion_status="unknown",
            tracking_confidence=0.0,
            source_tracker=source_tracker,
            valid=False,
            missing_reason=reason,
        )


@dataclass(frozen=True)
class PointTrack3DObservation:
    """One tracked point reconstructed from shared depth and camera evidence."""

    point_id: str
    object_track_id: str
    frame_index: int
    pixel_uv: Optional[tuple[float, float]]
    observed_depth: Optional[float]
    point_3d_camera: Optional[tuple[float, float, float]]
    point_3d_world: Optional[tuple[float, float, float]]
    visibility: VisibilityStatus | str
    occlusion_status: str
    tracking_confidence: float
    depth_quality: float
    reconstruction_quality: float
    source_tracker: str
    scale_status: SequenceScaleStatus | str
    geometry_mode: DynamicGeometryMode | str
    valid: bool
    missing_reason: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        uv = _optional_tuple(self.pixel_uv, 2, "pixel_uv")
        camera_point = _optional_tuple(self.point_3d_camera, 3, "point_3d_camera")
        world_point = _optional_tuple(self.point_3d_world, 3, "point_3d_world")
        scale_status = SequenceScaleStatus(self.scale_status)
        mode = DynamicGeometryMode(self.geometry_mode)
        visibility = VisibilityStatus(self.visibility)
        depth = None if self.observed_depth is None else float(self.observed_depth)
        for name in ("tracking_confidence", "depth_quality", "reconstruction_quality"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1].")
            object.__setattr__(self, name, value)
        if self.valid:
            if uv is None or camera_point is None or depth is None or depth <= 0.0:
                raise ValueError("Valid 3D track point requires pixel, depth, and camera point.")
            if not math.isfinite(depth) or self.missing_reason:
                raise ValueError("Valid 3D track point has invalid depth or missing reason.")
            if scale_status == SequenceScaleStatus.RELATIVE_PER_FRAME:
                raise ValueError("relative_per_frame cannot form a valid 3D trajectory.")
            if world_point is not None and not mode.allows_world_3d:
                raise ValueError("World 3D points require full_se3_3d mode.")
        else:
            if any(value is not None for value in (depth, camera_point, world_point)):
                raise ValueError("Invalid 3D track point must not contain fabricated geometry.")
            if not self.missing_reason:
                raise ValueError("Invalid 3D track point requires missing_reason.")
        object.__setattr__(self, "pixel_uv", uv)
        object.__setattr__(self, "observed_depth", depth)
        object.__setattr__(self, "point_3d_camera", camera_point)
        object.__setattr__(self, "point_3d_world", world_point)
        object.__setattr__(self, "visibility", visibility)
        object.__setattr__(self, "scale_status", scale_status)
        object.__setattr__(self, "geometry_mode", mode)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def missing(
        cls,
        point_2d: PointTrack2DObservation,
        *,
        reason: str,
        scale_status: SequenceScaleStatus | str,
        geometry_mode: DynamicGeometryMode | str,
    ) -> "PointTrack3DObservation":
        """Create missing 3D evidence while retaining 2D provenance in metadata."""

        return cls(
            point_id=point_2d.point_id,
            object_track_id=point_2d.object_track_id,
            frame_index=point_2d.frame_index,
            pixel_uv=point_2d.pixel_uv,
            observed_depth=None,
            point_3d_camera=None,
            point_3d_world=None,
            visibility=point_2d.visibility,
            occlusion_status=point_2d.occlusion_status,
            tracking_confidence=point_2d.tracking_confidence,
            depth_quality=0.0,
            reconstruction_quality=0.0,
            source_tracker=point_2d.source_tracker,
            scale_status=scale_status,
            geometry_mode=geometry_mode,
            valid=False,
            missing_reason=reason,
            metadata={
                "source_2d_valid": point_2d.valid,
                "independent_observation": bool(
                    point_2d.metadata.get("independent_observation", False)
                ),
            },
        )


@dataclass(frozen=True)
class ObjectTrack3DObservation:
    """All stable point IDs associated with one object or scene track."""

    object_track_id: str
    semantic_label: str
    points: tuple[PointTrack3DObservation, ...]
    observed_scale_3d_by_frame: Mapping[int, Optional[float]]
    scale_status: SequenceScaleStatus | str
    valid: bool
    quality: float
    missing_reason: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        points = tuple(self.points)
        scale_status = SequenceScaleStatus(self.scale_status)
        quality = float(self.quality)
        if not math.isfinite(quality) or not 0.0 <= quality <= 1.0:
            raise ValueError("Object track quality must be in [0, 1].")
        if any(point.object_track_id != self.object_track_id for point in points):
            raise ValueError("All points must belong to object_track_id.")
        keys = [(point.point_id, point.frame_index) for point in points]
        if len(keys) != len(set(keys)):
            raise ValueError("A point_id may appear at most once in one frame.")
        if self.valid and (not any(point.valid for point in points) or self.missing_reason):
            raise ValueError("Valid object track requires valid point evidence.")
        if not self.valid and not self.missing_reason:
            raise ValueError("Invalid object track requires missing_reason.")
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "observed_scale_3d_by_frame", dict(self.observed_scale_3d_by_frame))
        object.__setattr__(self, "scale_status", scale_status)
        object.__setattr__(self, "quality", quality)
        object.__setattr__(self, "metadata", dict(self.metadata))


class BasePointTracker(ABC):
    """Canonical independent image-point tracker interface."""

    @abstractmethod
    def track(
        self,
        images: Mapping[int, np.ndarray],
        *,
        object_track_id: str,
        initial_points: Optional[np.ndarray] = None,
    ) -> tuple[PointTrack2DObservation, ...]:
        """Return stable point IDs whose current pixels come from image observations."""


class SyntheticPointTracker(BasePointTracker):
    """Return supplied ground-truth image tracks for deterministic tests."""

    def __init__(self, trajectories: Mapping[str, Mapping[int, Sequence[float]]]) -> None:
        self.trajectories = {
            str(point_id): {int(frame): tuple(uv) for frame, uv in samples.items()}
            for point_id, samples in trajectories.items()
        }

    def track(
        self,
        images: Mapping[int, np.ndarray],
        *,
        object_track_id: str,
        initial_points: Optional[np.ndarray] = None,
    ) -> tuple[PointTrack2DObservation, ...]:
        del initial_points
        frame_indices = tuple(sorted(images))
        return tuple(
            PointTrack2DObservation(
                point_id=point_id,
                object_track_id=object_track_id,
                frame_index=frame_index,
                pixel_uv=self.trajectories[point_id][frame_index],
                visibility=VisibilityStatus.VISIBLE,
                occlusion_status="visible",
                tracking_confidence=1.0,
                source_tracker="synthetic_ground_truth",
                valid=True,
                metadata={
                    "independent_observation": True,
                    "generated_from_projection": False,
                    "ground_truth": True,
                },
            )
            for point_id in sorted(self.trajectories)
            for frame_index in frame_indices
            if frame_index in self.trajectories[point_id]
        )


class MockPointTracker(BasePointTracker):
    """Deterministic image-independent tracker for contract tests only."""

    def __init__(self, displacement_uv: tuple[float, float] = (1.0, 0.0)) -> None:
        self.displacement_uv = tuple(float(value) for value in displacement_uv)

    def track(
        self,
        images: Mapping[int, np.ndarray],
        *,
        object_track_id: str,
        initial_points: Optional[np.ndarray] = None,
    ) -> tuple[PointTrack2DObservation, ...]:
        indices = tuple(sorted(images))
        if not indices:
            return ()
        points = (
            np.asarray(initial_points, dtype=float).reshape(-1, 2)
            if initial_points is not None
            else np.asarray([[10.0, 10.0]], dtype=float)
        )
        output = []
        for point_index, initial in enumerate(points):
            for offset, frame_index in enumerate(indices):
                uv = initial + offset * np.asarray(self.displacement_uv)
                output.append(
                    PointTrack2DObservation(
                        point_id=f"mock_point_{point_index}",
                        object_track_id=object_track_id,
                        frame_index=frame_index,
                        pixel_uv=(float(uv[0]), float(uv[1])),
                        visibility=VisibilityStatus.VISIBLE,
                        occlusion_status="visible",
                        tracking_confidence=1.0,
                        source_tracker="mock_point_tracker",
                        valid=True,
                        metadata={
                            "independent_observation": True,
                            "generated_from_projection": False,
                            "mock": True,
                        },
                    )
                )
        return tuple(output)


class ExistingInterfaceAdapter(BasePointTracker):
    """Adapt an existing tracker callable to the canonical observation contract."""

    def __init__(
        self,
        callback: Callable[
            [Mapping[int, np.ndarray], str, Optional[np.ndarray]],
            Sequence[PointTrack2DObservation | Mapping[str, Any]],
        ],
        *,
        provider_name: str,
    ) -> None:
        self.callback = callback
        self.provider_name = provider_name

    def track(
        self,
        images: Mapping[int, np.ndarray],
        *,
        object_track_id: str,
        initial_points: Optional[np.ndarray] = None,
    ) -> tuple[PointTrack2DObservation, ...]:
        rows = self.callback(images, object_track_id, initial_points)
        output = []
        for row in rows:
            if isinstance(row, PointTrack2DObservation):
                output.append(row)
                continue
            payload = dict(row)
            payload.setdefault("source_tracker", self.provider_name)
            payload.setdefault("metadata", {})
            payload["metadata"] = {
                **dict(payload["metadata"]),
                "independent_observation": True,
                "generated_from_projection": False,
                "adapter": "existing_interface",
            }
            output.append(PointTrack2DObservation(**payload))
        return tuple(output)


def reconstruct_point_tracks_3d(
    points_2d: Sequence[PointTrack2DObservation],
    shared_clip: Shared3DClipObservation,
    readiness: Dynamic3DReadiness,
) -> tuple[PointTrack3DObservation, ...]:
    """Back-project independent tracks using the exact supplied shared clip."""

    frames = {frame.frame_index: frame for frame in shared_clip.frames}
    output: list[PointTrack3DObservation] = []
    for point in points_2d:
        common = {
            "scale_status": shared_clip.sequence_scale_status,
            "geometry_mode": readiness.mode,
        }
        if not readiness.dynamic_3d_ready:
            output.append(
                PointTrack3DObservation.missing(
                    point, reason=readiness.missing_reason, **common
                )
            )
            continue
        if not point.valid or point.pixel_uv is None:
            output.append(
                PointTrack3DObservation.missing(
                    point, reason=point.missing_reason or "missing_2d_track_point", **common
                )
            )
            continue
        frame = frames.get(point.frame_index)
        if frame is None or not frame.valid or frame.camera.K is None:
            output.append(
                PointTrack3DObservation.missing(
                    point, reason="missing_shared_3d_frame", **common
                )
            )
            continue
        depth_map = frame.depth.require_geometry_depth()
        u, v = point.pixel_uv
        column, row = int(round(u)), int(round(v))
        if not (0 <= row < depth_map.shape[0] and 0 <= column < depth_map.shape[1]):
            output.append(
                PointTrack3DObservation.missing(
                    point, reason="tracked_point_out_of_frame", **common
                )
            )
            continue
        valid_mask = np.asarray(frame.depth.valid_mask, dtype=bool)
        depth = float(depth_map[row, column])
        if not valid_mask[row, column] or not math.isfinite(depth) or depth <= 0.0:
            output.append(
                PointTrack3DObservation.missing(
                    point, reason="invalid_depth_at_tracked_point", **common
                )
            )
            continue
        point_scale = (
            GeometryScaleStatus.METRIC_3D
            if shared_clip.sequence_scale_status == SequenceScaleStatus.METRIC_SEQUENCE
            else GeometryScaleStatus.RELATIVE_3D
        )
        camera_point = backproject_pixel(
            u,
            v,
            depth,
            frame.camera.K,
            point_id=point.point_id,
            confidence=point.tracking_confidence,
            scale_status=point_scale,
            source_point_2d_id=point.point_id,
            metadata={"independent_observation": True},
        )
        if not camera_point.valid:
            output.append(
                PointTrack3DObservation.missing(
                    point, reason=camera_point.missing_reason, **common
                )
            )
            continue
        world_point = None
        if readiness.allows_world_3d:
            transform = shared_clip.T_world_from_camera_by_frame.get(point.frame_index)
            if transform is None:
                output.append(
                    PointTrack3DObservation.missing(
                        point, reason="missing_world_camera_transform", **common
                    )
                )
                continue
            homogeneous = np.concatenate([camera_point.as_array(), [1.0]])
            transformed = np.asarray(transform, dtype=float) @ homogeneous
            world_point = tuple(float(value) for value in transformed[:3] / transformed[3])
        quality = float(
            min(point.tracking_confidence, frame.depth.quality, frame.camera.quality)
        )
        output.append(
            PointTrack3DObservation(
                point_id=point.point_id,
                object_track_id=point.object_track_id,
                frame_index=point.frame_index,
                pixel_uv=point.pixel_uv,
                observed_depth=depth,
                point_3d_camera=tuple(camera_point.as_array()),
                point_3d_world=world_point,
                visibility=point.visibility,
                occlusion_status=point.occlusion_status,
                tracking_confidence=point.tracking_confidence,
                depth_quality=frame.depth.quality,
                reconstruction_quality=quality,
                source_tracker=point.source_tracker,
                scale_status=shared_clip.sequence_scale_status,
                geometry_mode=readiness.mode,
                valid=True,
                metadata={
                    "independent_observation": True,
                    "shared_clip_reused": True,
                    "depth_reestimated": False,
                    "intrinsics_reestimated": False,
                    "pose_reestimated": False,
                },
            )
        )
    return tuple(output)


def summarize_point_track_coverage(
    points: Sequence[PointTrack2DObservation], frame_count: int
) -> tuple[float, float, int]:
    """Return coverage, mean valid track length, and independent track count."""

    if frame_count <= 0:
        raise ValueError("frame_count must be positive.")
    lengths: dict[str, int] = {}
    for point in points:
        if point.valid:
            lengths[point.point_id] = lengths.get(point.point_id, 0) + 1
    if not lengths:
        return 0.0, 0.0, 0
    mean_length = float(np.mean(list(lengths.values())))
    coverage = float(np.mean([length / frame_count for length in lengths.values()]))
    return coverage, mean_length, len(lengths)
