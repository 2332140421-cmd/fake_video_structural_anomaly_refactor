"""Sparse object-centric 3D reconstruction from shared canonical evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Optional, Sequence

import numpy as np

from ..depth_provider import (
    DepthObservation,
    DepthRepresentation,
    DepthScaleStatus,
)
from ..geometry.backprojection import backproject_pixel
from ..geometry.camera import CameraObservation, CoordinateConvention, DepthDefinition
from ..geometry.transforms import camera_to_world
from ..observations import ObjectObservationJSON
from ..shared_3d_observation import (
    GeometryScaleStatus,
    GeometryScaleUnit,
    Object3DObservation,
    Point2DObservation,
    Point3DObservation,
    ReconstructionFrame,
    VisibilityStatus,
)
from .depth_sampling import DepthSamplingMethod, sample_depth


CenterMethod = Literal["coordinate_median", "geometric_median"]


@dataclass(frozen=True)
class Object3DReconstructorConfig:
    """Configuration for robust sparse object reconstruction."""

    depth_sampling_method: DepthSamplingMethod | str = DepthSamplingMethod.LOCAL_MEDIAN_3X3
    center_method: CenterMethod = "coordinate_median"
    minimum_valid_points: int = 2
    normalization_epsilon: float = 1e-8
    bbox_boundary_samples_per_edge: int = 3

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "depth_sampling_method", DepthSamplingMethod(self.depth_sampling_method)
        )
        if self.center_method not in {"coordinate_median", "geometric_median"}:
            raise ValueError("center_method must be coordinate_median or geometric_median.")
        if self.minimum_valid_points < 2:
            raise ValueError("minimum_valid_points must be at least 2.")
        if self.normalization_epsilon <= 0.0:
            raise ValueError("normalization_epsilon must be positive.")
        if self.bbox_boundary_samples_per_edge < 2:
            raise ValueError("bbox_boundary_samples_per_edge must be at least 2.")


def sample_bbox_boundary_points(
    obj: ObjectObservationJSON, samples_per_edge: int = 3
) -> tuple[Point2DObservation, ...]:
    """Create deterministic boundary samples from an object's xyxy bbox."""

    if obj.bbox is None or len(obj.bbox) != 4:
        return ()
    x1, y1, x2, y2 = (float(value) for value in obj.bbox)
    if not np.isfinite([x1, y1, x2, y2]).all() or x2 <= x1 or y2 <= y1:
        return ()
    xs = np.linspace(x1, x2, samples_per_edge)
    ys = np.linspace(y1, y2, samples_per_edge)
    coordinates: list[tuple[float, float]] = []
    coordinates.extend((float(x), y1) for x in xs)
    coordinates.extend((float(x), y2) for x in xs)
    coordinates.extend((x1, float(y)) for y in ys[1:-1])
    coordinates.extend((x2, float(y)) for y in ys[1:-1])
    return tuple(
        Point2DObservation(
            point_id=f"{obj.object_id}:boundary:{index}",
            x=x,
            y=y,
            confidence=float(np.clip(obj.confidence, 0.0, 1.0)),
            valid=True,
            source="bbox_boundary",
            metadata={"source_object_id": obj.object_id},
        )
        for index, (x, y) in enumerate(coordinates)
    )


def _geometric_median(points: np.ndarray, tolerance: float = 1e-8) -> np.ndarray:
    """Compute a geometric median with Weiszfeld iterations."""

    estimate = np.median(points, axis=0)
    for _ in range(128):
        distances = np.linalg.norm(points - estimate, axis=1)
        if np.any(distances < tolerance):
            return points[int(np.argmin(distances))]
        weights = 1.0 / np.maximum(distances, tolerance)
        updated = np.sum(points * weights[:, None], axis=0) / np.sum(weights)
        if np.linalg.norm(updated - estimate) <= tolerance:
            return updated
        estimate = updated
    return estimate


def _scale_descriptors(points: np.ndarray) -> dict[str, object]:
    """Compute axis-aligned, diagonal, PCA, and equivalent scale descriptors."""

    minima = np.min(points, axis=0)
    maxima = np.max(points, axis=0)
    extents = maxima - minima
    diagonal = float(np.linalg.norm(extents))
    descriptor: dict[str, object] = {
        "axis_aligned_extent": {
            "extent_x": float(extents[0]),
            "extent_y": float(extents[1]),
            "extent_z": float(extents[2]),
        },
        "bounding_box_diagonal": diagonal,
    }
    positive = extents[extents > 1e-12]
    descriptor["equivalent_linear_scale"] = (
        float(np.exp(np.mean(np.log(positive)))) if positive.size else None
    )
    if points.shape[0] >= 3:
        centered = points - np.mean(points, axis=0)
        _, _, axes = np.linalg.svd(centered, full_matrices=False)
        projected = centered @ axes.T
        pca_extents = np.max(projected, axis=0) - np.min(projected, axis=0)
        padded = np.zeros(3, dtype=float)
        padded[: min(3, pca_extents.size)] = pca_extents[:3]
        descriptor["pca_principal_extents"] = [float(value) for value in padded]
    else:
        descriptor["pca_principal_extents"] = None
    return descriptor


def _visibility(obj: ObjectObservationJSON, camera: CameraObservation) -> VisibilityStatus:
    """Infer only boundary truncation, not semantic occlusion."""

    if obj.bbox is None or len(obj.bbox) != 4:
        return VisibilityStatus.UNKNOWN
    x1, y1, x2, y2 = (float(value) for value in obj.bbox)
    margin = 1.0
    if (
        x1 <= margin
        or y1 <= margin
        or x2 >= camera.image_width - 1 - margin
        or y2 >= camera.image_height - 1 - margin
    ):
        return VisibilityStatus.PARTIALLY_VISIBLE
    return VisibilityStatus.VISIBLE


class Object3DReconstructor:
    """Recover a sparse camera-frame object structure from 2D points and Z depth."""

    def __init__(self, config: Optional[Object3DReconstructorConfig] = None) -> None:
        self.config = config or Object3DReconstructorConfig()

    @staticmethod
    def _scale_domain(
        depth: DepthObservation,
    ) -> tuple[GeometryScaleStatus, GeometryScaleUnit]:
        if (
            depth.depth_representation == DepthRepresentation.METRIC_DEPTH
            and depth.scale_status == DepthScaleStatus.METRIC_CALIBRATED
        ):
            return GeometryScaleStatus.METRIC_3D, GeometryScaleUnit.METER
        return GeometryScaleStatus.RELATIVE_3D, GeometryScaleUnit.RELATIVE_UNIT

    def _backproject_group(
        self,
        points_2d: Sequence[Point2DObservation],
        depth: DepthObservation,
        camera: CameraObservation,
        scale_status: GeometryScaleStatus,
    ) -> tuple[Point3DObservation, ...]:
        output: list[Point3DObservation] = []
        assert camera.K is not None
        for point in points_2d:
            if not point.valid or point.x is None or point.y is None:
                output.append(
                    backproject_pixel(
                        0.0,
                        0.0,
                        -1.0,
                        camera.K,
                        point_id=point.point_id,
                        source_point_2d_id=point.point_id,
                        valid=False,
                        missing_reason=point.missing_reason or "invalid_2d_point",
                    )
                )
                continue
            sampled = sample_depth(
                depth,
                point.x,
                point.y,
                method=self.config.depth_sampling_method,
            )
            sample_metadata = {
                "source_point_xy": [float(point.x), float(point.y)],
                "source_pixel": list(sampled.source_pixel),
                "sampled_depth": sampled.sampled_depth,
                "sampling_method": sampled.sampling_method.value,
                "local_valid_ratio": sampled.local_valid_ratio,
                "local_depth_iqr": sampled.local_depth_iqr,
                "point_quality": sampled.point_quality,
                "source_2d": point.source,
            }
            if not sampled.valid or sampled.sampled_depth is None:
                output.append(
                    backproject_pixel(
                        point.x,
                        point.y,
                        -1.0,
                        camera.K,
                        point_id=point.point_id,
                        source_point_2d_id=point.point_id,
                        valid=False,
                        missing_reason=sampled.missing_reason,
                        metadata=sample_metadata,
                    )
                )
                continue
            output.append(
                backproject_pixel(
                    point.x,
                    point.y,
                    sampled.sampled_depth,
                    camera.K,
                    point_id=point.point_id,
                    confidence=float(point.confidence * sampled.point_quality),
                    scale_status=scale_status,
                    source_point_2d_id=point.point_id,
                    metadata=sample_metadata,
                )
            )
        return tuple(output)

    def reconstruct(
        self,
        *,
        video_id: str,
        frame_index: Optional[int] = None,
        obj: ObjectObservationJSON,
        depth: DepthObservation,
        camera: CameraObservation,
        keypoints_2d: Sequence[Point2DObservation] = (),
        boundary_points_2d: Optional[Sequence[Point2DObservation]] = None,
    ) -> Object3DObservation:
        """Reconstruct one object without semantic physical-size priors."""

        resolved_frame_index = int(
            frame_index
            if frame_index is not None
            else (depth.frame_index if depth.frame_index is not None else 0)
        )
        base_metadata: dict[str, object] = {
            "calibration_source": camera.intrinsics_source or "none",
            "anchor_object_ids": [],
            "excluded_from_evaluation_ids": [],
            "metric_scale_source": depth.metadata.get("metric_scale_source", "none"),
            "depth_provider": depth.provider_name,
            "depth_scale_status": depth.scale_status.value,
            "physical_scale_prior_used": False,
            "source_bbox": list(obj.bbox) if obj.bbox is not None else None,
            "source_mask_path": obj.mask_path,
        }
        if not camera.valid or camera.K is None:
            return Object3DObservation.missing(
                video_id,
                resolved_frame_index,
                obj.label,
                obj.object_id,
                track_id=obj.track_id or obj.person_track_id,
                canonical_label=obj.canonical_label,
                reason="invalid_camera_intrinsics",
                metadata=base_metadata,
            )
        if camera.coordinate_convention != CoordinateConvention.OPENCV:
            return Object3DObservation.missing(
                video_id,
                resolved_frame_index,
                obj.label,
                obj.object_id,
                reason="incompatible_coordinate_convention",
                metadata=base_metadata,
            )
        if camera.depth_definition != DepthDefinition.Z_DEPTH:
            return Object3DObservation.missing(
                video_id,
                resolved_frame_index,
                obj.label,
                obj.object_id,
                reason="incompatible_depth_definition",
                metadata=base_metadata,
            )
        try:
            depth.require_geometry_depth()
        except ValueError as error:
            return Object3DObservation.missing(
                video_id,
                resolved_frame_index,
                obj.label,
                obj.object_id,
                reason="invalid_geometry_depth",
                metadata={**base_metadata, "depth_error": str(error)},
            )

        boundary = (
            tuple(boundary_points_2d)
            if boundary_points_2d is not None
            else sample_bbox_boundary_points(
                obj, self.config.bbox_boundary_samples_per_edge
            )
        )
        keypoints = tuple(keypoints_2d)
        requested_count = len(boundary) + len(keypoints)
        if requested_count == 0:
            return Object3DObservation.missing(
                video_id,
                resolved_frame_index,
                obj.label,
                obj.object_id,
                reason="no_2d_structure_points",
                metadata=base_metadata,
            )
        scale_status, scale_unit = self._scale_domain(depth)
        boundary_camera = self._backproject_group(
            boundary, depth, camera, scale_status
        )
        keypoints_camera = self._backproject_group(
            keypoints, depth, camera, scale_status
        )
        structure_camera = boundary_camera + keypoints_camera
        valid_points = tuple(point for point in structure_camera if point.valid)
        valid_ratio = len(valid_points) / requested_count
        base_metadata["valid_point_ratio"] = valid_ratio
        base_metadata["requested_point_count"] = requested_count
        base_metadata["valid_point_count"] = len(valid_points)
        if len(valid_points) < self.config.minimum_valid_points:
            return Object3DObservation.missing(
                video_id,
                resolved_frame_index,
                obj.label,
                obj.object_id,
                track_id=obj.track_id or obj.person_track_id,
                canonical_label=obj.canonical_label,
                reason="insufficient_valid_3d_points",
                metadata=base_metadata,
            )

        xyz = np.stack([point.as_array() for point in valid_points])
        if self.config.center_method == "geometric_median":
            center_xyz = _geometric_median(xyz)
        else:
            center_xyz = np.median(xyz, axis=0)
        descriptors = _scale_descriptors(xyz)
        observed_scale = float(descriptors["bounding_box_diagonal"])
        if not np.isfinite(observed_scale) or observed_scale <= self.config.normalization_epsilon:
            return Object3DObservation.missing(
                video_id,
                resolved_frame_index,
                obj.label,
                obj.object_id,
                reason="degenerate_observed_scale",
                metadata=base_metadata,
            )

        point_quality = float(np.mean([point.confidence for point in valid_points]))
        center_quality = float(np.clip(point_quality * valid_ratio, 0.0, 1.0))
        center_camera = Point3DObservation(
            point_id=f"{obj.object_id}:center",
            x=float(center_xyz[0]),
            y=float(center_xyz[1]),
            z=float(center_xyz[2]),
            coordinate_frame="camera",
            scale_status=scale_status,
            confidence=center_quality,
            valid=True,
            metadata={"center_method": self.config.center_method},
        )
        normalized = tuple(
            Point3DObservation(
                point_id=f"{point.point_id}:normalized",
                x=float((point.x - center_xyz[0]) / observed_scale),  # type: ignore[operator]
                y=float((point.y - center_xyz[1]) / observed_scale),  # type: ignore[operator]
                z=float((point.z - center_xyz[2]) / observed_scale),  # type: ignore[operator]
                coordinate_frame="object_normalized",
                scale_status=GeometryScaleStatus.NORMALIZED_SHAPE,
                confidence=point.confidence,
                valid=True,
                source_point_2d_id=point.source_point_2d_id,
                metadata={
                    "normalization_method": "center_then_divide_by_bounding_box_diagonal",
                    "source_point_3d_id": point.point_id,
                    "unit": "unitless",
                },
            )
            for point in valid_points
        )

        boundary_world: Optional[tuple[Point3DObservation, ...]] = None
        keypoints_world: Optional[tuple[Point3DObservation, ...]] = None
        structure_world: Optional[tuple[Point3DObservation, ...]] = None
        center_world: Optional[Point3DObservation] = None
        if camera.pose_valid:
            boundary_world = tuple(
                point for point in camera_to_world(boundary_camera, camera) if point.valid
            )
            keypoints_world = tuple(
                point for point in camera_to_world(keypoints_camera, camera) if point.valid
            )
            structure_world = tuple(
                point for point in camera_to_world(structure_camera, camera) if point.valid
            )
            transformed_center = camera_to_world((center_camera,), camera)[0]
            center_world = transformed_center if transformed_center.valid else None
        else:
            base_metadata["world_reconstruction_missing_reason"] = "no_camera_pose"

        object_quality = float(
            np.mean(
                [
                    center_quality,
                    float(np.clip(obj.confidence, 0.0, 1.0)),
                    depth.quality,
                    camera.quality,
                ]
            )
        )
        return Object3DObservation(
            video_id=video_id,
            frame_index=resolved_frame_index,
            track_id=obj.track_id or obj.person_track_id,
            semantic_label=obj.label,
            canonical_label=obj.canonical_label or obj.label,
            center_3d=center_camera,
            boundary_points_3d=boundary_camera,
            keypoints_3d=keypoints_camera,
            structure_points_3d=structure_camera,
            observed_scale_3d=observed_scale,
            normalized_structure_points=normalized,
            scale_status=scale_status,
            visibility=_visibility(obj, camera),
            reconstruction_quality=object_quality,
            valid=True,
            missing_reason="",
            source_object_2d_id=obj.object_id,
            metadata={
                **base_metadata,
                "center_method": self.config.center_method,
                "normalization_method": "center_then_divide_by_bounding_box_diagonal",
                "provenance": dict(obj.provenance),
            },
            center_3d_camera=center_camera,
            center_3d_world=center_world,
            boundary_points_3d_world=boundary_world,
            keypoints_3d_world=keypoints_world,
            structure_points_3d_world=structure_world,
            scale_method="bounding_box_diagonal",
            scale_unit=scale_unit,
            reconstruction_frame=ReconstructionFrame.CAMERA,
            scale_quality=center_quality,
            scale_descriptors=descriptors,
            depth_scale_status=depth.scale_status,
        )
