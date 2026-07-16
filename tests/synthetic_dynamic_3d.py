"""Synthetic truth utilities for dynamic 3D trajectory tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from semantic3d.depth_provider import (
    DepthObservation,
    DepthRepresentation,
    DepthScaleStatus,
    LargerValueMeans,
)
from semantic3d.dynamic_3d import (
    Dynamic3DReadiness,
    Dynamic3DReadinessThresholds,
    DynamicGeometryMode,
    PointTrack2DObservation,
    PointTrack3DObservation,
    SyntheticPointTracker,
    assess_dynamic_3d_readiness,
    reconstruct_point_tracks_3d,
)
from semantic3d.geometry.camera import CameraObservation
from semantic3d.sequence_geometry import (
    RelativePoseObservation,
    SequenceScaleStatus,
    Shared3DClipObservation,
)
from semantic3d.shared_3d_observation import Shared3DFrameObservation


@dataclass(frozen=True)
class SyntheticDynamicScene:
    """Shared clip and independently observed trajectories with known truth."""

    clip: Shared3DClipObservation
    readiness: Dynamic3DReadiness
    images: Mapping[int, np.ndarray]
    points_2d: tuple[PointTrack2DObservation, ...]
    points_3d: tuple[PointTrack3DObservation, ...]
    K: np.ndarray
    world_points: Mapping[str, Mapping[int, np.ndarray]]


def _transform_from_center(center: Sequence[float]) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[:3, 3] = np.asarray(center, dtype=float)
    return transform


def _project(world: np.ndarray, camera_center: np.ndarray, K: np.ndarray) -> np.ndarray:
    camera = world - camera_center
    pixel = K @ camera
    return pixel[:2] / pixel[2]


def make_synthetic_dynamic_scene(
    *,
    camera_centers: Sequence[Sequence[float]],
    world_points: Mapping[str, Sequence[Sequence[float]]],
    mode: DynamicGeometryMode = DynamicGeometryMode.FULL_SE3_3D,
    scale_status: SequenceScaleStatus = SequenceScaleStatus.RELATIVE_ALIGNED_SEQUENCE,
    scene_cut_frame: int | None = None,
    break_point_id_at: int | None = None,
) -> SyntheticDynamicScene:
    """Build a calibrated synthetic clip without invoking any estimator."""

    frame_count = len(camera_centers)
    if frame_count < 1 or any(len(samples) != frame_count for samples in world_points.values()):
        raise ValueError("Every truth trajectory must span all synthetic frames.")
    width, height = 160, 120
    K = np.asarray([[120.0, 0.0, 79.5], [0.0, 120.0, 59.5], [0.0, 0.0, 1.0]])
    images = {index: np.zeros((height, width, 3), dtype=np.uint8) for index in range(frame_count)}
    trajectories: dict[str, dict[int, tuple[float, float]]] = {}
    depth_maps = {index: np.full((height, width), 8.0, dtype=np.float32) for index in range(frame_count)}
    truth = {
        point_id: {index: np.asarray(xyz, dtype=float) for index, xyz in enumerate(samples)}
        for point_id, samples in world_points.items()
    }
    for point_id, samples in truth.items():
        for index, world in samples.items():
            camera_center = np.asarray(camera_centers[index], dtype=float)
            uv = _project(world, camera_center, K)
            tracked_id = (
                f"{point_id}_broken"
                if break_point_id_at is not None and index >= break_point_id_at
                else point_id
            )
            trajectories.setdefault(tracked_id, {})[index] = (float(uv[0]), float(uv[1]))
            column, row = int(round(uv[0])), int(round(uv[1]))
            if 0 <= row < height and 0 <= column < width:
                depth_maps[index][row, column] = float(world[2] - camera_center[2])
    frames = []
    twc_map = {}
    tcw_map = {}
    relative_poses = []
    for index, center_values in enumerate(camera_centers):
        twc = _transform_from_center(center_values)
        tcw = np.linalg.inv(twc)
        twc_map[index], tcw_map[index] = twc, tcw
        depth = DepthObservation(
            depth_map=depth_maps[index],
            raw_model_output=None,
            visualization_depth=None,
            depth_representation=(
                DepthRepresentation.METRIC_DEPTH
                if scale_status == SequenceScaleStatus.METRIC_SEQUENCE
                else DepthRepresentation.RELATIVE_DEPTH
            ),
            scale_status=(
                DepthScaleStatus.METRIC_CALIBRATED
                if scale_status == SequenceScaleStatus.METRIC_SEQUENCE
                else (
                    DepthScaleStatus.RELATIVE_PER_FRAME
                    if scale_status == SequenceScaleStatus.RELATIVE_PER_FRAME
                    else DepthScaleStatus.RELATIVE_SHARED_SEQUENCE
                )
            ),
            larger_value_means=LargerValueMeans.FARTHER,
            valid_mask=np.isfinite(depth_maps[index]) & (depth_maps[index] > 0.0),
            confidence_map=None,
            provider_name="synthetic_dynamic_truth",
            frame_index=index,
            valid=True,
            quality=1.0,
            metadata={"ground_truth": True},
        )
        camera = CameraObservation.from_parameters(
            K=K,
            image_width=width,
            image_height=height,
            intrinsics_source="synthetic_truth",
            quality=1.0,
            T_world_from_camera=twc,
            T_camera_from_world=tcw,
            pose_source="synthetic_truth",
        )
        frames.append(
            Shared3DFrameObservation(
                video_id="synthetic_video",
                frame_index=index,
                image_width=width,
                image_height=height,
                camera=camera,
                depth=depth,
                objects=(),
                valid=True,
                quality=1.0,
            )
        )
        previous = index - 1 if index else None
        relative = (
            np.eye(4)
            if previous is None
            else tcw @ twc_map[previous]
        )
        relative_poses.append(
            RelativePoseObservation.from_transforms(
                source_frame_index=previous,
                target_frame_index=index,
                T_world_from_camera=twc,
                relative_pose_from_previous=relative,
                pose_source="synthetic_reference_gauge" if previous is None else "synthetic_truth",
                pose_quality=1.0,
                background_support_count=0 if previous is None else 100,
                background_inlier_ratio=1.0,
                reprojection_error=0.0,
                metadata={"model": "reference_gauge" if previous is None else "full_se3"},
            )
        )
    cuts = {index: index == scene_cut_frame for index in range(frame_count)}
    clip = Shared3DClipObservation(
        video_id="synthetic_video",
        clip_id="synthetic_clip",
        frame_indices=tuple(range(frame_count)),
        frames=tuple(frames),
        reference_frame_index=0,
        T_world_from_camera_by_frame=twc_map,
        T_camera_from_world_by_frame=tcw_map,
        relative_poses=tuple(relative_poses),
        sequence_scale_status=scale_status,
        depth_alignment_observations=(),
        scene_cut_flags=cuts,
        background_track_ids=(),
        foreground_object_ids=("synthetic_object",),
        provider_name="synthetic_dynamic_truth",
        valid=True,
        quality=1.0,
        metadata={"pose_scale_compatible_with_depth": True, "ground_truth": True},
    )
    tracker = SyntheticPointTracker(trajectories)
    points_2d = tracker.track(images, object_track_id="synthetic_object")
    static_ratio = 1.0 if mode == DynamicGeometryMode.STATIC_CAMERA_3D else 0.0
    rotation_ratio = 1.0 if mode == DynamicGeometryMode.ROTATION_COMPENSATED else 0.0
    full_ratio = 1.0 if mode == DynamicGeometryMode.FULL_SE3_3D else 0.0
    readiness = assess_dynamic_3d_readiness(
        clip,
        valid_shared_frame_ratio=1.0,
        pose_graph_connected_ratio=1.0,
        static_pose_ratio=static_ratio,
        rotation_only_ratio=rotation_ratio,
        full_se3_ratio=full_ratio,
        depth_alignment_valid_ratio=(0.0 if scale_status == SequenceScaleStatus.RELATIVE_PER_FRAME else 1.0),
        independent_track_coverage=1.0,
        mean_track_length=float(frame_count),
        reprojection_error_before=1.0,
        reprojection_error_after=0.0,
        depth_stability_before=1.0,
        depth_stability_after=0.0,
        background_3d_stability_before=1.0,
        background_3d_stability_after=0.0,
        thresholds=Dynamic3DReadinessThresholds(
            minimum_static_pose_ratio=0.75,
            minimum_full_se3_ratio=0.75,
        ),
    )
    points_3d = reconstruct_point_tracks_3d(points_2d, clip, readiness)
    return SyntheticDynamicScene(
        clip=clip,
        readiness=readiness,
        images=images,
        points_2d=points_2d,
        points_3d=points_3d,
        K=K,
        world_points=truth,
    )


def constant_velocity_points(frame_count: int = 5) -> dict[str, list[np.ndarray]]:
    """Return one smooth object point and one static background point."""

    return {
        "object_p0": [np.asarray([0.1 * index, 0.0, 5.0]) for index in range(frame_count)],
        "background_p0": [np.asarray([0.5, 0.2, 7.0]) for _ in range(frame_count)],
    }
