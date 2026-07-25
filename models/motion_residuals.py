"""D1 facade: adapt unified tracks to the existing dynamic residual producers."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from data.schemas import ClipObservation, ResidualEvidence, TrackObservation
from semantic3d.dynamic_3d.direction_residual import compute_direction_consistency_residuals
from semantic3d.dynamic_3d.motion_model import PointConstantVelocityModel
from semantic3d.dynamic_3d.readiness import DynamicGeometryMode
from semantic3d.dynamic_3d.relative_velocity import compute_relative_velocity_residuals
from semantic3d.dynamic_3d.reprojection_residual import compute_dynamic_reprojection_residual
from semantic3d.dynamic_3d.track_observation import (
    PointTrack2DObservation,
    PointTrack3DObservation,
)
from semantic3d.dynamic_3d.track_residual import compute_track_3d_continuity_residuals
from semantic3d.sequence_geometry import SequenceScaleStatus
from semantic3d.shared_3d_observation import VisibilityStatus
from .geometry import transform_points


def _frame_transforms(clip: ClipObservation) -> dict[int, np.ndarray | None]:
    output: dict[int, np.ndarray | None] = {}
    current: np.ndarray | None = np.eye(4)
    for position, frame in enumerate(clip.frames):
        if position and current is not None:
            current = (
                None
                if frame.relative_pose_from_previous is None
                else frame.relative_pose_from_previous @ current
            )
        output[frame.frame_index] = current
    return output


def _legacy_points(
    track: TrackObservation,
    transforms: dict[int, np.ndarray | None],
) -> list[PointTrack3DObservation]:
    if track.points_3d is None:
        return []
    output = []
    for index, frame in enumerate(track.frame_indices):
        transform_frame_from_clip = transforms.get(frame)
        if not bool(track.valid_mask[index]) or transform_frame_from_clip is None:
            continue
        camera_point = np.asarray(track.points_3d[index], dtype=float)
        aligned = transform_points(
            camera_point[None, :],
            np.linalg.inv(transform_frame_from_clip),
        )[0]
        output.append(
            PointTrack3DObservation(
                point_id=track.track_id,
                object_track_id=track.object_id,
                frame_index=frame,
                pixel_uv=tuple(track.actual_xy[index]),
                observed_depth=float(camera_point[2]),
                point_3d_camera=tuple(camera_point),
                point_3d_world=tuple(aligned),
                visibility=VisibilityStatus.VISIBLE,
                occlusion_status="visible",
                tracking_confidence=track.confidence,
                depth_quality=track.confidence,
                reconstruction_quality=track.confidence,
                source_tracker="paper_core_actual_track",
                scale_status=SequenceScaleStatus.METRIC_SEQUENCE,
                geometry_mode=DynamicGeometryMode.FULL_SE3_3D,
                valid=True,
                missing_reason="",
                metadata={
                    "independent_observation": True,
                    "coordinate_frame": "clip_local_aligned",
                },
            )
        )
    return output


def _adapt(name: str, value: float, confidence: float, track: TrackObservation, frame: int) -> ResidualEvidence:
    return ResidualEvidence.observed(
        name,
        "track",
        value,
        confidence=confidence,
        spatial_support={
            "kind": "track",
            "track_id": track.track_id,
            "xy": track.actual_xy[track.frame_indices.index(frame)].tolist(),
            "frame_index": frame,
        },
        temporal_support={"frame_index": frame, "track_id": track.track_id},
    )


def compute_motion_residuals(clip: ClipObservation) -> list[ResidualEvidence]:
    output: list[ResidualEvidence] = []
    all_points: list[PointTrack3DObservation] = []
    point_owner: dict[str, TrackObservation] = {}
    transforms = _frame_transforms(clip)
    motion_model = PointConstantVelocityModel()
    for track in clip.tracks:
        points = _legacy_points(track, transforms)
        all_points.extend(points)
        point_owner[track.track_id] = track
        if len(points) < 2:
            output.append(
                ResidualEvidence.unavailable(
                    "dynamic_reprojection", "track", "insufficient_metric_track_history"
                )
            )
            continue
        for position in range(2, len(points)):
            previous, current = points[position - 1], points[position]
            history = points[position - 2 : position]
            frame = next(item for item in clip.frames if item.frame_index == current.frame_index)
            pose = frame.relative_pose_from_previous
            transform_frame_from_clip = transforms.get(current.frame_index)
            prediction = motion_model.predict(
                history,
                target_frame_index=current.frame_index,
            )
            if (
                pose is None
                or frame.intrinsics is None
                or transform_frame_from_clip is None
                or not prediction.valid
            ):
                output.append(
                    ResidualEvidence.unavailable(
                        "dynamic_reprojection",
                        "track",
                        prediction.missing_reason
                        or "relative_pose_or_intrinsics_unavailable",
                        temporal_support={"frame_index": current.frame_index},
                    )
                )
                continue
            predicted = transform_points(
                np.asarray(prediction.predicted_point_3d, dtype=float)[None, :],
                transform_frame_from_clip,
            )[0]
            current_2d = PointTrack2DObservation(
                point_id=current.point_id,
                object_track_id=current.object_track_id,
                frame_index=current.frame_index,
                pixel_uv=current.pixel_uv,
                visibility=VisibilityStatus.VISIBLE,
                occlusion_status="visible",
                tracking_confidence=current.tracking_confidence,
                source_tracker="paper_core_actual_track",
                valid=True,
                metadata={"independent_observation": True},
            )
            result = compute_dynamic_reprojection_residual(
                previous,
                current_2d,
                K_current=frame.intrinsics,
                image_width=frame.image.shape[1],
                image_height=frame.image.shape[0],
                relative_pose_current_from_previous=pose,
                geometry_mode=DynamicGeometryMode.FULL_SE3_3D,
                is_background=False,
                predicted_foreground_point_current_camera=predicted,
                has_history_motion_model=True,
                motion_model_type="constant_velocity_3d",
                history_frames=[item.frame_index for item in history],
                support_point_ids=[previous.point_id],
            )
            if result.residual_evidence.valid:
                output.append(
                    _adapt(
                        "dynamic_reprojection",
                        result.residual_evidence.value,
                        result.residual_evidence.quality,
                        track,
                        current.frame_index,
                    )
                )
    continuity = compute_track_3d_continuity_residuals(all_points)
    for item in continuity:
        track = point_owner.get(item.point_id)
        if item.valid and track is not None:
            output.append(
                _adapt(
                    "track_3d_continuity",
                    item.raw_residual,
                    item.raw_evidence.quality,
                    track,
                    item.current_frame_index,
                )
            )
    for item in compute_direction_consistency_residuals(all_points):
        track = point_owner.get(item.point_id)
        if item.own_history.valid and track is not None:
            output.append(
                _adapt(
                    "direction_consistency",
                    item.own_history.value,
                    item.own_history.quality,
                    track,
                    item.current_frame_index,
                )
            )
    scales: dict[str, dict[int, float | None]] = {}
    for track in clip.tracks:
        by_frame: dict[int, float | None] = {}
        for frame_index in track.frame_indices:
            frame = next(item for item in clip.frames if item.frame_index == frame_index)
            obj = next(
                (
                    item
                    for item in frame.objects
                    if item.object_id == track.object_id or item.track_id == track.track_id
                ),
                None,
            )
            points = None if obj is None else obj.metric_surface_xyz
            if points is None or len(points) < 3:
                by_frame[frame_index] = None
            else:
                low, high = np.quantile(points, (0.05, 0.95), axis=0)
                scale = float(np.max(high - low))
                by_frame[frame_index] = scale if scale > 0.0 else None
        scales[track.object_id] = by_frame
    for item in compute_relative_velocity_residuals(all_points, scales):
        track = point_owner.get(item.point_id)
        if item.speed_change_residual.valid and track is not None:
            output.append(
                _adapt(
                    "relative_velocity",
                    item.speed_change_residual.value,
                    item.speed_change_residual.quality,
                    track,
                    item.current_frame_index,
                )
            )
    return output


__all__ = [
    "compute_dynamic_reprojection_residual",
    "compute_track_3d_continuity_residuals",
    "compute_direction_consistency_residuals",
    "compute_relative_velocity_residuals",
    "compute_motion_residuals",
]
