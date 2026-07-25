"""D2 facade over the existing short-baseline projection residual."""

from __future__ import annotations

import math

import numpy as np

from data.schemas import ClipObservation, ResidualEvidence
from .geometry import backproject_points
from semantic3d.pose_d2.contracts import PairwisePoseObservation, PoseProviderStatus
from semantic3d.pose_d2.residuals import compute_d2_projection_residual


def _pose(source: int, target: int, transform: np.ndarray, confidence: float) -> PairwisePoseObservation:
    return PairwisePoseObservation(
        frame_t=source,
        frame_t1=target,
        rotation=transform[:3, :3],
        translation=transform[:3, 3],
        T_target_from_source=transform,
        pose_convention="X_target_camera=T_target_from_source@X_source_camera",
        camera_to_world_or_world_to_camera="camera_t_to_camera_t1",
        translation_scale_status="metric_model_depth",
        inlier_count=1,
        inlier_ratio=1.0,
        reprojection_error=0.0,
        static_background_ratio=1.0,
        dynamic_foreground_ratio=0.0,
        confidence=confidence,
        provider_status=PoseProviderStatus.ESTIMATED_VALID,
        failure_reason="",
        background_candidates=1,
        foreground_rejected=0,
        geometric_inliers=1,
        degeneracy_status="none",
        provider_name="paper_core_shared_pose",
        valid=True,
    )


def compute_reprojection_residuals(clip: ClipObservation) -> list[ResidualEvidence]:
    output: list[ResidualEvidence] = []
    frames = {frame.frame_index: frame for frame in clip.frames}
    for track in clip.tracks:
        if track.points_3d is None:
            continue
        for position in range(1, len(track.frame_indices)):
            source_index = track.frame_indices[position - 1]
            target_index = track.frame_indices[position]
            target = frames.get(target_index)
            if (
                target is None
                or target.relative_pose_from_previous is None
                or target.intrinsics is None
            ):
                output.append(
                    ResidualEvidence.unavailable(
                        "point_reprojection",
                        "point",
                        "relative_pose_or_intrinsics_unavailable",
                        temporal_support={"frame_index": target_index},
                    )
                )
                continue
            pose = _pose(
                source_index,
                target_index,
                target.relative_pose_from_previous,
                target.confidence.get("relative_pose", 1.0),
            )
            result = compute_d2_projection_residual(
                evidence_id=f"{clip.clip_id}:{track.track_id}:{target_index}",
                evidence_type="point",
                video_id=clip.video_id,
                clip_id=clip.clip_id,
                pose=pose,
                source_point_camera_m=track.points_3d[position - 1],
                target_observed_uv=track.actual_xy[position],
                K_target=target.intrinsics,
                image_width=target.image.shape[1],
                image_height=target.image.shape[0],
                target_depth_m=target.metric_depth,
                target_depth_valid_mask=target.depth_valid_mask,
                object_id=track.object_id,
                track_id=track.track_id,
                point_id=track.track_id,
                point_confidence=track.confidence,
            )
            support = {
                "kind": "point",
                "xy": track.actual_xy[position].tolist(),
                "frame_index": target_index,
                "track_id": track.track_id,
                "object_id": track.object_id,
            }
            if not result.valid:
                output.append(
                    ResidualEvidence.unavailable(
                        "point_reprojection",
                        "point",
                        result.failure_reason,
                        spatial_support=support,
                        temporal_support={"frame_index": target_index},
                    )
                )
                continue
            output.append(
                ResidualEvidence.observed(
                    "point_reprojection",
                    "point",
                    result.point_reprojection_residual,
                    confidence=min(result.pose_confidence, result.point_confidence),
                    spatial_support=support,
                    temporal_support={"frame_index": target_index},
                )
            )
            if math.isfinite(result.depth_reprojection_residual):
                output.append(
                    ResidualEvidence.observed(
                        "depth_reprojection",
                        "point",
                        result.depth_reprojection_residual,
                        confidence=min(result.pose_confidence, result.point_confidence),
                        spatial_support=support,
                        temporal_support={"frame_index": target_index},
                    )
                )
    for source, target in zip(clip.frames, clip.frames[1:]):
        correspondences = target.boundary_correspondences
        if correspondences is None:
            continue
        if (
            source.metric_depth is None
            or source.intrinsics is None
            or target.intrinsics is None
            or target.relative_pose_from_previous is None
        ):
            output.append(
                ResidualEvidence.unavailable(
                    "boundary_reprojection",
                    "boundary",
                    "metric_depth_intrinsics_or_pose_unavailable",
                    temporal_support={"frame_index": target.frame_index},
                )
            )
            continue
        pose = _pose(
            source.frame_index,
            target.frame_index,
            target.relative_pose_from_previous,
            target.confidence.get("relative_pose", 1.0),
        )
        for index, row in enumerate(correspondences):
            source_xy, target_xy = row[:2], row[2:]
            x, y = (int(round(value)) for value in source_xy)
            if not (0 <= y < source.image.shape[0] and 0 <= x < source.image.shape[1]):
                continue
            if source.depth_valid_mask is not None and not source.depth_valid_mask[y, x]:
                continue
            point = backproject_points(
                source_xy[None, :],
                np.asarray([source.metric_depth[y, x]]),
                source.intrinsics,
            )
            if not len(point):
                continue
            result = compute_d2_projection_residual(
                evidence_id=f"{clip.clip_id}:boundary:{target.frame_index}:{index}",
                evidence_type="boundary",
                video_id=clip.video_id,
                clip_id=clip.clip_id,
                pose=pose,
                source_point_camera_m=point[0],
                target_observed_uv=target_xy,
                K_target=target.intrinsics,
                image_width=target.image.shape[1],
                image_height=target.image.shape[0],
                target_depth_m=target.metric_depth,
                target_depth_valid_mask=target.depth_valid_mask,
                point_id=f"boundary_{index}",
                point_confidence=target.confidence.get("boundary_correspondence", 1.0),
            )
            support = {
                "kind": "point",
                "xy": target_xy.tolist(),
                "frame_index": target.frame_index,
            }
            if result.valid and math.isfinite(result.boundary_reprojection_residual):
                output.append(
                    ResidualEvidence.observed(
                        "boundary_reprojection",
                        "boundary",
                        result.boundary_reprojection_residual,
                        confidence=min(result.pose_confidence, result.point_confidence),
                        spatial_support=support,
                        temporal_support={"frame_index": target.frame_index},
                    )
                )
            else:
                output.append(
                    ResidualEvidence.unavailable(
                        "boundary_reprojection",
                        "boundary",
                        result.failure_reason or "boundary_reprojection_unavailable",
                        spatial_support=support,
                        temporal_support={"frame_index": target.frame_index},
                    )
                )
    return output


__all__ = ["compute_d2_projection_residual", "compute_reprojection_residuals"]
