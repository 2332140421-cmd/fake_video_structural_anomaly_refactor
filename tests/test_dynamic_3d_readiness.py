from __future__ import annotations

import math

from semantic3d.dynamic_3d import (
    Dynamic3DReadinessThresholds,
    DynamicGeometryMode,
    assess_dynamic_3d_readiness,
    relative_improvement,
)
from semantic3d.sequence_geometry import RelativePoseObservation

from synthetic_dynamic_3d import constant_velocity_points, make_synthetic_dynamic_scene


def _scene(mode: DynamicGeometryMode = DynamicGeometryMode.STATIC_CAMERA_3D):
    return make_synthetic_dynamic_scene(
        camera_centers=[(0.0, 0.0, 0.0)] * 4,
        world_points=constant_velocity_points(4),
        mode=mode,
    )


def test_readiness_modes_authorize_only_supported_geometry() -> None:
    static = _scene(DynamicGeometryMode.STATIC_CAMERA_3D).readiness
    rotation = _scene(DynamicGeometryMode.ROTATION_COMPENSATED).readiness
    full = _scene(DynamicGeometryMode.FULL_SE3_3D).readiness
    assert static.mode == DynamicGeometryMode.STATIC_CAMERA_3D
    assert static.mode.allows_camera_frame_3d_tracks and not static.allows_world_3d
    assert rotation.mode == DynamicGeometryMode.ROTATION_COMPENSATED
    assert rotation.mode.allows_rotation_compensation
    assert not rotation.mode.allows_camera_frame_3d_tracks
    assert full.mode == DynamicGeometryMode.FULL_SE3_3D
    assert full.allows_world_3d


def test_real2_style_depth_degradation_is_not_ready() -> None:
    scene = _scene(DynamicGeometryMode.ROTATION_COMPENSATED)
    result = assess_dynamic_3d_readiness(
        scene.clip,
        valid_shared_frame_ratio=1.0,
        pose_graph_connected_ratio=1.0,
        static_pose_ratio=2 / 7,
        rotation_only_ratio=5 / 7,
        full_se3_ratio=0.0,
        depth_alignment_valid_ratio=1.0,
        independent_track_coverage=1.0,
        mean_track_length=4.0,
        reprojection_error_before=0.433465,
        reprojection_error_after=0.0658748,
        depth_stability_before=0.0327663,
        depth_stability_after=0.0442668,
        background_3d_stability_before=0.03,
        background_3d_stability_after=0.02,
    )
    assert result.mode == DynamicGeometryMode.UNAVAILABLE
    assert not result.dynamic_3d_ready
    assert "depth_stability_improved" in result.missing_reason
    assert result.depth_stability_improvement < 0.0


def test_missing_comparison_is_nan_not_zero() -> None:
    assert math.isnan(relative_improvement(float("nan"), 1.0))
    scene = _scene()
    result = assess_dynamic_3d_readiness(
        scene.clip,
        valid_shared_frame_ratio=1.0,
        pose_graph_connected_ratio=1.0,
        static_pose_ratio=1.0,
        rotation_only_ratio=0.0,
        full_se3_ratio=0.0,
        depth_alignment_valid_ratio=1.0,
        independent_track_coverage=1.0,
        mean_track_length=4.0,
        reprojection_error_before=float("nan"),
        reprojection_error_after=float("nan"),
        depth_stability_before=1.0,
        depth_stability_after=0.0,
        background_3d_stability_before=1.0,
        background_3d_stability_after=0.0,
    )
    assert math.isnan(result.reprojection_improvement)
    assert not result.dynamic_3d_ready


def test_valid_static_identity_remains_distinct_from_missing_pose() -> None:
    identity = _scene().clip.relative_poses[1]
    missing = RelativePoseObservation.missing(
        target_frame_index=1,
        source_frame_index=0,
        reason="no_background_evidence",
    )
    assert identity.valid and identity.is_identity_relative_pose
    assert not missing.valid and not missing.is_identity_relative_pose


def test_thresholds_load_from_project_config() -> None:
    thresholds = Dynamic3DReadinessThresholds.from_yaml(
        "configs/dynamic_3d_readiness.yaml"
    )
    assert thresholds.minimum_pose_graph_connected_ratio == 0.8
    assert thresholds.minimum_reprojection_improvement == 0.05
