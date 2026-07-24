"""Tests for P4-C3B-M4 pose, static verification, alignment, and D2."""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from semantic3d.pose_d2 import (
    D2VisibilityStatus,
    PairwisePoseObservation,
    PoseProviderStatus,
    ShortBaselinePoseProvider,
    StaticVerificationObservation,
    build_clip_local_alignment,
    compute_d2_projection_residual,
    estimate_metric_transform_from_correspondences,
)


def _K(width: int = 320, height: int = 240) -> np.ndarray:
    return np.asarray(
        [[300.0, 0.0, width / 2.0], [0.0, 300.0, height / 2.0], [0.0, 0.0, 1.0]]
    )


def _pose(
    transform: np.ndarray | None = None,
    *,
    status: PoseProviderStatus = PoseProviderStatus.ESTIMATED_VALID,
    confidence: float = 0.9,
) -> PairwisePoseObservation:
    matrix = np.eye(4) if transform is None else np.asarray(transform, dtype=float)
    return PairwisePoseObservation(
        frame_t=0,
        frame_t1=1,
        rotation=matrix[:3, :3],
        translation=matrix[:3, 3],
        T_target_from_source=matrix,
        pose_convention="X_target_camera=T_target_from_source@X_source_camera",
        camera_to_world_or_world_to_camera="camera_t_to_camera_t1",
        translation_scale_status="metric_model_depth",
        inlier_count=50,
        inlier_ratio=0.8,
        reprojection_error=0.2,
        static_background_ratio=0.9,
        dynamic_foreground_ratio=0.1,
        confidence=confidence,
        provider_status=status,
        failure_reason="",
        background_candidates=70,
        foreground_rejected=20,
        geometric_inliers=50,
        degeneracy_status="none",
        provider_name="synthetic_pose",
        valid=True,
    )


def _project(K: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = (K @ points.T).T
    return homogeneous[:, :2] / homogeneous[:, 2:3]


def test_known_rotation_translation_metric_pnp_and_pose_direction() -> None:
    K = _K()
    source_uv = np.asarray(
        [[u, v] for v in np.linspace(70, 170, 6) for u in np.linspace(80, 240, 7)]
    )
    source_uv = np.rint(source_uv)
    z_values = 3.0 + 0.001 * (source_uv[:, 0] - 160.0)
    rays = (np.linalg.inv(K) @ np.column_stack((source_uv, np.ones(len(source_uv)))).T).T
    points = rays * z_values[:, None]
    angle = math.radians(3.0)
    rotation = np.asarray(
        [
            [math.cos(angle), 0.0, math.sin(angle)],
            [0.0, 1.0, 0.0],
            [-math.sin(angle), 0.0, math.cos(angle)],
        ]
    )
    translation = np.asarray([0.10, -0.02, 0.03])
    target_points = (rotation @ points.T).T + translation
    target_uv = _project(K, target_points)
    depth = np.full((240, 320), np.nan, dtype=float)
    valid = np.zeros(depth.shape, dtype=bool)
    for uv, point in zip(source_uv.astype(int), points, strict=True):
        depth[uv[1], uv[0]] = point[2]
        valid[uv[1], uv[0]] = True

    result = estimate_metric_transform_from_correspondences(
        source_uv,
        target_uv,
        depth,
        valid,
        K,
        K,
        minimum_inliers=8,
    )
    assert result.transform is not None
    assert result.inlier_mask.sum() >= 30
    predicted = (
        result.transform[:3, :3] @ points.T
    ).T + result.transform[:3, 3]
    assert np.median(np.linalg.norm(_project(K, predicted) - target_uv, axis=1)) < 0.2
    assert np.linalg.norm(result.transform[:3, 3] - translation) < 0.03

    inverse_prediction = (
        np.linalg.inv(result.transform)[:3, :3] @ points.T
    ).T + np.linalg.inv(result.transform)[:3, 3]
    assert np.mean(np.linalg.norm(_project(K, inverse_prediction) - target_uv, axis=1)) > 5.0


def test_static_identity_requires_verified_multi_evidence() -> None:
    with pytest.raises(ValueError, match="verified static evidence"):
        PairwisePoseObservation(
            frame_t=0,
            frame_t1=1,
            rotation=np.eye(3),
            translation=np.zeros(3),
            T_target_from_source=np.eye(4),
            pose_convention="X_target_camera=T_target_from_source@X_source_camera",
            camera_to_world_or_world_to_camera="camera_t_to_camera_t1",
            translation_scale_status="zero_static",
            inlier_count=20,
            inlier_ratio=1.0,
            reprojection_error=0.0,
            static_background_ratio=1.0,
            dynamic_foreground_ratio=0.0,
            confidence=1.0,
            provider_status=PoseProviderStatus.VERIFIED_STATIC,
            failure_reason="",
            background_candidates=20,
            foreground_rejected=0,
            geometric_inliers=20,
            degeneracy_status="none",
            provider_name="bad_fallback",
            valid=True,
        )


def test_provider_failure_never_contains_identity_pose() -> None:
    provider = ShortBaselinePoseProvider()
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    result = provider.estimate_pair(
        image,
        image,
        frame_t=0,
        frame_t1=1,
        K_source=None,
        K_target=None,
        source_depth_m=None,
        source_depth_valid_mask=None,
    )
    assert result.provider_status == PoseProviderStatus.BLOCKED_BY_INTRINSICS
    assert result.T_target_from_source is None
    assert not result.valid
    with pytest.raises(ValueError, match="Provider failure cannot contain"):
        PairwisePoseObservation(
            frame_t=0,
            frame_t1=1,
            rotation=np.eye(3),
            translation=np.zeros(3),
            T_target_from_source=np.eye(4),
            pose_convention="X_target_camera=T_target_from_source@X_source_camera",
            camera_to_world_or_world_to_camera="camera_t_to_camera_t1",
            translation_scale_status="not_available",
            inlier_count=0,
            inlier_ratio=0.0,
            reprojection_error=float("nan"),
            static_background_ratio=0.0,
            dynamic_foreground_ratio=0.0,
            confidence=0.0,
            provider_status=PoseProviderStatus.PROVIDER_FAILED,
            failure_reason="synthetic_provider_failure",
            background_candidates=0,
            foreground_rejected=0,
            geometric_inliers=0,
            degeneracy_status="provider_failure",
            provider_name="failed_provider",
            valid=False,
        )


def test_dynamic_foreground_is_excluded_from_pose_support() -> None:
    rng = np.random.default_rng(4)
    gray = rng.integers(0, 256, size=(180, 240), dtype=np.uint8)
    source = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    affine = np.asarray([[1.0, 0.0, 2.0], [0.0, 1.0, 0.0]], dtype=float)
    target = cv2.warpAffine(source, affine, (240, 180))
    foreground = np.zeros((180, 240), dtype=bool)
    foreground[:, :100] = True
    depth = np.full((180, 240), 4.0, dtype=float)
    valid = np.ones(depth.shape, dtype=bool)
    result = ShortBaselinePoseProvider().estimate_pair(
        source,
        target,
        frame_t=0,
        frame_t1=1,
        K_source=_K(240, 180),
        K_target=_K(240, 180),
        source_depth_m=depth,
        source_depth_valid_mask=valid,
        source_foreground_mask=foreground,
        target_foreground_mask=foreground,
    )
    assert result.foreground_rejected > 0
    assert result.dynamic_foreground_ratio > 0.0
    assert result.metadata["foreground_masks_excluded"] is True


def test_low_texture_is_blocked_not_static_identity() -> None:
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    depth = np.full((120, 160), 3.0)
    result = ShortBaselinePoseProvider().estimate_pair(
        image,
        image,
        frame_t=0,
        frame_t1=1,
        K_source=_K(160, 120),
        K_target=_K(160, 120),
        source_depth_m=depth,
        source_depth_valid_mask=np.ones(depth.shape, dtype=bool),
    )
    assert result.provider_status == PoseProviderStatus.BLOCKED_BY_CORRESPONDENCE
    assert result.T_target_from_source is None
    assert not result.valid


def test_clip_alignment_uses_reference_gauge_not_world_frame() -> None:
    transform = np.eye(4)
    transform[0, 3] = 0.1
    alignments = build_clip_local_alignment("clip", [0, 1], [_pose(transform)])
    assert len(alignments) == 2
    assert alignments[0].metadata["reference_gauge_only"] is True
    assert alignments[1].coordinate_frame == "clip_local_aligned"
    assert np.allclose(alignments[1].T_clip_from_camera, np.linalg.inv(transform))
    assert alignments[1].metadata["world_frame_claimed"] is False


def test_d2_known_pose_has_zero_residual_and_no_authenticity_decision() -> None:
    K = _K()
    transform = np.eye(4)
    transform[0, 3] = 0.1
    point = np.asarray([0.0, 0.0, 2.0])
    target = (transform @ np.asarray([*point, 1.0]))[:3]
    uv = _project(K, target[None, :])[0]
    result = compute_d2_projection_residual(
        evidence_id="known",
        evidence_type="point",
        video_id="v",
        clip_id="c",
        pose=_pose(transform, confidence=0.8),
        source_point_camera_m=point,
        target_observed_uv=uv,
        K_target=K,
        image_width=320,
        image_height=240,
        point_confidence=0.7,
    )
    assert result.valid
    assert result.point_reprojection_residual == pytest.approx(0.0, abs=1e-12)
    assert result.pose_confidence == 0.8
    assert result.point_confidence == 0.7
    assert result.residual_is_authenticity_decision is False
    assert result.metadata["authenticity_threshold_applied"] is False


def test_d2_out_of_frame_is_nan() -> None:
    transform = np.eye(4)
    transform[0, 3] = 20.0
    result = compute_d2_projection_residual(
        evidence_id="outside",
        evidence_type="point",
        video_id="v",
        clip_id="c",
        pose=_pose(transform),
        source_point_camera_m=[0.0, 0.0, 2.0],
        target_observed_uv=[160.0, 120.0],
        K_target=_K(),
        image_width=320,
        image_height=240,
    )
    assert result.visibility_status == D2VisibilityStatus.OUT_OF_FRAME
    assert not result.valid
    assert math.isnan(result.point_reprojection_residual)


@pytest.mark.parametrize(
    ("target_depth", "expected"),
    [
        (1.0, D2VisibilityStatus.OCCLUDED),
        (5.0, D2VisibilityStatus.DEPTH_CONFLICT),
    ],
)
def test_d2_occlusion_and_depth_conflict_are_not_high_residual(
    target_depth: float,
    expected: D2VisibilityStatus,
) -> None:
    depth = np.full((240, 320), target_depth, dtype=float)
    result = compute_d2_projection_residual(
        evidence_id=expected.value,
        evidence_type="point",
        video_id="v",
        clip_id="c",
        pose=_pose(),
        source_point_camera_m=[0.0, 0.0, 3.0],
        target_observed_uv=[160.0, 120.0],
        K_target=_K(),
        image_width=320,
        image_height=240,
        target_depth_m=depth,
        target_depth_valid_mask=np.ones(depth.shape, dtype=bool),
    )
    assert result.visibility_status == expected
    assert not result.valid
    assert math.isnan(result.depth_reprojection_residual)


def test_static_verification_contract_accepts_independent_evidence() -> None:
    verification = StaticVerificationObservation(
        source_frame_index=0,
        target_frame_index=1,
        global_flow_small=True,
        background_displacement_small=True,
        homography_motion_small=True,
        essential_motion_not_supported=False,
        parallax_small=True,
        image_difference_stable=True,
        evidence_count=5,
        required_evidence_count=4,
        verified_static=True,
        median_global_flow=0.1,
        median_background_flow=0.1,
        median_parallax=0.05,
        homography_rotation_degrees=0.01,
        pnp_translation_norm=0.02,
        confidence=5 / 6,
    )
    pose = PairwisePoseObservation(
        frame_t=0,
        frame_t1=1,
        rotation=np.eye(3),
        translation=np.zeros(3),
        T_target_from_source=np.eye(4),
        pose_convention="X_target_camera=T_target_from_source@X_source_camera",
        camera_to_world_or_world_to_camera="camera_t_to_camera_t1",
        translation_scale_status="zero_static",
        inlier_count=20,
        inlier_ratio=1.0,
        reprojection_error=0.1,
        static_background_ratio=0.9,
        dynamic_foreground_ratio=0.1,
        confidence=0.8,
        provider_status=PoseProviderStatus.VERIFIED_STATIC,
        failure_reason="",
        background_candidates=20,
        foreground_rejected=5,
        geometric_inliers=20,
        degeneracy_status="none",
        provider_name="verified",
        valid=True,
        static_verification=verification,
    )
    assert pose.valid
    assert np.array_equal(pose.T_target_from_source, np.eye(4))
