from __future__ import annotations

import numpy as np

from semantic3d.sequence_geometry import (
    CameraMotionRegime,
    LayeredPoseEstimator,
    PoseModelType,
    classify_motion_regime,
)


def _regime(**overrides):
    values = {
        "source_frame_index": 0,
        "target_frame_index": 1,
        "background_candidate_count": 100,
        "background_point_count": 90,
        "background_match_count": 80,
        "background_inlier_count": 75,
        "median_background_flow": 0.1,
        "p90_background_flow": 0.3,
        "median_parallax": 0.05,
        "homography_inlier_ratio": 0.9,
        "essential_matrix_inlier_ratio": 0.0,
        "model_reprojection_error": 0.1,
        "image_difference": 0.01,
        "foreground_excluded_ratio": 0.1,
        "spatial_coverage_ratio": 0.7,
        "quadrant_support": (20, 20, 20, 20),
        "feature_concentrated": False,
    }
    values.update(overrides)
    return classify_motion_regime(**values)


def test_static_camera_requires_and_preserves_background_evidence() -> None:
    result = _regime()
    assert result.valid
    assert result.regime == CameraMotionRegime.STATIC_CAMERA
    assert result.supports_identity_pose
    assert result.evidence_source == "foreground_filtered_lk_near_zero_flow"


def test_no_background_evidence_never_becomes_identity() -> None:
    result = _regime(
        background_candidate_count=0,
        background_point_count=0,
        background_match_count=0,
        background_inlier_count=0,
        median_background_flow=float("nan"),
        p90_background_flow=float("nan"),
        median_parallax=float("nan"),
        homography_inlier_ratio=0.0,
    )
    assert not result.valid
    assert not result.supports_identity_pose
    assert result.regime == CameraMotionRegime.LOW_TEXTURE


def test_identical_textured_frames_produce_static_identity_not_reference_gauge() -> None:
    generator = np.random.default_rng(7)
    image = generator.integers(0, 255, size=(160, 220, 3), dtype=np.uint8)
    K = np.asarray([[220.0, 0.0, 109.5], [0.0, 220.0, 79.5], [0.0, 0.0, 1.0]])
    result = LayeredPoseEstimator().estimate_pair(
        image,
        image.copy(),
        K,
        source_frame_index=10,
        target_frame_index=11,
    )
    assert result.motion_regime.regime == CameraMotionRegime.STATIC_CAMERA
    assert result.selected.valid
    assert result.selected.pose_model_type == PoseModelType.STATIC_IDENTITY
    assert result.selected.evidence_source
    assert np.allclose(result.selected.T_target_from_source, np.eye(4))
    assert result.selected.pose_model_type != PoseModelType.REFERENCE_GAUGE

