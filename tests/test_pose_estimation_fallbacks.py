from __future__ import annotations

import numpy as np

from semantic3d.sequence_geometry import (
    PoseModelType,
    TranslationScaleStatus,
    estimate_adaptive_pose_candidates,
    estimate_essential_pose_from_correspondences,
    estimate_rotation_pose_from_correspondences,
)


def _project(points: np.ndarray, K: np.ndarray) -> np.ndarray:
    projected = (K @ points.T).T
    return projected[:, :2] / projected[:, 2:3]


def test_rotation_only_is_not_marked_as_full_se3() -> None:
    K = np.asarray([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    x, y = np.meshgrid(np.linspace(-2.0, 2.0, 8), np.linspace(-1.5, 1.5, 6))
    points = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, 8.0)))
    angle = np.deg2rad(4.0)
    rotation = np.asarray(
        [[np.cos(angle), 0.0, np.sin(angle)], [0.0, 1.0, 0.0], [-np.sin(angle), 0.0, np.cos(angle)]]
    )
    candidate = estimate_rotation_pose_from_correspondences(
        _project(points, K),
        _project((rotation @ points.T).T, K),
        K,
        source_frame_index=0,
        target_frame_index=1,
    )
    assert candidate.valid
    assert candidate.pose_model_type == PoseModelType.ROTATION_HOMOGRAPHY
    assert candidate.rotation_valid
    assert not candidate.translation_valid
    assert not candidate.full_se3
    assert candidate.translation_scale_status == TranslationScaleStatus.NOT_AVAILABLE


def test_general_se3_correspondences_recover_rotation_and_translation_direction() -> None:
    generator = np.random.default_rng(12)
    K = np.asarray([[700.0, 0.0, 320.0], [0.0, 700.0, 240.0], [0.0, 0.0, 1.0]])
    points = np.column_stack(
        (
            generator.uniform(-2.0, 2.0, 200),
            generator.uniform(-1.5, 1.5, 200),
            generator.uniform(5.0, 12.0, 200),
        )
    )
    angle = np.deg2rad(3.0)
    rotation = np.asarray(
        [[np.cos(angle), 0.0, np.sin(angle)], [0.0, 1.0, 0.0], [-np.sin(angle), 0.0, np.cos(angle)]]
    )
    translation = np.asarray([0.35, 0.02, 0.05])
    target_points = (rotation @ points.T).T + translation
    result = estimate_essential_pose_from_correspondences(
        _project(points, K),
        _project(target_points, K),
        K,
        source_frame_index=0,
        target_frame_index=1,
    )
    assert result.valid and result.full_se3
    assert result.pose_model_type == PoseModelType.ESSENTIAL_SE3
    recovered_rotation = result.T_target_from_source[:3, :3]
    np.testing.assert_allclose(recovered_rotation, rotation, atol=2e-3)
    recovered_direction = result.T_target_from_source[:3, 3]
    assert abs(float(np.dot(recovered_direction, translation / np.linalg.norm(translation)))) > 0.98
    assert result.translation_scale_status == TranslationScaleStatus.DIRECTION_ONLY


def test_scene_cut_prevents_all_cross_boundary_stride_edges() -> None:
    generator = np.random.default_rng(13)
    image = generator.integers(0, 255, size=(100, 140, 3), dtype=np.uint8)
    images = {index: image.copy() for index in range(5)}
    K = np.asarray([[150.0, 0.0, 69.5], [0.0, 150.0, 49.5], [0.0, 0.0, 1.0]])
    pairs = estimate_adaptive_pose_candidates(
        images,
        K,
        frame_indices=range(5),
        scene_cut_flags={3: True},
        temporal_strides=(1, 2, 4),
    )
    assert all(not (pair.tracks.source_frame_index < 3 <= pair.tracks.target_frame_index) for pair in pairs)

