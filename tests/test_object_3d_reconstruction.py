from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from semantic3d.depth_provider import LegacyDepthProviderAdapter
from semantic3d.observations import ObjectObservationJSON
from semantic3d.reconstruction.depth_sampling import sample_depth
from semantic3d.reconstruction.object_3d_reconstructor import Object3DReconstructor
from semantic3d.shared_3d_observation import GeometryScaleStatus, GeometryScaleUnit

from synthetic_geometry import (
    cuboid_xyz,
    projected_point_observations,
    rasterize_sparse_depth,
    synthetic_camera,
    synthetic_depth_observation,
)


def _object(bbox: list[float] | None = None) -> ObjectObservationJSON:
    return ObjectObservationJSON(
        object_id="cube",
        label="synthetic_cuboid",
        mask_area=10_000.0,
        frame_area=640.0 * 480.0,
        depth=8.0,
        confidence=1.0,
        bbox=bbox,
        track_id="track_cube",
        canonical_label="synthetic_cuboid",
        provenance={"source": "synthetic_ground_truth"},
    )


def _reconstruct_cube(*, metric: bool = True, with_pose: bool = True):
    camera = synthetic_camera(with_pose=with_pose)
    xyz = cuboid_xyz()
    assert camera.K is not None
    points_2d = projected_point_observations(xyz, camera.K)
    depth_map = rasterize_sparse_depth(xyz, camera.K)
    depth = synthetic_depth_observation(depth_map, metric=metric)
    pixels = np.asarray([[point.x, point.y] for point in points_2d], dtype=float)
    bbox = [
        float(np.min(pixels[:, 0])),
        float(np.min(pixels[:, 1])),
        float(np.max(pixels[:, 0])),
        float(np.max(pixels[:, 1])),
    ]
    reconstructed = Object3DReconstructor().reconstruct(
        video_id="synthetic_video",
        frame_index=0,
        obj=_object(bbox),
        depth=depth,
        camera=camera,
        boundary_points_2d=points_2d,
    )
    return reconstructed, xyz, depth, camera, points_2d


def test_local_median_filters_single_depth_outlier() -> None:
    depth_map = np.full((7, 7), 5.0, dtype=np.float32)
    depth_map[3, 3] = 100.0
    depth = synthetic_depth_observation(depth_map)
    sampled = sample_depth(depth, 3.0, 3.0, "local_median_3x3")
    assert sampled.valid
    assert sampled.sampled_depth == pytest.approx(5.0)
    assert sampled.local_valid_ratio == pytest.approx(1.0)


def test_synthetic_cube_center_extent_and_diagonal_are_recovered() -> None:
    reconstructed, xyz, _, _, _ = _reconstruct_cube(metric=True)
    assert reconstructed.valid
    assert reconstructed.center_3d_camera is not None
    np.testing.assert_allclose(
        reconstructed.center_3d_camera.as_array(), np.median(xyz, axis=0), atol=1e-6
    )
    extents = reconstructed.scale_descriptors["axis_aligned_extent"]
    assert extents["extent_x"] == pytest.approx(2.0, abs=1e-6)
    assert extents["extent_y"] == pytest.approx(2.0, abs=1e-6)
    assert extents["extent_z"] == pytest.approx(2.0, abs=1e-6)
    assert reconstructed.observed_scale_3d == pytest.approx(np.sqrt(12.0), abs=1e-6)
    assert reconstructed.scale_descriptors["pca_principal_extents"] is not None
    assert reconstructed.scale_descriptors["equivalent_linear_scale"] == pytest.approx(2.0)
    assert reconstructed.metadata["physical_scale_prior_used"] is False
    assert all(
        {
            "source_pixel",
            "sampled_depth",
            "sampling_method",
            "local_valid_ratio",
            "local_depth_iqr",
            "point_quality",
        }.issubset(point.metadata)
        for point in reconstructed.boundary_points_3d
        if point.valid
    )


def test_identity_pose_produces_explicit_world_points() -> None:
    reconstructed, _, _, _, _ = _reconstruct_cube(metric=True, with_pose=True)
    assert reconstructed.center_3d_camera is not None
    assert reconstructed.center_3d_world is not None
    assert reconstructed.structure_points_3d_world is not None
    np.testing.assert_allclose(
        reconstructed.center_3d_world.as_array(),
        reconstructed.center_3d_camera.as_array(),
        atol=1e-12,
    )


def test_object_center_is_robust_to_one_extreme_depth() -> None:
    camera = synthetic_camera()
    assert camera.K is not None
    xyz = np.asarray(
        [[-1.0, 0.0, 5.0], [0.0, -1.0, 5.0], [0.0, 0.0, 5.0], [0.0, 1.0, 5.0], [10.0, 0.0, 100.0]]
    )
    points_2d = projected_point_observations(xyz, camera.K)
    depth = synthetic_depth_observation(rasterize_sparse_depth(xyz, camera.K))
    result = Object3DReconstructor().reconstruct(
        video_id="synthetic_video",
        frame_index=0,
        obj=_object([100.0, 100.0, 500.0, 400.0]),
        depth=depth,
        camera=camera,
        boundary_points_2d=points_2d,
    )
    assert result.center_3d_camera is not None
    np.testing.assert_allclose(result.center_3d_camera.as_array(), [0.0, 0.0, 5.0], atol=1e-6)


def test_relative_depth_produces_relative_unit_and_rejects_cross_frame_scale() -> None:
    reconstructed, _, _, _, _ = _reconstruct_cube(metric=False)
    assert reconstructed.scale_status == GeometryScaleStatus.RELATIVE_3D
    assert reconstructed.scale_unit == GeometryScaleUnit.RELATIVE_UNIT
    assert reconstructed.depth_scale_status.value == "relative_per_frame"
    with pytest.raises(ValueError, match="relative_per_frame"):
        reconstructed.require_cross_frame_scale_comparable()


def test_only_metric_calibrated_depth_produces_meter_unit() -> None:
    reconstructed, _, _, _, _ = _reconstruct_cube(metric=True)
    assert reconstructed.scale_status == GeometryScaleStatus.METRIC_3D
    assert reconstructed.scale_unit == GeometryScaleUnit.METER
    assert reconstructed.require_cross_frame_scale_comparable() > 0.0


def test_normalized_points_remain_unitless_and_separate_from_scale() -> None:
    reconstructed, _, _, _, _ = _reconstruct_cube(metric=True)
    assert reconstructed.normalized_structure_points
    assert all(
        point.scale_status == GeometryScaleStatus.NORMALIZED_SHAPE
        and point.coordinate_frame == "object_normalized"
        and point.metadata["unit"] == "unitless"
        for point in reconstructed.normalized_structure_points
    )
    assert reconstructed.observed_scale_3d is not None


def test_missing_pose_keeps_camera_points_but_not_world_points() -> None:
    reconstructed, _, _, _, _ = _reconstruct_cube(metric=False, with_pose=False)
    assert reconstructed.valid
    assert reconstructed.center_3d_camera is not None
    assert reconstructed.center_3d_world is None
    assert reconstructed.structure_points_3d_world is None
    assert reconstructed.metadata["world_reconstruction_missing_reason"] == "no_camera_pose"


def test_legacy_normalized_depth_is_rejected_by_reconstructor(tmp_path: Path) -> None:
    frame_path = tmp_path / "frame.png"
    cv2.imwrite(str(frame_path), np.zeros((480, 640, 3), dtype=np.uint8))

    class LegacyProvider:
        def predict_depth(self, _path: Path) -> np.ndarray:
            return np.full((480, 640), 5.0, dtype=np.float32)

    depth = LegacyDepthProviderAdapter(LegacyProvider()).predict_observation(frame_path, 0)
    camera = synthetic_camera()
    result = Object3DReconstructor().reconstruct(
        video_id="video",
        frame_index=0,
        obj=_object([100.0, 100.0, 200.0, 200.0]),
        depth=depth,
        camera=camera,
    )
    assert not result.valid
    assert result.missing_reason == "invalid_geometry_depth"
    assert result.center_3d is None
