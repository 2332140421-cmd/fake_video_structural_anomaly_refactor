"""P4-C3B-M2 synthetic geometry and evidence-contract tests."""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from semantic3d.metric_scene3d import (
    ImageCoordinateTransform,
    MetricPointType,
    MetricSurfacePoint,
    align_binary_mask,
    backproject_metric_arrays,
    build_object_surface_pointcloud,
    build_single_frame_structure_graph,
    canonicalize_z_depth,
    project_metric_arrays,
    reconstruct_boundary_points,
    select_geometric_track_points,
    to_shared_object_observation,
)
from semantic3d.occlusion.mask_observation import InstanceMaskObservation
from semantic3d.shared_3d_observation import CoordinateFrame


def intrinsics(width: int = 100, height: int = 80) -> np.ndarray:
    """Return non-identity synthetic pinhole intrinsics."""

    return np.asarray(
        [[80.0, 0.0, width / 2.0], [0.0, 82.0, height / 2.0], [0.0, 0.0, 1.0]]
    )


def metric_point(
    point_id: str,
    point_type: MetricPointType,
    xyz: tuple[float, float, float],
) -> MetricSurfacePoint:
    """Construct a valid synthetic M2 point."""

    return MetricSurfacePoint(
        point_id=point_id,
        point_type=point_type,
        frame_id="frame",
        object_id="object",
        track_id="track",
        u=10.0,
        v=10.0,
        x_m=xyz[0],
        y_m=xyz[1],
        z_m=xyz[2],
        depth_confidence=0.9,
        confidence=0.8,
        uncertainty=0.1,
        uncertainty_definition="synthetic",
        visibility="visible",
        valid=True,
        failure_reason="",
        coordinate_frame=CoordinateFrame.CAMERA_FRAME_METRIC,
        depth_unit="meter",
        depth_definition="z_depth",
        intrinsics_source="synthetic_calibrated",
        pose_source="unavailable_single_frame",
        provider_name="synthetic_metric",
        provenance={"sensor_ground_truth": False},
    )


def test_pixel_metric_3d_pixel_roundtrip() -> None:
    pixels = np.asarray([[50.0, 40.0], [20.0, 12.0], [88.0, 63.0]])
    depths = np.asarray([2.0, 3.5, 1.2])
    xyz = backproject_metric_arrays(pixels, depths, intrinsics())
    projected = project_metric_arrays(xyz, intrinsics())
    np.testing.assert_allclose(projected, pixels, atol=1e-10)


@pytest.mark.parametrize(
    ("transform", "source", "expected"),
    [
        (
            ImageCoordinateTransform(100, 80, 200, 160, 2.0, 2.0, operation="resize"),
            np.asarray([[20.0, 10.0]]),
            np.asarray([[40.0, 20.0]]),
        ),
        (
            ImageCoordinateTransform(
                100, 80, 50, 40, 1.0, 1.0, 10.0, 20.0, operation="crop"
            ),
            np.asarray([[10.0, 20.0]]),
            np.asarray([[0.0, 0.0]]),
        ),
        (
            ImageCoordinateTransform(
                100, 50, 200, 200, 2.0, 2.0, pad_y=50.0, operation="letterbox"
            ),
            np.asarray([[25.0, 10.0]]),
            np.asarray([[50.0, 70.0]]),
        ),
    ],
)
def test_resize_crop_letterbox_coordinate_consistency(
    transform: ImageCoordinateTransform,
    source: np.ndarray,
    expected: np.ndarray,
) -> None:
    target = transform.source_to_target(source)
    np.testing.assert_allclose(target, expected)
    np.testing.assert_allclose(transform.target_to_source(target), source)
    ray_source = backproject_metric_arrays(source, np.asarray([2.0]), intrinsics(100, transform.source_height))
    ray_target = backproject_metric_arrays(
        target, np.asarray([2.0]), transform.transform_intrinsics(intrinsics(100, transform.source_height))
    )
    np.testing.assert_allclose(ray_target, ray_source)


def test_mask_depth_different_resolution_alignment() -> None:
    mask = np.zeros((40, 50), dtype=bool)
    mask[10:30, 10:30] = True
    mask_transform = ImageCoordinateTransform(
        100, 80, 50, 40, 0.5, 0.5, operation="resize"
    )
    depth_transform = ImageCoordinateTransform.identity(100, 80)
    aligned = align_binary_mask(
        mask, mask_transform=mask_transform, target_transform=depth_transform
    )
    assert aligned.shape == (80, 100)
    assert 1500 <= np.count_nonzero(aligned) <= 1700


def test_ray_distance_is_converted_to_z_depth() -> None:
    K = intrinsics()
    rows, columns = np.indices((80, 100), dtype=float)
    z_true = np.full((80, 100), 2.0)
    ray_norm = np.sqrt(
        ((columns - K[0, 2]) / K[0, 0]) ** 2
        + ((rows - K[1, 2]) / K[1, 1]) ** 2
        + 1.0
    )
    converted = canonicalize_z_depth(z_true * ray_norm, K, depth_definition="ray_distance")
    np.testing.assert_allclose(converted, z_true)


def test_boundary_backprojection_and_foreground_background_depth() -> None:
    mask = np.zeros((80, 100), dtype=bool)
    mask[20:60, 30:70] = True
    depth = np.full(mask.shape, 4.0)
    depth[mask] = 2.0
    points = reconstruct_boundary_points(
        frame_id="frame",
        object_id="object",
        track_id="track",
        mask=mask,
        mask_quality=0.9,
        depth_map=depth,
        valid_mask=np.ones(mask.shape, dtype=bool),
        K=intrinsics(),
        confidence_map=np.ones(mask.shape),
        uncertainty_map=None,
        provider_name="synthetic",
        intrinsics_source="synthetic_calibrated",
        sample_count=16,
        side_radius_px=3,
    )
    assert points
    valid = [point for point in points if point.point.valid]
    assert valid
    assert all(point.point.point_type == MetricPointType.BOUNDARY_POINT for point in valid)
    assert any(point.background_depth_m == pytest.approx(4.0) for point in valid)
    assert any(point.boundary_depth_jump_m == pytest.approx(2.0) for point in valid)
    assert all(point.foreground_depth_m == pytest.approx(2.0) for point in valid)


def test_robust_mask_extent_rejects_single_depth_outlier() -> None:
    mask = np.zeros((80, 100), dtype=bool)
    mask[20:60, 30:70] = True
    depth = np.full(mask.shape, 2.0)
    depth[30, 40] = 100.0
    cloud = build_object_surface_pointcloud(
        frame_id="frame",
        object_id="object",
        track_id="track",
        class_name="box",
        mask=mask,
        mask_quality=0.9,
        depth_map=depth,
        valid_mask=np.ones(mask.shape, dtype=bool),
        K=intrinsics(),
        confidence_map=np.ones(mask.shape),
        uncertainty_map=np.full(mask.shape, 0.1),
        provider_name="synthetic",
        intrinsics_source="synthetic_calibrated",
    )
    assert cloud.valid
    assert cloud.z_extent_m == pytest.approx(0.0)
    assert cloud.robust_centroid_m[2] == pytest.approx(2.0)
    assert cloud.metadata["extent_method"] == "per_axis_robust_quantiles"


def test_internal_points_are_geometric_not_semantic() -> None:
    mask = np.zeros((80, 100), dtype=bool)
    mask[10:70, 15:85] = True
    rows, columns = np.indices(mask.shape)
    checker = (((rows // 8) + (columns // 8)) % 2 * 255).astype(np.uint8)
    image = cv2.cvtColor(checker, cv2.COLOR_GRAY2BGR)
    observation = InstanceMaskObservation.from_visible_mask(
        video_id="video",
        frame_index=0,
        object_track_id="track",
        semantic_label="generic_object",
        mask=mask,
        confidence=0.9,
        source_provider="synthetic_segmentation",
        metadata={"formal_mask_evidence": True, "legacy_bbox_fallback": False},
    )
    points = select_geometric_track_points(
        image=image,
        mask_observation=observation,
        depth_map=np.full(mask.shape, 2.0),
        valid_mask=np.ones(mask.shape, dtype=bool),
        K=intrinsics(),
        confidence_map=np.ones(mask.shape),
        uncertainty_map=None,
        frame_id="frame",
        object_id="object",
        provider_name="synthetic",
        intrinsics_source="synthetic_calibrated",
        erosion_pixels=4,
    )
    assert points
    assert all(point.point_type == MetricPointType.GEOMETRIC_TRACK_POINT for point in points)
    assert all(mask[int(round(point.v)), int(round(point.u))] for point in points)
    assert all(
        point.provenance["trackability_verified_across_frames"] is False
        for point in points
    )


def test_structure_graph_rejects_dense_surface_semantics() -> None:
    boundary = metric_point("boundary", MetricPointType.BOUNDARY_POINT, (0.0, 0.0, 2.0))
    internal = metric_point(
        "internal", MetricPointType.GEOMETRIC_TRACK_POINT, (0.1, 0.0, 2.0)
    )
    graph = build_single_frame_structure_graph(
        frame_id="frame",
        object_id="object",
        track_id="track",
        boundary_points=[boundary],
        internal_points=[internal],
        knn_k=1,
    )
    assert graph.valid
    assert graph.coordinate_frame == CoordinateFrame.CAMERA_FRAME_METRIC
    assert graph.edges[0].edge_length_m == pytest.approx(0.1)
    dense = metric_point(
        "dense", MetricPointType.DENSE_OBJECT_SURFACE_POINT, (0.2, 0.0, 2.0)
    )
    with pytest.raises(ValueError, match="Dense surface points"):
        build_single_frame_structure_graph(
            frame_id="frame",
            object_id="object",
            track_id="track",
            boundary_points=[boundary],
            internal_points=[dense],
        )


def test_world_frame_cannot_be_mislabeled_as_m2_output() -> None:
    with pytest.raises(ValueError, match="camera_frame_metric"):
        MetricSurfacePoint(
            **{
                **metric_point(
                    "point", MetricPointType.GEOMETRIC_TRACK_POINT, (0.0, 0.0, 2.0)
                ).__dict__,
                "coordinate_frame": CoordinateFrame.WORLD_FRAME,
            }
        )


def test_missing_depth_point_preserves_nan_not_zero() -> None:
    point = MetricSurfacePoint.missing(
        point_id="missing",
        point_type=MetricPointType.BOUNDARY_POINT,
        frame_id="frame",
        object_id="object",
        track_id="track",
        u=1.0,
        v=2.0,
        reason="invalid_depth",
    )
    assert not point.valid
    assert all(math.isnan(value) for value in (point.x_m, point.y_m, point.z_m))
    assert point.failure_reason == "invalid_depth"


def test_object_cloud_adapts_to_shared_metric_camera_contract() -> None:
    mask = np.zeros((80, 100), dtype=bool)
    mask[20:60, 30:70] = True
    cloud = build_object_surface_pointcloud(
        frame_id="frame",
        object_id="object",
        track_id="track",
        class_name="box",
        mask=mask,
        mask_quality=0.9,
        depth_map=np.full(mask.shape, 2.0),
        valid_mask=np.ones(mask.shape, dtype=bool),
        K=intrinsics(),
        confidence_map=np.ones(mask.shape),
        uncertainty_map=None,
        provider_name="synthetic",
        intrinsics_source="synthetic_calibrated",
        max_recorded_points=20,
    )
    shared = to_shared_object_observation(cloud, video_id="video", frame_index=0)
    assert shared.valid
    assert shared.center_3d_camera is not None
    assert (
        shared.center_3d_camera.coordinate_frame
        == CoordinateFrame.CAMERA_FRAME_METRIC.value
    )
    assert shared.metadata["world_frame_available"] is False
    assert shared.scale_descriptors["warning"].startswith("diagnostic extent diagonal")
