from __future__ import annotations

import math

import numpy as np
import pytest

from semantic3d.geometry.camera import CameraObservation, CoordinateConvention
from semantic3d.method_completion import (
    DimensionScalePrior,
    ExtentEstimator,
    MetricDepthDefinition,
    MetricDepthEvidence,
    MetricDepthType,
    MetricObjectRegion,
    MetricScaleStatus,
    MetricScaleThresholds,
    MetricSingleObjectScaleBranch,
    MultiIntervalScalePriorRegistry,
    ObjectPhysicalScalePrior,
    ProviderStatus,
    SizeInterval,
)


def camera(width: int = 640, height: int = 480, scale: float = 1.0, cx: float | None = None):
    K = np.array(
        [[500.0 * scale, 0.0, 320.0 * scale if cx is None else cx],
         [0.0, 500.0 * scale, 240.0 * scale],
         [0.0, 0.0, 1.0]],
        dtype=float,
    )
    return CameraObservation(
        K=K,
        distortion=None,
        T_world_camera=None,
        T_camera_world=None,
        image_width=width,
        image_height=height,
        coordinate_convention=CoordinateConvention.OPENCV,
        intrinsics_source="synthetic_calibrated",
        pose_source="missing",
        valid=True,
        quality=1.0,
    )


def registry() -> MultiIntervalScalePriorRegistry:
    return MultiIntervalScalePriorRegistry(
        {
            "box": ObjectPhysicalScalePrior(
                "box",
                {
                    "height_m": DimensionScalePrior("height_m", (SizeInterval(1.9, 2.1),)),
                    "width_m": DimensionScalePrior("width_m", (SizeInterval(0.9, 1.1),)),
                },
            )
        }
    )


def scene(*, z: float = 5.0, scale: int = 1, ray: bool = False, border=frozenset()):
    shape = (480 * scale, 640 * scale)
    mask = np.zeros(shape, dtype=bool)
    mask[100 * scale : 300 * scale, 200 * scale : 300 * scale] = True
    depth = np.full(shape, z, dtype=float)
    definition = MetricDepthDefinition.Z_DEPTH
    cam = camera(640 * scale, 480 * scale, float(scale))
    if ray:
        rows, columns = np.indices(shape)
        K = cam.K
        assert K is not None
        norm = np.sqrt(
            ((columns - K[0, 2]) / K[0, 0]) ** 2
            + ((rows - K[1, 2]) / K[1, 1]) ** 2
            + 1.0
        )
        depth = z * norm
        definition = MetricDepthDefinition.RAY_DISTANCE
    obj = MetricObjectRegion(
        "video", "clip", "frame", "object", "track", "box",
        (200 * scale, 100 * scale, 300 * scale, 300 * scale), shape,
        mask=mask, border_contacts=frozenset(border),
    )
    dep = MetricDepthEvidence(
        depth, np.ones(shape, bool), np.ones(shape), MetricDepthType.METRIC,
        "meter", MetricScaleStatus.MODEL_PREDICTED, definition, "synthetic",
        ProviderStatus.OK, 1.0,
    )
    return obj, dep, cam


def branch(**kwargs):
    return MetricSingleObjectScaleBranch(
        registry(), estimator=ExtentEstimator.PROJECTED_EXTENT,
        thresholds=MetricScaleThresholds(allow_approximated_intrinsics=False), **kwargs
    )


def test_known_pinhole_height_and_width_recover_metric_size():
    result = branch().evaluate(*scene())
    assert result.evidence.valid
    assert result.estimated_dimensions_m["height_m"] == pytest.approx(2.0)
    assert result.estimated_dimensions_m["width_m"] == pytest.approx(1.0)
    assert result.evidence.residual_value == pytest.approx(0.0)


def test_same_object_at_different_depth_recovers_same_size():
    near = scene(z=5.0)
    far_obj, far_depth, far_camera = scene(z=10.0)
    far_mask = np.zeros(far_obj.image_shape, bool)
    far_mask[100:200, 200:250] = True
    far_obj = MetricObjectRegion(
        far_obj.video_id, far_obj.clip_id, far_obj.frame_id, far_obj.object_id,
        far_obj.track_id, far_obj.class_name, (200, 100, 250, 200), far_obj.image_shape,
        mask=far_mask,
    )
    near_result = branch().evaluate(*near)
    far_result = branch().evaluate(far_obj, far_depth, far_camera)
    assert near_result.estimated_dimensions_m["height_m"] == pytest.approx(
        far_result.estimated_dimensions_m["height_m"]
    )


def test_metric_depth_and_focal_length_scale_formula():
    obj, depth, cam = scene(z=5.0)
    result = branch().evaluate(obj, depth, cam)
    depth2 = MetricDepthEvidence(
        depth.depth_map * 1.5, depth.valid_mask, depth.confidence_map,
        depth.depth_type, depth.depth_unit, depth.scale_status, depth.depth_definition,
        depth.provider_name, depth.provider_status, depth.quality,
    )
    result_depth = branch().evaluate(obj, depth2, cam)
    K2 = cam.K.copy()
    K2[0, 0] *= 2
    K2[1, 1] *= 2
    cam2 = CameraObservation(
        K2, None, None, None, 640, 480, CoordinateConvention.OPENCV,
        "synthetic_calibrated", "missing", True, 1.0,
    )
    result_focal = branch().evaluate(obj, depth, cam2)
    assert result_depth.extent.y_extent_m == pytest.approx(result.extent.y_extent_m * 1.5)
    assert result_focal.extent.y_extent_m == pytest.approx(result.extent.y_extent_m / 2)


def test_resize_intrinsics_and_ray_distance_preserve_size():
    normal = branch().evaluate(*scene(scale=1))
    resized = branch().evaluate(*scene(scale=2))
    ray = branch().evaluate(*scene(scale=1, ray=True))
    assert resized.extent.y_extent_m == pytest.approx(normal.extent.y_extent_m)
    assert ray.extent.y_extent_m == pytest.approx(normal.extent.y_extent_m, rel=1e-6)


def test_crop_principal_point_update_preserves_pointcloud_extent():
    obj, depth, cam = scene()
    point_branch = MetricSingleObjectScaleBranch(
        registry(), estimator=ExtentEstimator.MASK_POINTCLOUD_EXTENT,
        thresholds=MetricScaleThresholds(min_point_count=10),
    )
    original = point_branch.evaluate(obj, depth, cam)
    crop_mask = obj.mask[:, 100:]
    crop_obj = MetricObjectRegion(
        "video", "clip", "crop", "object", "track", "box", (100, 100, 200, 300),
        crop_mask.shape, mask=crop_mask,
    )
    crop_depth = MetricDepthEvidence(
        depth.depth_map[:, 100:], depth.valid_mask[:, 100:], depth.confidence_map[:, 100:],
        depth.depth_type, depth.depth_unit, depth.scale_status, depth.depth_definition,
        depth.provider_name, depth.provider_status, depth.quality,
    )
    crop_cam = camera(width=540, height=480, cx=220.0)
    cropped = point_branch.evaluate(crop_obj, crop_depth, crop_cam)
    assert cropped.extent.x_extent_m == pytest.approx(original.extent.x_extent_m)
    assert cropped.extent.y_extent_m == pytest.approx(original.extent.y_extent_m)


@pytest.mark.parametrize(
    ("depth_type", "unit", "definition", "reason"),
    [
        (MetricDepthType.RELATIVE, "relative", MetricDepthDefinition.Z_DEPTH, "metric_scale_unavailable"),
        (MetricDepthType.METRIC, "meter", MetricDepthDefinition.UNKNOWN, "unknown_depth_definition"),
        (MetricDepthType.METRIC, "relative", MetricDepthDefinition.Z_DEPTH, "metric_depth_unit_not_meter"),
    ],
)
def test_non_metric_or_ambiguous_depth_is_blocked(depth_type, unit, definition, reason):
    obj, depth, cam = scene()
    changed = MetricDepthEvidence(
        depth.depth_map, depth.valid_mask, depth.confidence_map, depth_type, unit,
        depth.scale_status, definition, depth.provider_name, depth.provider_status, depth.quality,
    )
    result = branch().evaluate(obj, changed, cam)
    assert not result.evidence.valid
    assert math.isnan(result.evidence.residual_value)
    assert result.evidence.failure_reason == reason


def test_dimension_observability_is_independent():
    obj, depth, cam = scene(border={"left"})
    result = branch().evaluate(obj, depth, cam)
    assert result.observability.height_observable
    assert not result.observability.width_observable
    assert result.evidence.valid
    assert "height_m" in result.dimension_residuals
    assert "width_m" not in result.dimension_residuals


def test_severe_truncation_and_provider_failure_are_nan_not_anomaly():
    obj, depth, cam = scene()
    truncated = MetricObjectRegion(
        obj.video_id, obj.clip_id, obj.frame_id, obj.object_id, obj.track_id,
        obj.class_name, obj.bbox, obj.image_shape, mask=obj.mask, severe_truncation=True,
    )
    result = branch().evaluate(truncated, depth, cam)
    assert not result.evidence.valid and math.isnan(result.evidence.residual_value)
    failed_depth = MetricDepthEvidence(
        depth.depth_map, depth.valid_mask, depth.confidence_map, depth.depth_type,
        depth.depth_unit, depth.scale_status, depth.depth_definition, depth.provider_name,
        ProviderStatus.PROVIDER_FAILED, 0.0,
    )
    failed = branch().evaluate(obj, failed_depth, cam)
    assert failed.evidence.provider_status == ProviderStatus.PROVIDER_FAILED
    assert math.isnan(failed.evidence.residual_value)
