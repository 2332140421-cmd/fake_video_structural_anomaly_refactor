from __future__ import annotations

import numpy as np
import pytest

from semantic3d.geometry.camera import CameraObservation, CoordinateConvention
from semantic3d.method_completion import (
    MetricDepthDefinition,
    MetricDepthEvidence,
    MetricDepthType,
    MetricObjectRegion,
    MetricScaleStatus,
    ProviderStatus,
    estimate_mask_pointcloud_extent,
)


def test_mask_pointcloud_extent_uses_robust_quantiles():
    shape = (100, 120)
    mask = np.zeros(shape, bool)
    mask[20:80, 30:90] = True
    depth = np.full(shape, 5.0)
    depth[20, 30] = 500.0
    observation = MetricDepthEvidence(
        depth, np.ones(shape, bool), np.ones(shape), MetricDepthType.METRIC,
        "meter", MetricScaleStatus.MODEL_PREDICTED, MetricDepthDefinition.Z_DEPTH,
        "synthetic", ProviderStatus.OK, 1.0,
    )
    obj = MetricObjectRegion(
        "v", "c", "f", "o", "t", "box", (30, 20, 90, 80), shape, mask=mask
    )
    camera = CameraObservation(
        np.array([[200.0, 0.0, 60.0], [0.0, 200.0, 50.0], [0.0, 0.0, 1.0]]),
        None, None, None, 120, 100, CoordinateConvention.OPENCV,
        "synthetic_calibrated", "missing", True, 1.0,
    )
    robust = estimate_mask_pointcloud_extent(obj, observation, camera, quantile_low=0.05, quantile_high=0.95)
    raw_z_range = float(depth[mask].max() - depth[mask].min())
    assert robust.valid
    assert robust.point_count == int(mask.sum())
    assert robust.z_extent_m < raw_z_range * 0.01
    assert "quantile" in robust.extent_estimator


def test_formal_mask_is_required_for_pointcloud_mode():
    shape = (10, 10)
    depth = MetricDepthEvidence(
        np.ones(shape), np.ones(shape, bool), None, MetricDepthType.METRIC, "meter",
        MetricScaleStatus.MODEL_PREDICTED, MetricDepthDefinition.Z_DEPTH,
        "synthetic", ProviderStatus.OK, 1.0,
    )
    obj = MetricObjectRegion("v", "c", "f", "o", "t", "box", (1, 1, 5, 5), shape)
    camera = CameraObservation(
        np.array([[10.0, 0, 5], [0, 10.0, 5], [0, 0, 1]]), None, None, None,
        10, 10, CoordinateConvention.OPENCV, "calibrated", "missing", True, 1.0,
    )
    result = estimate_mask_pointcloud_extent(obj, depth, camera)
    assert not result.valid
    assert result.failure_reason == "formal_mask_required"
