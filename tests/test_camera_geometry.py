from __future__ import annotations

import numpy as np
import pytest

from semantic3d.geometry.camera import (
    CameraObservation,
    CoordinateConvention,
    DepthDefinition,
    PixelCenterConvention,
    TransformConvention,
)

from synthetic_geometry import synthetic_intrinsics, synthetic_pose


def test_camera_records_one_consistent_coordinate_convention() -> None:
    camera = CameraObservation.from_parameters(
        K=synthetic_intrinsics(),
        image_width=640,
        image_height=480,
        intrinsics_source="synthetic_ground_truth",
        quality=1.0,
    )
    assert camera.coordinate_convention == CoordinateConvention.OPENCV
    assert camera.depth_definition == DepthDefinition.Z_DEPTH
    assert camera.transform_convention == TransformConvention.COLUMN_VECTOR
    assert camera.pixel_center_convention == PixelCenterConvention.INTEGER_CENTERS


def test_one_pose_direction_computes_the_inverse() -> None:
    world_from_camera = synthetic_pose(translation=(1.0, -2.0, 3.0))
    camera = CameraObservation.from_parameters(
        K=synthetic_intrinsics(),
        image_width=640,
        image_height=480,
        intrinsics_source="synthetic_ground_truth",
        quality=1.0,
        T_world_from_camera=world_from_camera,
        pose_source="synthetic_ground_truth",
    )
    assert camera.pose_valid
    np.testing.assert_allclose(
        camera.T_camera_from_world, np.linalg.inv(world_from_camera), atol=1e-12
    )


def test_non_inverse_pose_pair_is_explicitly_invalid() -> None:
    camera = CameraObservation(
        K=synthetic_intrinsics(),
        distortion=None,
        T_world_camera=synthetic_pose(translation=(1.0, 0.0, 0.0)),
        T_camera_world=synthetic_pose(translation=(2.0, 0.0, 0.0)),
        image_width=640,
        image_height=480,
        coordinate_convention=CoordinateConvention.OPENCV,
        intrinsics_source="synthetic_ground_truth",
        pose_source="bad_test_pose",
        valid=True,
        quality=1.0,
    )
    assert not camera.valid
    assert camera.missing_reason == "inconsistent_camera_transforms"


def test_approximate_intrinsics_are_non_unit_quality_and_pose_optional() -> None:
    camera = CameraObservation.from_parameters(
        K=synthetic_intrinsics(),
        image_width=640,
        image_height=480,
        intrinsics_source="approximate",
        quality=0.5,
    )
    assert camera.valid
    assert not camera.pose_valid
    assert camera.T_world_from_camera is None

    with pytest.raises(ValueError, match="quality < 1"):
        CameraObservation.from_parameters(
            K=synthetic_intrinsics(),
            image_width=640,
            image_height=480,
            intrinsics_source="approximate",
            quality=1.0,
        )


def test_incompatible_coordinate_convention_is_rejected() -> None:
    with pytest.raises(ValueError, match="OpenCV"):
        CameraObservation(
            K=synthetic_intrinsics(),
            distortion=None,
            T_world_camera=None,
            T_camera_world=None,
            image_width=640,
            image_height=480,
            coordinate_convention=CoordinateConvention.UNKNOWN,
            intrinsics_source="synthetic_ground_truth",
            pose_source="",
            valid=True,
            quality=1.0,
        )
