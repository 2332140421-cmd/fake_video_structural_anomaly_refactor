import numpy as np

from models.geometry import (
    backproject_points,
    predict_target_positions,
    project_points,
    transform_points,
)


def test_metric_backprojection_transform_and_reprojection():
    intrinsics = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]])
    pixels = np.array([[50.0, 40.0], [60.0, 40.0]])
    points = backproject_points(pixels, np.array([2.0, 2.0]), intrinsics)
    np.testing.assert_allclose(points, [[0.0, 0.0, 2.0], [0.2, 0.0, 2.0]])
    np.testing.assert_allclose(project_points(points, intrinsics), pixels)
    transform = np.eye(4)
    transform[0, 3] = 0.1
    moved = transform_points(points, transform)
    np.testing.assert_allclose(moved[:, 0], points[:, 0] + 0.1)
    predicted = predict_target_positions(points, transform, intrinsics)
    np.testing.assert_allclose(predicted[:, 0], pixels[:, 0] + 5.0)


def test_normal_prediction_is_small_and_manual_offset_is_detected():
    intrinsics = np.array([[80.0, 0.0, 32.0], [0.0, 80.0, 24.0], [0.0, 0.0, 1.0]])
    point = np.array([[0.0, 0.0, 2.0]])
    predicted = predict_target_positions(point, np.eye(4), intrinsics)[0]
    normal = np.linalg.norm(predicted - np.array([32.0, 24.0]))
    shifted = np.linalg.norm(predicted - np.array([42.0, 24.0]))
    assert normal < 1e-8
    assert shifted == 10.0
