from __future__ import annotations

from _bootstrap import ensure_project_test_environment

ensure_project_test_environment(__file__)

import numpy as np
import pytest

from semantic3d.aggregation import (
    aggregate_map_by_mask,
    aggregate_points_by_mask,
    topk_mean,
)
from semantic3d.multilevel_residuals import (
    ObjectMaskObservation,
    build_object_level_residuals,
    build_object_level_residuals_with_details,
    build_object_pair_residuals,
    summarize_clip_residuals,
)


def test_topk_mean() -> None:
    values = np.array([1.0, 2.0, 3.0, 100.0])

    assert topk_mean(values, k_ratio=0.5) == pytest.approx(51.5)
    assert topk_mean(np.array([]), empty_policy="zero") == pytest.approx(0.0)
    with pytest.raises(ValueError, match="empty"):
        topk_mean(np.array([]), empty_policy="raise")


def test_aggregate_map_by_mask() -> None:
    residual_map = np.arange(9, dtype=float).reshape(3, 3)
    mask = np.array([[0, 1, 1], [0, 0, 0], [0, 0, 0]], dtype=bool)

    assert aggregate_map_by_mask(residual_map, mask, method="mean") == pytest.approx(1.5)
    assert aggregate_map_by_mask(residual_map, mask, method="max") == pytest.approx(2.0)


def test_aggregate_points_by_mask() -> None:
    mask = np.zeros((5, 5), dtype=bool)
    mask[1:4, 1:4] = True
    points = np.array([[1, 1], [3, 3], [4, 4], [-1, 2]], dtype=float)
    residuals = np.array([0.2, 0.8, 10.0, 20.0], dtype=float)

    assert aggregate_points_by_mask(points, residuals, mask, method="mean") == pytest.approx(
        0.5
    )


def test_build_object_level_residuals_from_maps_and_tracks() -> None:
    mask_a = np.zeros((6, 6), dtype=bool)
    mask_b = np.zeros((6, 6), dtype=bool)
    mask_a[1:3, 1:3] = True
    mask_b[3:5, 3:5] = True
    objects = [
        ObjectMaskObservation("a", "soccer_ball", mask_a),
        ObjectMaskObservation("b", "elephant", mask_b),
    ]
    flow = np.zeros((6, 6), dtype=float)
    depth = np.zeros((6, 6), dtype=float)
    corr = np.zeros((6, 6), dtype=float)
    flow[mask_b] = 1.0
    depth[mask_b] = 2.0
    corr[mask_a] = 0.5
    points = np.array([[1, 1], [4, 4], [0, 0]], dtype=float)
    point_residuals = np.array([0.1, 0.9, 5.0], dtype=float)

    residuals = build_object_level_residuals(
        objects,
        flow_residual_map=flow,
        depth_residual_map=depth,
        corr_residual_map=corr,
        track_points_xy=points,
        track_residuals=point_residuals,
        aggregation_method="mean",
    )

    assert residuals[0].corr == pytest.approx(0.5)
    assert residuals[0].track == pytest.approx(0.1)
    assert residuals[1].flow == pytest.approx(1.0)
    assert residuals[1].depth_cons == pytest.approx(2.0)
    assert residuals[1].track == pytest.approx(0.9)


def test_missing_residual_sources_are_zero_and_reported() -> None:
    mask = np.ones((4, 4), dtype=bool)
    objects = [ObjectMaskObservation("a", "soccer_ball", mask)]

    residuals, details = build_object_level_residuals_with_details(objects)

    assert residuals[0].flow == pytest.approx(0.0)
    assert residuals[0].track == pytest.approx(0.0)
    assert "flow_residual_map" in details["missing_sources"]
    assert "track_points_xy/track_residuals" in details["missing_sources"]


def test_build_object_pair_residuals_skips_diagonal_and_duplicates() -> None:
    mask = np.ones((4, 4), dtype=bool)
    objects = [
        ObjectMaskObservation("a", "soccer_ball", mask),
        ObjectMaskObservation("b", "elephant", mask),
        ObjectMaskObservation("c", "person", mask),
    ]
    scale_depth = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]])

    pairs = build_object_pair_residuals(objects, scale_depth_matrix=scale_depth)

    assert len(pairs) == 3
    assert [(p.object_id_a, p.object_id_b) for p in pairs] == [
        ("a", "b"),
        ("a", "c"),
        ("b", "c"),
    ]
    assert [p.scale_depth for p in pairs] == [1.0, 2.0, 3.0]


def test_summarize_clip_residuals_keeps_details() -> None:
    mask = np.ones((4, 4), dtype=bool)
    objects = [
        ObjectMaskObservation("a", "soccer_ball", mask),
        ObjectMaskObservation("b", "elephant", mask),
    ]
    object_residuals = build_object_level_residuals(objects)
    pair_residuals = build_object_pair_residuals(
        objects,
        scale_depth_matrix=np.array([[0.0, 2.0], [2.0, 0.0]]),
    )

    summary = summarize_clip_residuals(
        object_residuals,
        pair_residuals,
        aggregation_method="max",
    )

    assert summary.clip_score == pytest.approx(2.0)
    assert summary.details["pair_scores"]["a->b"] == pytest.approx(2.0)
    assert summary.details["aggregation_method"] == "max"
