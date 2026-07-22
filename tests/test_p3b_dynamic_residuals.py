from __future__ import annotations

import math

import numpy as np

from semantic3d.dynamic_3d import (
    DynamicGeometryMode,
    ObjectStructureGraph,
    PointTrack3DObservation,
    StructureEdge,
    compute_direction_consistency_residuals,
    compute_relative_velocity_residuals,
    compute_structure_temporal_residuals,
)
from semantic3d.sequence_geometry import SequenceScaleStatus


def _point(point_id: str, frame: int, xyz, *, mode=DynamicGeometryMode.STATIC_CAMERA_3D, object_id="obj") -> PointTrack3DObservation:
    xyz = tuple(float(value) for value in xyz)
    return PointTrack3DObservation(
        point_id, object_id, frame, (10.0, 10.0), max(xyz[2], 0.1), xyz, None,
        "visible", "visible", 1.0, 1.0, 1.0, "synthetic_independent",
        SequenceScaleStatus.RELATIVE_ALIGNED_SEQUENCE, mode, True,
        metadata={"independent_observation": True},
    )


def _rigid_points(positions_by_frame):
    rows = []
    for frame, origin in enumerate(positions_by_frame):
        origin = np.asarray(origin, dtype=float)
        rows.extend([
            _point("p0", frame, origin + [0.0, 0.0, 5.0]),
            _point("p1", frame, origin + [1.0, 0.0, 5.0]),
            _point("p2", frame, origin + [0.0, 1.0, 5.0]),
        ])
    return rows


def _graph(label="car") -> ObjectStructureGraph:
    edges = (
        StructureEdge("p0", "p1", 1.0, 1.0, 1.0, True),
        StructureEdge("p0", "p2", 1.0, 1.0, 1.0, True),
    )
    return ObjectStructureGraph("obj", label, 0, ("p0", "p1", "p2"), edges, "fixed_knn", DynamicGeometryMode.STATIC_CAMERA_3D, True, 1.0)


def test_rigid_constant_velocity_direction_and_speed_change_are_zero() -> None:
    points = _rigid_points([(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)])
    directions = compute_direction_consistency_residuals(points)
    assert directions and max(item.own_history.value for item in directions if item.own_history.valid) < 1e-9
    scales = {"obj": {frame: 1.0 for frame in range(4)}}
    speeds = compute_relative_velocity_residuals(points, scales)
    assert max(item.speed_change_residual.value for item in speeds if item.speed_change_residual.valid) < 1e-9


def test_sudden_direction_change_increases_direction_residual() -> None:
    points = _rigid_points([(0, 0, 0), (1, 0, 0), (1, 1, 0)])
    results = compute_direction_consistency_residuals(points)
    assert min(item.own_history.value for item in results if item.own_history.valid) > 0.9


def test_whole_object_acceleration_changes_speed_not_structure() -> None:
    points = _rigid_points([(0, 0, 0), (1, 0, 0), (3, 0, 0)])
    scales = {"obj": {frame: 1.0 for frame in range(3)}}
    speed = compute_relative_velocity_residuals(points, scales)
    assert min(item.speed_change_residual.value for item in speed if item.speed_change_residual.valid) > 0.9
    structure = compute_structure_temporal_residuals(_graph(), points, scales["obj"])
    assert max(item.object_structure_residual.value for item in structure if item.valid) < 1e-9


def test_single_point_jump_localizes_point_and_adjacent_edges() -> None:
    points = _rigid_points([(0, 0, 0), (0, 0, 0), (0, 0, 0)])
    points = [
        _point(item.point_id, item.frame_index, (3.0, 0.0, 5.0))
        if item.point_id == "p0" and item.frame_index == 2 else item
        for item in points
    ]
    result = compute_structure_temporal_residuals(_graph(), points, {0: 1.0, 1: 1.0, 2: 1.0})[-1]
    assert result.object_structure_residual.value > 1.0
    assert "p0" in result.anomalous_point_ids
    assert any("p0" in edge_id for edge_id in result.anomalous_edge_ids)


def test_human_bone_stable_motion_low_and_stretch_high() -> None:
    graph = ObjectStructureGraph(
        "obj", "person", 0, ("p0", "p1"),
        (StructureEdge("p0", "p1", 1.0, 1.0, 1.0, True),),
        "semantic_human_skeleton", DynamicGeometryMode.STATIC_CAMERA_3D, True, 1.0,
    )
    stable = [_point("p0", 0, (0, 0, 5)), _point("p1", 0, (1, 0, 5)), _point("p0", 1, (0, 1, 5)), _point("p1", 1, (1, 1, 5))]
    stretched = stable + [_point("p0", 2, (0, 2, 5)), _point("p1", 2, (2, 2, 5))]
    stable_value = compute_structure_temporal_residuals(graph, stable, {0: 1.0, 1: 1.0})[-1].object_structure_residual.value
    stretch_value = compute_structure_temporal_residuals(graph, stretched, {0: 1.0, 1: 1.0, 2: 1.0})[-1].object_structure_residual.value
    assert stable_value < 1e-9 and stretch_value > 0.9


def test_rotation_only_has_no_complete_3d_speed_and_unavailable_is_nan() -> None:
    rotation_points = [
        _point("p", frame, (0.1 * frame, 0, 5), mode=DynamicGeometryMode.ROTATION_COMPENSATED)
        for frame in range(3)
    ]
    speeds = compute_relative_velocity_residuals(rotation_points, {"obj": {0: 1.0, 1: 1.0, 2: 1.0}})
    assert speeds and all(not item.valid and math.isnan(item.normalized_relative_speed) for item in speeds)
    assert {item.missing_reason for item in speeds} == {"rotation_only_no_complete_3d_speed"}


def test_unavailable_geometry_produces_only_nan_formal_evidence() -> None:
    points = [
        _point("p", frame, (0.1 * frame, 0, 5), mode=DynamicGeometryMode.UNAVAILABLE)
        for frame in range(3)
    ]
    directions = compute_direction_consistency_residuals(points)
    assert directions and all(not item.valid for item in directions)
    assert all(math.isnan(item.own_history.value) for item in directions)
    speeds = compute_relative_velocity_residuals(points, {"obj": {0: 1.0, 1: 1.0, 2: 1.0}})
    assert speeds and all(not item.valid for item in speeds)
    assert all(math.isnan(item.speed_change_residual.value) for item in speeds)


def test_missing_structure_edge_is_nan_not_zero() -> None:
    points = [_point("p0", 0, (0, 0, 5)), _point("p1", 0, (1, 0, 5)), _point("p0", 1, (0, 0, 5))]
    result = compute_structure_temporal_residuals(_graph(), points, {0: 1.0, 1: 1.0})[-1]
    missing = next(row for row in result.edge_residuals if row.point_id_b == "p1")
    assert not missing.valid
    assert math.isnan(missing.normalized_edge_length_change.value)
