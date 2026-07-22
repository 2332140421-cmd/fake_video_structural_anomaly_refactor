from __future__ import annotations

import math

from semantic3d.dynamic_3d import (
    PointTrack2DObservation,
    bind_point_tracks_to_objects,
)
from semantic3d.observations import FrameObservationJSON, ObjectObservationJSON


def _point(point_id: str, frame: int, uv: tuple[float, float]) -> PointTrack2DObservation:
    return PointTrack2DObservation(
        point_id=point_id,
        object_track_id="unassigned",
        frame_index=frame,
        pixel_uv=uv,
        visibility="visible",
        occlusion_status="visible",
        tracking_confidence=1.0,
        source_tracker="synthetic_independent",
        valid=True,
        metadata={"independent_observation": True, "generated_from_projection": False},
    )


def _frame(frame: int, track_id: str = "obj_a", bbox=(10.0, 10.0, 60.0, 60.0)) -> FrameObservationJSON:
    return FrameObservationJSON(
        frame_index=frame,
        frame_id=f"f{frame}",
        width=100,
        height=100,
        objects=[ObjectObservationJSON(
            object_id=f"det_{frame}", label="car", mask_area=2500.0,
            frame_area=10000.0, depth=5.0, confidence=1.0,
            bbox=list(bbox), track_id=track_id,
        )],
    )


def test_synthetic_object_assignment_uses_stable_track_and_shrunk_bbox() -> None:
    result = bind_point_tracks_to_objects(
        [_point("p", frame, (30.0 + frame, 30.0)) for frame in range(4)],
        [_frame(frame) for frame in range(4)],
        video_id="video", clip_id="clip",
    )
    assert result.bindings[0].valid
    assert result.bindings[0].object_track_id == "obj_a"
    assert result.bindings[0].assignment_source == "shrunk_bbox"
    assert result.bindings[0].track_consistency_ratio == 1.0
    assert {point.object_track_id for point in result.points_2d} == {"obj_a"}


def test_object_assignment_switch_invalidates_continuity() -> None:
    result = bind_point_tracks_to_objects(
        [_point("p", 0, (30.0, 30.0)), _point("p", 1, (30.0, 30.0))],
        [_frame(0, "obj_a"), _frame(1, "obj_b")],
        video_id="video", clip_id="clip",
    )
    assert not result.bindings[0].valid
    assert result.bindings[0].missing_reason == "object_assignment_switched"
    assert result.statistics["assignment_switch_count"] == 1
    assert all(not point.valid for point in result.points_2d)


def test_point_leaving_bbox_is_assignment_lost_not_zero_evidence() -> None:
    result = bind_point_tracks_to_objects(
        [_point("p", 0, (30.0, 30.0)), _point("p", 1, (90.0, 90.0)), _point("p", 2, (31.0, 30.0))],
        [_frame(frame) for frame in range(3)],
        video_id="video", clip_id="clip", minimum_consistency_ratio=0.5,
    )
    lost = next(point for point in result.points_2d if point.frame_index == 1)
    assert not lost.valid and lost.missing_reason == "assignment_lost"
    assert lost.pixel_uv is None
    assert result.statistics["assignment_lost_count"] == 1


def test_background_point_is_explicit() -> None:
    result = bind_point_tracks_to_objects(
        [_point("bg", frame, (90.0, 90.0)) for frame in range(3)],
        [_frame(frame) for frame in range(3)],
        video_id="video", clip_id="clip",
    )
    assert result.bindings[0].point_role.value == "background_point"
    assert result.bindings[0].object_track_id == "background"
    assert result.statistics["background_bindings"] == 1
