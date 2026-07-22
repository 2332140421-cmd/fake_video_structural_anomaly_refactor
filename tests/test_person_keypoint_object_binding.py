from __future__ import annotations

from semantic3d.dynamic_3d import bind_person_keypoints_to_shared_3d
from semantic3d.keypoint_provider import Keypoint2D, MockKeypointProvider
from semantic3d.observations import FrameObservationJSON, ObjectObservationJSON
from synthetic_dynamic_3d import make_synthetic_dynamic_scene


def _frames(count: int):
    return tuple(FrameObservationJSON(
        index, f"f{index}", 160, 120,
        [ObjectObservationJSON("person_det", "person", 1600, 19200, 5, bbox=[40, 20, 120, 110], track_id="person_track")],
        image_path=f"unused_{index}.png",
    ) for index in range(count))


def test_keypoints_bind_to_person_track_with_stable_left_right_ids() -> None:
    scene = make_synthetic_dynamic_scene(camera_centers=[(0, 0, 0)] * 3, world_points={"unused": [(0, 0, 5)] * 3}, mode="static_camera_3d")
    provider = MockKeypointProvider((
        Keypoint2D("left_shoulder", 60, 40, 0.9, True, "mock"),
        Keypoint2D("right_shoulder", 100, 40, 0.9, True, "mock"),
        Keypoint2D("left_elbow", 55, 60, 0.1, False, "mock"),
    ))
    result = bind_person_keypoints_to_shared_3d(video_id="v", clip_id="c", frames=_frames(3), provider=provider, shared_clip=scene.clip, readiness=scene.readiness)
    valid = [point for point in result.points_2d if point.valid]
    assert valid and {point.object_track_id for point in valid} == {"person_track"}
    assert {point.point_id for point in valid} == {"person_track:keypoint:left_shoulder", "person_track:keypoint:right_shoulder"}
    missing_elbows = [point for point in result.points_2d if point.point_id.endswith("left_elbow")]
    assert missing_elbows and all(not point.valid for point in missing_elbows)
