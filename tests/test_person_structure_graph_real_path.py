from __future__ import annotations

from semantic3d.dynamic_3d import bind_person_keypoints_to_shared_3d, build_object_structure_graph
from semantic3d.keypoint_provider import Keypoint2D, MockKeypointProvider
from semantic3d.observations import FrameObservationJSON, ObjectObservationJSON
from synthetic_dynamic_3d import make_synthetic_dynamic_scene


def test_fixed_human_skeleton_uses_available_edges_only() -> None:
    scene = make_synthetic_dynamic_scene(camera_centers=[(0, 0, 0)] * 3, world_points={"unused": [(0, 0, 5)] * 3}, mode="static_camera_3d")
    frames = tuple(FrameObservationJSON(i, f"f{i}", 160, 120, [ObjectObservationJSON("p", "person", 1000, 19200, 5, bbox=[30, 20, 130, 115], track_id="pt")], image_path=f"unused_{i}.png") for i in range(3))
    provider = MockKeypointProvider((
        Keypoint2D("left_shoulder", 60, 40, 1, True, "mock"),
        Keypoint2D("right_shoulder", 100, 40, 1, True, "mock"),
        Keypoint2D("right_elbow", 105, 60, 1, True, "mock"),
        Keypoint2D("left_elbow", 55, 60, 0, False, "mock"),
    ))
    result = bind_person_keypoints_to_shared_3d(video_id="v", clip_id="c", frames=frames, provider=provider, shared_clip=scene.clip, readiness=scene.readiness)
    graph = build_object_structure_graph(result.object_tracks, object_track_id="pt", semantic_label="person")
    assert graph.valid and graph.graph_type == "semantic_human_skeleton"
    edges = {(edge.point_id_a, edge.point_id_b) for edge in graph.edges}
    assert any("right_elbow" in endpoint for edge in edges for endpoint in edge)
    assert not any("left_elbow" in endpoint for edge in edges for endpoint in edge)
