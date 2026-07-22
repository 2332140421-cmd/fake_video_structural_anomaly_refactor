from __future__ import annotations

from semantic3d.occlusion import ObjectVisibilityObservation, VisibilityState
from semantic3d.occlusion.scene_cut_statistics import compute_scene_cut_statistics


def _state(track: str, frame: int, state: str) -> ObjectVisibilityObservation:
    invalid = state == "scene_cut"
    return ObjectVisibilityObservation(track, frame, "uncertain", state, 0, 0, 0, 0, 1, (), 0 if invalid else 1, not invalid, "scene_cut_breaks_visibility_history" if invalid else "")


def test_clip_scene_cut_is_unique_and_initialization_is_not_real_cut() -> None:
    observations = [_state(track, 0, "scene_cut") for track in ("a", "b", "c")]
    observations += [_state(track, 3, "scene_cut") for track in ("a", "b", "c")]
    stats = compute_scene_cut_statistics(frame_indices=range(5), scene_cut_flags={0: False, 3: True}, visibility_observations=observations)
    assert stats.clip_scene_cut_count == 1
    assert stats.object_visibility_scene_cut_markers == 3
    assert stats.track_initialization_markers == 3
    assert stats.state_machine_boundary_markers == 6
    assert stats.legacy_scene_cut_state_count == 6
