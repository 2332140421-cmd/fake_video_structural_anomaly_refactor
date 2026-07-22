"""Versioned scene-cut and state-machine boundary statistics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .visibility_state import ObjectVisibilityObservation, VisibilityState


@dataclass(frozen=True)
class SceneCutStatistics:
    """Separate physical clip cuts from per-object state markers."""

    statistics_version: str
    clip_scene_cut_count: int
    clip_scene_cut_frame_indices: tuple[int, ...]
    object_visibility_scene_cut_markers: int
    track_initialization_markers: int
    state_machine_boundary_markers: int
    legacy_scene_cut_state_count: int


def compute_scene_cut_statistics(
    *,
    frame_indices: Sequence[int],
    scene_cut_flags: Mapping[int, bool],
    visibility_observations: Sequence[ObjectVisibilityObservation],
) -> SceneCutStatistics:
    """Count each real frame boundary once and initialization per object."""

    ordered = tuple(sorted(set(int(value) for value in frame_indices)))
    first = ordered[0] if ordered else None
    actual_cuts = tuple(
        index for index in ordered
        if index != first and bool(scene_cut_flags.get(index, False))
    )
    legacy = sum(
        item.current_state == VisibilityState.SCENE_CUT
        for item in visibility_observations
    )
    actual_markers = sum(
        item.current_state == VisibilityState.SCENE_CUT
        and item.frame_index in actual_cuts
        for item in visibility_observations
    )
    initialization = sum(
        item.frame_index == first
        for item in visibility_observations
    ) if first is not None else 0
    return SceneCutStatistics(
        statistics_version="scene_cut_statistics_v2",
        clip_scene_cut_count=len(actual_cuts),
        clip_scene_cut_frame_indices=actual_cuts,
        object_visibility_scene_cut_markers=actual_markers,
        track_initialization_markers=initialization,
        state_machine_boundary_markers=legacy,
        legacy_scene_cut_state_count=legacy,
    )
