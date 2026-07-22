"""P4-B stage graph and deterministic invalidation rules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class StageDefinition:
    """One offline stage and its direct dependencies."""

    name: str
    dependencies: tuple[str, ...]


STAGES: tuple[StageDefinition, ...] = (
    StageDefinition("01_video_index", ()),
    StageDefinition("02_frame_decode", ("01_video_index",)),
    StageDefinition("03_object_detection", ("02_frame_decode",)),
    StageDefinition("04_instance_segmentation", ("03_object_detection",)),
    StageDefinition("05_keypoints", ("03_object_detection",)),
    StageDefinition("06_depth", ("02_frame_decode",)),
    StageDefinition("07_tracking", ("03_object_detection",)),
    StageDefinition("08_sequence_geometry", ("04_instance_segmentation", "06_depth", "07_tracking")),
    StageDefinition("09_shared_3d", ("05_keypoints", "08_sequence_geometry")),
    StageDefinition("10_static_evidence", ("09_shared_3d",)),
    StageDefinition("11_dynamic_evidence", ("09_shared_3d", "07_tracking")),
    StageDefinition("12_occlusion_evidence", ("04_instance_segmentation", "06_depth", "07_tracking")),
    StageDefinition("13_multilevel_aggregation", ("10_static_evidence", "11_dynamic_evidence", "12_occlusion_evidence")),
)

STAGE_BY_NAME = {stage.name: stage for stage in STAGES}


def descendants(stage_name: str) -> tuple[str, ...]:
    """Return transitive downstream stages in execution order."""

    if stage_name not in STAGE_BY_NAME:
        raise KeyError(f"Unknown stage: {stage_name}")
    affected = {stage_name}
    changed = True
    while changed:
        changed = False
        for stage in STAGES:
            if stage.name not in affected and any(dep in affected for dep in stage.dependencies):
                affected.add(stage.name)
                changed = True
    return tuple(stage.name for stage in STAGES if stage.name in affected)


def execution_plan(target_stage: str | None = None) -> tuple[StageDefinition, ...]:
    """Return all stages up to a requested target, or the complete graph."""

    if target_stage is None:
        return STAGES
    if target_stage not in STAGE_BY_NAME:
        raise KeyError(f"Unknown stage: {target_stage}")
    index = next(i for i, stage in enumerate(STAGES) if stage.name == target_stage)
    return STAGES[: index + 1]


def dependency_closure(stage_names: Iterable[str]) -> tuple[str, ...]:
    """Resolve direct selections to a dependency-complete ordered set."""

    selected = set(stage_names)
    changed = True
    while changed:
        changed = False
        for name in tuple(selected):
            for dependency in STAGE_BY_NAME[name].dependencies:
                if dependency not in selected:
                    selected.add(dependency)
                    changed = True
    return tuple(stage.name for stage in STAGES if stage.name in selected)
