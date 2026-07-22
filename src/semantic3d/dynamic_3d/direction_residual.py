"""Direction consistency evidence for object-bound dynamic point tracks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from ..validity import ResidualEvidence
from .motion_model import trajectory_coordinate
from .readiness import DynamicGeometryMode
from .track_observation import PointTrack3DObservation


@dataclass(frozen=True)
class DirectionConsistencyResidual:
    """Own-history, object-median, and local-neighbour direction evidence."""

    point_id: str
    object_track_id: str
    current_frame_index: int
    geometry_mode: DynamicGeometryMode | str
    direction_domain: str
    own_history: ResidualEvidence
    object_median: ResidualEvidence
    local_neighbour: ResidualEvidence
    valid: bool
    missing_reason: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        mode = DynamicGeometryMode(self.geometry_mode)
        if self.valid and (not self.own_history.valid or self.missing_reason):
            raise ValueError("Valid direction result requires own-history evidence.")
        if not self.valid and not self.missing_reason:
            raise ValueError("Invalid direction result requires missing_reason.")
        object.__setattr__(self, "geometry_mode", mode)
        object.__setattr__(self, "metadata", dict(self.metadata))


def _missing_evidence(name: str, reason: str, source_ids: Sequence[str]) -> ResidualEvidence:
    return ResidualEvidence.missing(name, reason, source_ids=source_ids)


def _direction_distance(first: np.ndarray, second: np.ndarray, minimum_displacement: float) -> Optional[float]:
    norm_a, norm_b = float(np.linalg.norm(first)), float(np.linalg.norm(second))
    if norm_a <= minimum_displacement or norm_b <= minimum_displacement:
        return None
    cosine = float(np.clip(np.dot(first, second) / (norm_a * norm_b), -1.0, 1.0))
    return 1.0 - cosine


def compute_direction_consistency_residuals(
    observations: Sequence[PointTrack3DObservation],
    *,
    neighbour_ids: Optional[Mapping[str, Sequence[str]]] = None,
    minimum_displacement: float = 1e-6,
) -> tuple[DirectionConsistencyResidual, ...]:
    """Compute direction disagreement while treating near-static motion as missing."""

    if minimum_displacement <= 0.0:
        raise ValueError("minimum_displacement must be positive.")
    by_key: dict[tuple[str, str], list[PointTrack3DObservation]] = {}
    for point in observations:
        by_key.setdefault((point.object_track_id, point.point_id), []).append(point)
    lookup = {
        (point.object_track_id, point.point_id, point.frame_index): point
        for point in observations
    }
    neighbours = {str(key): tuple(value) for key, value in (neighbour_ids or {}).items()}
    output = []
    for (object_id, point_id), samples in sorted(by_key.items()):
        samples.sort(key=lambda item: item.frame_index)
        if len(samples) < 3:
            continue
        for first, previous, current in zip(samples, samples[1:], samples[2:]):
            source_ids = (object_id, point_id, str(current.frame_index))
            mode = current.geometry_mode
            domain = "rotation_compensated_bearing" if mode == DynamicGeometryMode.ROTATION_COMPENSATED else "shared_scale_3d"
            reason = ""
            if mode == DynamicGeometryMode.UNAVAILABLE:
                reason = "dynamic_geometry_unavailable"
            elif len({first.geometry_mode, previous.geometry_mode, current.geometry_mode}) != 1:
                reason = "inconsistent_geometry_mode"
            coordinates = [trajectory_coordinate(item) for item in (first, previous, current)]
            if not reason and any(value is None for value in coordinates):
                reason = "missing_trajectory_coordinate"
            if reason:
                missing = _missing_evidence("r_direction_own_history", reason, source_ids)
                output.append(DirectionConsistencyResidual(point_id, object_id, current.frame_index, mode, domain, missing, _missing_evidence("r_direction_object_median", reason, source_ids), _missing_evidence("r_direction_local_neighbour", reason, source_ids), False, reason))
                continue
            xyz0, xyz1, xyz2 = coordinates
            assert xyz0 is not None and xyz1 is not None and xyz2 is not None
            previous_delta, current_delta = xyz1 - xyz0, xyz2 - xyz1
            own_value = _direction_distance(current_delta, previous_delta, minimum_displacement)
            if own_value is None:
                reason = "direction_undefined_for_near_static_displacement"
                own = _missing_evidence("r_direction_own_history", reason, source_ids)
            else:
                own = ResidualEvidence.observed("r_direction_own_history", own_value, quality=min(item.reconstruction_quality for item in (first, previous, current)), source_ids=source_ids, metadata={"formula": "1-cos(delta_t,delta_t_minus_1)", "direction_domain": domain})
            object_displacements = []
            for (candidate_object, candidate_id), candidate_samples in by_key.items():
                if candidate_object != object_id or candidate_id == point_id:
                    continue
                before = lookup.get((object_id, candidate_id, previous.frame_index))
                after = lookup.get((object_id, candidate_id, current.frame_index))
                if before is None or after is None:
                    continue
                before_xyz, after_xyz = trajectory_coordinate(before), trajectory_coordinate(after)
                if before_xyz is not None and after_xyz is not None:
                    object_displacements.append(after_xyz - before_xyz)
            if object_displacements:
                median_delta = np.median(np.asarray(object_displacements), axis=0)
                value = _direction_distance(current_delta, median_delta, minimum_displacement)
            else:
                value = None
            object_evidence = (
                ResidualEvidence.observed("r_direction_object_median", value, quality=current.reconstruction_quality, source_ids=source_ids)
                if value is not None
                else _missing_evidence("r_direction_object_median", "insufficient_object_direction_support", source_ids)
            )
            local_displacements = []
            for neighbour_id in neighbours.get(point_id, ()):
                before = lookup.get((object_id, neighbour_id, previous.frame_index))
                after = lookup.get((object_id, neighbour_id, current.frame_index))
                if before is None or after is None:
                    continue
                before_xyz, after_xyz = trajectory_coordinate(before), trajectory_coordinate(after)
                if before_xyz is not None and after_xyz is not None:
                    local_displacements.append(after_xyz - before_xyz)
            local_value = None
            if local_displacements:
                local_value = _direction_distance(current_delta, np.median(np.asarray(local_displacements), axis=0), minimum_displacement)
            local_evidence = (
                ResidualEvidence.observed("r_direction_local_neighbour", local_value, quality=current.reconstruction_quality, source_ids=source_ids)
                if local_value is not None
                else _missing_evidence("r_direction_local_neighbour", "insufficient_neighbour_direction_support", source_ids)
            )
            output.append(DirectionConsistencyResidual(
                point_id=point_id, object_track_id=object_id,
                current_frame_index=current.frame_index, geometry_mode=mode,
                direction_domain=domain, own_history=own,
                object_median=object_evidence, local_neighbour=local_evidence,
                valid=own.valid, missing_reason="" if own.valid else own.missing_reason,
                metadata={"near_static_is_not_anomaly": True},
            ))
    return tuple(output)
