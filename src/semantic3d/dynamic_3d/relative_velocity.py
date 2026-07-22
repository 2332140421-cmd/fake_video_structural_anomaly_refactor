"""Scale-normalized relative speed diagnostics and consistency residuals."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from ..sequence_geometry import SequenceScaleStatus
from ..validity import ResidualEvidence
from .motion_model import trajectory_coordinate
from .readiness import DynamicGeometryMode
from .track_observation import PointTrack3DObservation


@dataclass(frozen=True)
class RelativeVelocityResidual:
    """Non-metric speed plus history/object consistency evidence."""

    point_id: str
    object_track_id: str
    previous_frame_index: int
    current_frame_index: int
    raw_displacement: float
    normalized_relative_speed: float
    speed_unit: str
    speed_diagnostic: ResidualEvidence
    speed_change_residual: ResidualEvidence
    object_median_speed_residual: ResidualEvidence
    valid: bool
    missing_reason: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.valid and (not self.speed_diagnostic.valid or self.missing_reason):
            raise ValueError("Valid speed result requires a diagnostic value.")
        if not self.valid:
            if not self.missing_reason or math.isfinite(float(self.raw_displacement)) or math.isfinite(float(self.normalized_relative_speed)):
                raise ValueError("Invalid speed result requires NaN values and a reason.")
        object.__setattr__(self, "metadata", dict(self.metadata))


def _missing(point: PointTrack3DObservation, previous_frame: int, reason: str) -> RelativeVelocityResidual:
    ids = (point.object_track_id, point.point_id, str(point.frame_index))
    return RelativeVelocityResidual(
        point.point_id, point.object_track_id, previous_frame, point.frame_index,
        float("nan"), float("nan"), "unavailable",
        ResidualEvidence.missing("relative_speed_diagnostic", reason, source_ids=ids),
        ResidualEvidence.missing("r_relative_speed_change", reason, source_ids=ids),
        ResidualEvidence.missing("r_point_vs_object_median_speed", reason, source_ids=ids),
        False, reason,
    )


def compute_relative_velocity_residuals(
    observations: Sequence[PointTrack3DObservation],
    object_scale_by_track_and_frame: Mapping[str, Mapping[int, Optional[float]]],
    *,
    seconds_per_frame: Optional[float] = None,
) -> tuple[RelativeVelocityResidual, ...]:
    """Compute scale-normalized speed; high speed alone is never anomaly evidence."""

    if seconds_per_frame is not None and seconds_per_frame <= 0.0:
        raise ValueError("seconds_per_frame must be positive.")
    by_key: dict[tuple[str, str], list[PointTrack3DObservation]] = {}
    for point in observations:
        by_key.setdefault((point.object_track_id, point.point_id), []).append(point)
    per_transition_speed: dict[tuple[str, str, int], float] = {}
    rows: list[RelativeVelocityResidual] = []
    pending: list[tuple[PointTrack3DObservation, PointTrack3DObservation, Optional[PointTrack3DObservation]]] = []
    for (_, _), samples in sorted(by_key.items()):
        samples.sort(key=lambda item: item.frame_index)
        for index, (previous, current) in enumerate(zip(samples, samples[1:])):
            earlier = samples[index - 1] if index > 0 else None
            pending.append((previous, current, earlier))
            if current.geometry_mode in {DynamicGeometryMode.ROTATION_COMPENSATED, DynamicGeometryMode.UNAVAILABLE}:
                continue
            first, second = trajectory_coordinate(previous), trajectory_coordinate(current)
            scale = object_scale_by_track_and_frame.get(current.object_track_id, {}).get(current.frame_index)
            frame_delta = current.frame_index - previous.frame_index
            if first is None or second is None or frame_delta <= 0 or scale is None or not math.isfinite(float(scale)) or float(scale) <= 0.0:
                continue
            time_delta = frame_delta * (seconds_per_frame or 1.0)
            speed = float(np.linalg.norm(second - first) / (time_delta * float(scale)))
            per_transition_speed[(current.object_track_id, current.point_id, current.frame_index)] = speed
    for previous, current, earlier in pending:
        reason = ""
        if current.geometry_mode == DynamicGeometryMode.ROTATION_COMPENSATED:
            reason = "rotation_only_no_complete_3d_speed"
        elif current.geometry_mode == DynamicGeometryMode.UNAVAILABLE:
            reason = "dynamic_geometry_unavailable"
        speed = per_transition_speed.get((current.object_track_id, current.point_id, current.frame_index))
        if speed is None and not reason:
            reason = "invalid_or_missing_object_scale"
        if reason:
            rows.append(_missing(current, previous.frame_index, reason))
            continue
        assert speed is not None
        first, second = trajectory_coordinate(previous), trajectory_coordinate(current)
        assert first is not None and second is not None
        raw = float(np.linalg.norm(second - first))
        unit = "object_scale_per_second" if seconds_per_frame is not None else "object_scale_per_frame"
        source_ids = (current.object_track_id, current.point_id, str(current.frame_index))
        diagnostic = ResidualEvidence.observed("relative_speed_diagnostic", speed, quality=min(previous.reconstruction_quality, current.reconstruction_quality), source_ids=source_ids, metadata={"unit": unit, "not_anomaly_by_magnitude_alone": True, "raw_displacement_relative_units": raw})
        previous_speed = None if earlier is None else per_transition_speed.get((current.object_track_id, current.point_id, previous.frame_index))
        speed_change = (
            ResidualEvidence.observed("r_relative_speed_change", abs(speed - previous_speed), quality=diagnostic.quality, source_ids=source_ids)
            if previous_speed is not None
            else ResidualEvidence.missing("r_relative_speed_change", "insufficient_speed_history", source_ids=source_ids)
        )
        peer_speeds = [
            value for (object_id, point_id, frame), value in per_transition_speed.items()
            if object_id == current.object_track_id and frame == current.frame_index and point_id != current.point_id
        ]
        object_residual = (
            ResidualEvidence.observed("r_point_vs_object_median_speed", abs(speed - float(np.median(peer_speeds))), quality=diagnostic.quality, source_ids=source_ids)
            if peer_speeds
            else ResidualEvidence.missing("r_point_vs_object_median_speed", "insufficient_object_speed_support", source_ids=source_ids)
        )
        rows.append(RelativeVelocityResidual(current.point_id, current.object_track_id, previous.frame_index, current.frame_index, raw, speed, unit, diagnostic, speed_change, object_residual, True, metadata={"metric_speed_claimed": False, "sequence_scale_status": current.scale_status.value if isinstance(current.scale_status, SequenceScaleStatus) else str(current.scale_status)}))
    return tuple(rows)
