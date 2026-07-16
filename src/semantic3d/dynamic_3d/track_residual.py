"""Three-frame 3D point-track continuity evidence."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from ..sequence_geometry import SequenceScaleStatus
from ..validity import ResidualEvidence
from .readiness import DynamicGeometryMode
from .track_observation import PointTrack3DObservation


@dataclass(frozen=True)
class Track3DContinuityResidual:
    """Constant-velocity and second-difference evidence for one tracked point."""

    point_id: str
    object_track_id: str
    previous_previous_frame_index: Optional[int]
    previous_frame_index: Optional[int]
    current_frame_index: int
    coordinate_frame: str
    first_order_displacement: float
    predicted_point_3d: Optional[tuple[float, float, float]]
    observed_point_3d: Optional[tuple[float, float, float]]
    raw_residual: float
    normalized_residual: float
    raw_evidence: ResidualEvidence
    normalized_evidence: ResidualEvidence
    valid: bool
    missing_reason: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        predicted = _point_or_none(self.predicted_point_3d)
        observed = _point_or_none(self.observed_point_3d)
        first_order = float(self.first_order_displacement)
        raw = float(self.raw_residual)
        normalized = float(self.normalized_residual)
        if self.valid:
            if predicted is None or observed is None:
                raise ValueError("Valid track residual requires predicted and observed points.")
            if not all(math.isfinite(value) and value >= 0.0 for value in (first_order, raw)):
                raise ValueError("Valid displacement and raw residual must be non-negative.")
            if self.missing_reason or not self.raw_evidence.valid:
                raise ValueError("Valid track residual requires valid raw evidence.")
            if self.normalized_evidence.valid != math.isfinite(normalized):
                raise ValueError("Normalized evidence validity must match normalized_residual.")
        else:
            if any(math.isfinite(value) for value in (first_order, raw, normalized)):
                raise ValueError("Invalid track residual values must be NaN.")
            if predicted is not None or observed is not None or not self.missing_reason:
                raise ValueError("Invalid track residual requires no 3D points and a reason.")
            if self.raw_evidence.valid or self.normalized_evidence.valid:
                raise ValueError("Invalid track residual cannot contain valid evidence.")
        object.__setattr__(self, "predicted_point_3d", predicted)
        object.__setattr__(self, "observed_point_3d", observed)
        object.__setattr__(self, "first_order_displacement", first_order)
        object.__setattr__(self, "raw_residual", raw)
        object.__setattr__(self, "normalized_residual", normalized)
        object.__setattr__(self, "metadata", dict(self.metadata))


def _point_or_none(value: Optional[Sequence[float]]) -> Optional[tuple[float, float, float]]:
    if value is None:
        return None
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError("A 3D residual point must contain three finite values.")
    return tuple(float(item) for item in array)


def _missing_result(
    point_id: str,
    object_track_id: str,
    current_frame_index: int,
    reason: str,
    *,
    previous_previous_frame_index: Optional[int] = None,
    previous_frame_index: Optional[int] = None,
) -> Track3DContinuityResidual:
    source_ids = (point_id, object_track_id, str(current_frame_index))
    return Track3DContinuityResidual(
        point_id=point_id,
        object_track_id=object_track_id,
        previous_previous_frame_index=previous_previous_frame_index,
        previous_frame_index=previous_frame_index,
        current_frame_index=current_frame_index,
        coordinate_frame="unknown",
        first_order_displacement=float("nan"),
        predicted_point_3d=None,
        observed_point_3d=None,
        raw_residual=float("nan"),
        normalized_residual=float("nan"),
        raw_evidence=ResidualEvidence.missing(
            "r_track_3d_continuity_raw", reason, source_ids=source_ids
        ),
        normalized_evidence=ResidualEvidence.missing(
            "r_track_3d_continuity_normalized", reason, source_ids=source_ids
        ),
        valid=False,
        missing_reason=reason,
    )


def _trajectory_point(
    point: PointTrack3DObservation,
) -> tuple[Optional[np.ndarray], str]:
    if point.geometry_mode == DynamicGeometryMode.FULL_SE3_3D:
        return (
            None if point.point_3d_world is None else np.asarray(point.point_3d_world),
            "world",
        )
    if point.geometry_mode == DynamicGeometryMode.STATIC_CAMERA_3D:
        return (
            None if point.point_3d_camera is None else np.asarray(point.point_3d_camera),
            "camera_static_gauge",
        )
    return None, "unknown"


def compute_track_3d_continuity_residuals(
    observations: Sequence[PointTrack3DObservation],
    *,
    object_observed_scale_3d_by_frame: Optional[Mapping[int, float]] = None,
    scene_cut_flags: Optional[Mapping[int, bool]] = None,
    epsilon: float = 1e-8,
) -> tuple[Track3DContinuityResidual, ...]:
    """Compute constant-velocity residuals without crossing invalid geometry."""

    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive.")
    by_point: dict[tuple[str, str], list[PointTrack3DObservation]] = {}
    for point in observations:
        by_point.setdefault((point.object_track_id, point.point_id), []).append(point)
    results: list[Track3DContinuityResidual] = []
    cuts = dict(scene_cut_flags or {})
    scales = dict(object_observed_scale_3d_by_frame or {})
    for (object_track_id, point_id), samples in sorted(by_point.items()):
        samples.sort(key=lambda item: item.frame_index)
        if len(samples) < 3:
            last = samples[-1]
            results.append(
                _missing_result(
                    point_id,
                    object_track_id,
                    last.frame_index,
                    "insufficient_same_point_history",
                    previous_frame_index=(samples[-2].frame_index if len(samples) > 1 else None),
                )
            )
            continue
        for first, second, current in zip(samples, samples[1:], samples[2:]):
            indices = (first.frame_index, second.frame_index, current.frame_index)
            reason = ""
            if len(set(indices)) != 3 or not (indices[0] < indices[1] < indices[2]):
                reason = "invalid_track_frame_order"
            elif any(cuts.get(index, False) for index in indices[1:]):
                reason = "scene_cut_breaks_track"
            elif any(not point.valid for point in (first, second, current)):
                reason = "invalid_3d_track_observation"
            elif any(
                point.scale_status == SequenceScaleStatus.RELATIVE_PER_FRAME
                for point in (first, second, current)
            ):
                reason = "relative_per_frame_not_comparable"
            elif len({point.scale_status for point in (first, second, current)}) != 1:
                reason = "inconsistent_sequence_scale_status"
            elif len({point.geometry_mode for point in (first, second, current)}) != 1:
                reason = "inconsistent_dynamic_geometry_mode"
            elif current.geometry_mode == DynamicGeometryMode.ROTATION_COMPENSATED:
                reason = "rotation_only_no_world_3d_continuity"
            if reason:
                results.append(
                    _missing_result(
                        point_id,
                        object_track_id,
                        current.frame_index,
                        reason,
                        previous_previous_frame_index=first.frame_index,
                        previous_frame_index=second.frame_index,
                    )
                )
                continue
            first_xyz, coordinate_frame = _trajectory_point(first)
            second_xyz, second_frame = _trajectory_point(second)
            current_xyz, current_frame = _trajectory_point(current)
            if (
                first_xyz is None
                or second_xyz is None
                or current_xyz is None
                or len({coordinate_frame, second_frame, current_frame}) != 1
            ):
                results.append(
                    _missing_result(
                        point_id,
                        object_track_id,
                        current.frame_index,
                        "missing_consistent_3d_coordinates",
                        previous_previous_frame_index=first.frame_index,
                        previous_frame_index=second.frame_index,
                    )
                )
                continue
            predicted = second_xyz + (second_xyz - first_xyz)
            first_order = float(np.linalg.norm(second_xyz - first_xyz))
            raw = float(np.linalg.norm(current_xyz - predicted))
            scale = float(scales.get(current.frame_index, math.nan))
            normalized = raw / (scale + epsilon) if math.isfinite(scale) and scale > 0 else math.nan
            source_ids = (point_id, object_track_id, str(current.frame_index))
            raw_evidence = ResidualEvidence.observed(
                "r_track_3d_continuity_raw",
                raw,
                quality=min(
                    first.reconstruction_quality,
                    second.reconstruction_quality,
                    current.reconstruction_quality,
                ),
                source_ids=source_ids,
                metadata={
                    "method": "constant_velocity_prediction_second_difference",
                    "coordinate_frame": coordinate_frame,
                    "anomaly_threshold_applied": False,
                },
            )
            normalized_evidence = (
                ResidualEvidence.observed(
                    "r_track_3d_continuity_normalized",
                    normalized,
                    quality=raw_evidence.quality,
                    source_ids=source_ids,
                    metadata={
                        "normalizer": "object_observed_scale_3d",
                        "scale": scale,
                        "anomaly_threshold_applied": False,
                    },
                )
                if math.isfinite(normalized)
                else ResidualEvidence.missing(
                    "r_track_3d_continuity_normalized",
                    "missing_object_observed_scale_3d",
                    source_ids=source_ids,
                    metadata={"raw_residual_preserved": raw},
                )
            )
            results.append(
                Track3DContinuityResidual(
                    point_id=point_id,
                    object_track_id=object_track_id,
                    previous_previous_frame_index=first.frame_index,
                    previous_frame_index=second.frame_index,
                    current_frame_index=current.frame_index,
                    coordinate_frame=coordinate_frame,
                    first_order_displacement=first_order,
                    predicted_point_3d=tuple(predicted),
                    observed_point_3d=tuple(current_xyz),
                    raw_residual=raw,
                    normalized_residual=normalized,
                    raw_evidence=raw_evidence,
                    normalized_evidence=normalized_evidence,
                    valid=True,
                    metadata={
                        "first_order_displacement_diagnostic_only": True,
                        "constant_velocity_prediction": True,
                        "second_difference_residual": True,
                    },
                )
            )
    return tuple(results)
