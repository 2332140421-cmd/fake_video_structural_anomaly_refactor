"""History-only object and point motion prediction models."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from ..geometry.camera import validate_intrinsics
from .readiness import DynamicGeometryMode
from .track_observation import PointTrack3DObservation


@dataclass(frozen=True)
class ObjectMotionPrediction:
    """Prediction formed exclusively from frames preceding the target frame."""

    point_id: str
    object_track_id: str
    target_frame_index: int
    predicted_point_3d: Optional[tuple[float, float, float]]
    predicted_uv: Optional[tuple[float, float]]
    history_frames: tuple[int, ...]
    support_point_ids: tuple[str, ...]
    model_type: str
    prediction_quality: float
    valid: bool
    missing_reason: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        quality = float(self.prediction_quality)
        if not math.isfinite(quality) or not 0.0 <= quality <= 1.0:
            raise ValueError("prediction_quality must be in [0, 1].")
        if any(frame >= self.target_frame_index for frame in self.history_frames):
            raise ValueError("Motion prediction cannot read the target or future frame.")
        if self.valid:
            if self.predicted_point_3d is None or len(self.history_frames) < 2 or self.missing_reason:
                raise ValueError("Valid prediction requires two history frames and 3D state.")
            if not np.isfinite(np.asarray(self.predicted_point_3d, dtype=float)).all():
                raise ValueError("Predicted point must be finite.")
        elif self.predicted_point_3d is not None or self.predicted_uv is not None or not self.missing_reason:
            raise ValueError("Invalid prediction requires no fabricated coordinates and a reason.")
        object.__setattr__(self, "history_frames", tuple(self.history_frames))
        object.__setattr__(self, "support_point_ids", tuple(self.support_point_ids))
        object.__setattr__(self, "prediction_quality", quality)
        object.__setattr__(self, "metadata", dict(self.metadata))


def _missing(point_id: str, object_track_id: str, target: int, model: str, reason: str) -> ObjectMotionPrediction:
    return ObjectMotionPrediction(point_id, object_track_id, target, None, None, (), (), model, 0.0, False, reason)


def trajectory_coordinate(point: PointTrack3DObservation) -> Optional[np.ndarray]:
    """Return an allowed trajectory coordinate or a unit bearing in rotation mode."""

    if not point.valid:
        return None
    if point.geometry_mode == DynamicGeometryMode.FULL_SE3_3D:
        value = point.point_3d_world
    elif point.geometry_mode == DynamicGeometryMode.ROTATION_COMPENSATED and point.metadata.get("rotation_compensated_bearing") is not None:
        value = point.metadata["rotation_compensated_bearing"]
    else:
        value = point.point_3d_camera
    if value is None:
        return None
    array = np.asarray(value, dtype=float)
    if point.geometry_mode == DynamicGeometryMode.ROTATION_COMPENSATED:
        norm = float(np.linalg.norm(array))
        return None if norm <= 1e-12 else array / norm
    if point.geometry_mode == DynamicGeometryMode.UNAVAILABLE:
        return None
    return array


def _project(point: np.ndarray, K: Optional[np.ndarray]) -> Optional[tuple[float, float]]:
    if K is None or point[2] <= 1e-12:
        return None
    matrix = validate_intrinsics(K)
    homogeneous = matrix @ point
    if abs(float(homogeneous[2])) <= 1e-12:
        return None
    uv = homogeneous[:2] / homogeneous[2]
    return float(uv[0]), float(uv[1])


class BaseObjectMotionModel(ABC):
    """Canonical history-only object motion model interface."""

    @abstractmethod
    def predict(
        self,
        history: Sequence[PointTrack3DObservation],
        *,
        target_frame_index: int,
        K_current: Optional[np.ndarray] = None,
    ) -> ObjectMotionPrediction:
        """Predict one target point without consuming its target observation."""


class PointConstantVelocityModel(BaseObjectMotionModel):
    """Predict a point with its two latest valid history states."""

    model_type = "point_constant_velocity_history_only"

    def predict(self, history: Sequence[PointTrack3DObservation], *, target_frame_index: int, K_current: Optional[np.ndarray] = None) -> ObjectMotionPrediction:
        past = sorted((item for item in history if item.frame_index < target_frame_index and item.valid), key=lambda item: item.frame_index)
        if not past:
            return _missing("unknown", "unknown", target_frame_index, self.model_type, "insufficient_history")
        point_id, object_id = past[-1].point_id, past[-1].object_track_id
        if len(past) < 2:
            return _missing(point_id, object_id, target_frame_index, self.model_type, "insufficient_history")
        previous_previous, previous = past[-2:]
        if previous_previous.point_id != previous.point_id or previous_previous.object_track_id != previous.object_track_id:
            return _missing(point_id, object_id, target_frame_index, self.model_type, "history_identity_mismatch")
        if previous_previous.geometry_mode != previous.geometry_mode:
            return _missing(point_id, object_id, target_frame_index, self.model_type, "history_geometry_mode_mismatch")
        first, second = trajectory_coordinate(previous_previous), trajectory_coordinate(previous)
        if first is None or second is None:
            return _missing(point_id, object_id, target_frame_index, self.model_type, "history_geometry_unavailable")
        history_delta = previous.frame_index - previous_previous.frame_index
        target_delta = target_frame_index - previous.frame_index
        if history_delta <= 0 or target_delta <= 0:
            return _missing(point_id, object_id, target_frame_index, self.model_type, "invalid_history_frame_order")
        predicted = second + (second - first) * (target_delta / history_delta)
        if previous.geometry_mode == DynamicGeometryMode.ROTATION_COMPENSATED:
            norm = float(np.linalg.norm(predicted))
            if norm <= 1e-12:
                return _missing(point_id, object_id, target_frame_index, self.model_type, "invalid_predicted_bearing")
            predicted = predicted / norm
            model_type = "point_constant_angular_velocity_history_only"
        else:
            model_type = self.model_type
        quality = min(previous_previous.reconstruction_quality, previous.reconstruction_quality)
        return ObjectMotionPrediction(
            point_id=point_id,
            object_track_id=object_id,
            target_frame_index=target_frame_index,
            predicted_point_3d=tuple(float(value) for value in predicted),
            predicted_uv=_project(predicted, K_current),
            history_frames=(previous_previous.frame_index, previous.frame_index),
            support_point_ids=(point_id,),
            model_type=model_type,
            prediction_quality=quality,
            valid=True,
            metadata={"target_observation_used": False, "coordinate_kind": "bearing" if previous.geometry_mode == DynamicGeometryMode.ROTATION_COMPENSATED else "3d"},
        )


class ObjectMedianTranslationModel(BaseObjectMotionModel):
    """Predict one point from the robust median displacement of other points."""

    model_type = "object_median_translation_leave_one_point_out"

    def __init__(self, histories_by_point: Mapping[str, Sequence[PointTrack3DObservation]]) -> None:
        self.histories_by_point = {str(key): tuple(value) for key, value in histories_by_point.items()}

    def predict(self, history: Sequence[PointTrack3DObservation], *, target_frame_index: int, K_current: Optional[np.ndarray] = None) -> ObjectMotionPrediction:
        target_history = sorted((item for item in history if item.frame_index < target_frame_index and item.valid), key=lambda item: item.frame_index)
        if not target_history:
            return _missing("unknown", "unknown", target_frame_index, self.model_type, "insufficient_history")
        target = target_history[-1]
        if target.geometry_mode == DynamicGeometryMode.ROTATION_COMPENSATED:
            return _missing(target.point_id, target.object_track_id, target_frame_index, self.model_type, "rotation_only_no_translation_model")
        base = trajectory_coordinate(target)
        if base is None:
            return _missing(target.point_id, target.object_track_id, target_frame_index, self.model_type, "history_geometry_unavailable")
        displacements, support_ids, qualities = [], [], []
        model_history_frames = {target.frame_index}
        for point_id, samples in self.histories_by_point.items():
            if point_id == target.point_id:
                continue
            past = sorted((item for item in samples if item.frame_index < target_frame_index and item.valid), key=lambda item: item.frame_index)
            if len(past) < 2 or past[-1].object_track_id != target.object_track_id:
                continue
            first, second = trajectory_coordinate(past[-2]), trajectory_coordinate(past[-1])
            if first is None or second is None:
                continue
            gap = past[-1].frame_index - past[-2].frame_index
            if gap <= 0:
                continue
            displacements.append((second - first) / gap)
            support_ids.append(point_id)
            qualities.append(min(past[-2].reconstruction_quality, past[-1].reconstruction_quality))
            model_history_frames.update((past[-2].frame_index, past[-1].frame_index))
        if not displacements:
            return _missing(target.point_id, target.object_track_id, target_frame_index, self.model_type, "no_leave_one_out_support_points")
        velocity = np.median(np.asarray(displacements), axis=0)
        predicted = base + velocity * (target_frame_index - target.frame_index)
        return ObjectMotionPrediction(
            point_id=target.point_id,
            object_track_id=target.object_track_id,
            target_frame_index=target_frame_index,
            predicted_point_3d=tuple(float(value) for value in predicted),
            predicted_uv=_project(predicted, K_current),
            history_frames=tuple(sorted(model_history_frames)),
            support_point_ids=tuple(support_ids),
            model_type=self.model_type,
            prediction_quality=float(min(target.reconstruction_quality, np.median(qualities))),
            valid=True,
            metadata={"target_observation_used": False, "leave_one_point_out": True},
        )
