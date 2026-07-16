"""Independent cross-frame reprojection QA and dynamic residual evidence."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from ..geometry.camera import validate_intrinsics, validate_rigid_transform
from ..validity import ResidualEvidence
from .readiness import DynamicGeometryMode
from .track_observation import PointTrack2DObservation, PointTrack3DObservation


class ReprojectionEvidenceType(str, Enum):
    """Interpretation of a measured reprojection error."""

    BACKGROUND_QA = "background_reprojection_quality"
    CAMERA_COMPENSATED_MOTION = "camera_compensated_motion_diagnostic"
    DYNAMIC_RESIDUAL = "dynamic_reprojection_residual"
    MISSING = "missing"


@dataclass(frozen=True)
class DynamicReprojectionResidual:
    """Prediction against an independently tracked current-frame pixel."""

    point_id: str
    object_track_id: str
    source_frame_index: int
    target_frame_index: int
    geometry_mode: DynamicGeometryMode | str
    evidence_type: ReprojectionEvidenceType | str
    predicted_uv: Optional[tuple[float, float]]
    observed_uv: Optional[tuple[float, float]]
    pixel_error: float
    normalized_pixel_error: float
    quality: float
    diagnostic_evidence: ResidualEvidence
    residual_evidence: ResidualEvidence
    valid: bool
    missing_reason: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        mode = DynamicGeometryMode(self.geometry_mode)
        evidence_type = ReprojectionEvidenceType(self.evidence_type)
        predicted = _uv_or_none(self.predicted_uv)
        observed = _uv_or_none(self.observed_uv)
        error = float(self.pixel_error)
        normalized = float(self.normalized_pixel_error)
        quality = float(self.quality)
        if not math.isfinite(quality) or not 0.0 <= quality <= 1.0:
            raise ValueError("Reprojection quality must be in [0, 1].")
        if self.valid:
            if predicted is None or observed is None:
                raise ValueError("Valid reprojection requires predicted and observed pixels.")
            if not all(math.isfinite(value) and value >= 0.0 for value in (error, normalized)):
                raise ValueError("Valid reprojection errors must be non-negative.")
            if self.missing_reason or not self.diagnostic_evidence.valid:
                raise ValueError("Valid reprojection requires diagnostic evidence.")
        else:
            if any(math.isfinite(value) for value in (error, normalized)):
                raise ValueError("Invalid reprojection errors must be NaN.")
            if predicted is not None or observed is not None or not self.missing_reason:
                raise ValueError("Invalid reprojection requires no pixels and a reason.")
            if self.diagnostic_evidence.valid or self.residual_evidence.valid:
                raise ValueError("Invalid reprojection cannot contain valid evidence.")
        if evidence_type != ReprojectionEvidenceType.DYNAMIC_RESIDUAL and self.residual_evidence.valid:
            raise ValueError("Only a motion-predicted foreground comparison is anomaly evidence.")
        object.__setattr__(self, "geometry_mode", mode)
        object.__setattr__(self, "evidence_type", evidence_type)
        object.__setattr__(self, "predicted_uv", predicted)
        object.__setattr__(self, "observed_uv", observed)
        object.__setattr__(self, "pixel_error", error)
        object.__setattr__(self, "normalized_pixel_error", normalized)
        object.__setattr__(self, "quality", quality)
        object.__setattr__(self, "metadata", dict(self.metadata))


def _uv_or_none(value: Optional[Sequence[float]]) -> Optional[tuple[float, float]]:
    if value is None:
        return None
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.shape != (2,) or not np.isfinite(array).all():
        raise ValueError("Pixel coordinate must contain two finite values.")
    return float(array[0]), float(array[1])


def _missing(
    previous: PointTrack3DObservation,
    current: PointTrack2DObservation,
    mode: DynamicGeometryMode,
    reason: str,
) -> DynamicReprojectionResidual:
    source_ids = (previous.point_id, str(previous.frame_index), str(current.frame_index))
    return DynamicReprojectionResidual(
        point_id=previous.point_id,
        object_track_id=previous.object_track_id,
        source_frame_index=previous.frame_index,
        target_frame_index=current.frame_index,
        geometry_mode=mode,
        evidence_type=ReprojectionEvidenceType.MISSING,
        predicted_uv=None,
        observed_uv=None,
        pixel_error=float("nan"),
        normalized_pixel_error=float("nan"),
        quality=0.0,
        diagnostic_evidence=ResidualEvidence.missing(
            "camera_compensated_reprojection_diagnostic", reason, source_ids=source_ids
        ),
        residual_evidence=ResidualEvidence.missing(
            "r_dynamic_reprojection", reason, source_ids=source_ids
        ),
        valid=False,
        missing_reason=reason,
    )


def _project_array(point_camera: np.ndarray, K: np.ndarray) -> Optional[np.ndarray]:
    if not np.isfinite(point_camera).all() or point_camera[2] <= 1e-12:
        return None
    homogeneous = K @ point_camera
    if not np.isfinite(homogeneous).all() or abs(homogeneous[2]) <= 1e-12:
        return None
    return homogeneous[:2] / homogeneous[2]


def compute_dynamic_reprojection_residual(
    previous_point: PointTrack3DObservation,
    current_observation: PointTrack2DObservation,
    *,
    K_current: np.ndarray,
    image_width: int,
    image_height: int,
    relative_pose_current_from_previous: np.ndarray,
    geometry_mode: DynamicGeometryMode | str,
    is_background: bool,
    predicted_foreground_point_current_camera: Optional[Sequence[float]] = None,
    has_history_motion_model: bool = False,
) -> DynamicReprojectionResidual:
    """Compare a geometric prediction with an independent current observation.

    Background output is camera-geometry QA. Foreground camera-only output is a
    motion diagnostic. It becomes a dynamic residual only when an independent
    history motion prediction is supplied.
    """

    mode = DynamicGeometryMode(geometry_mode)
    if mode == DynamicGeometryMode.UNAVAILABLE:
        return _missing(previous_point, current_observation, mode, "dynamic_geometry_unavailable")
    if previous_point.point_id != current_observation.point_id:
        return _missing(previous_point, current_observation, mode, "point_id_mismatch")
    if previous_point.object_track_id != current_observation.object_track_id:
        return _missing(previous_point, current_observation, mode, "object_track_id_mismatch")
    if not previous_point.valid or previous_point.point_3d_camera is None:
        return _missing(previous_point, current_observation, mode, "invalid_previous_3d_point")
    if not current_observation.valid or current_observation.pixel_uv is None:
        return _missing(previous_point, current_observation, mode, "missing_current_2d_observation")
    if not bool(current_observation.metadata.get("independent_observation", False)):
        return _missing(previous_point, current_observation, mode, "current_observation_not_independent")
    if bool(current_observation.metadata.get("generated_from_projection", False)):
        return _missing(previous_point, current_observation, mode, "same_source_projection_loop")
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive.")
    matrix = validate_intrinsics(K_current)
    transform = validate_rigid_transform(
        relative_pose_current_from_previous, "relative_pose_current_from_previous"
    )
    previous_xyz = np.asarray(previous_point.point_3d_camera, dtype=float)
    if predicted_foreground_point_current_camera is not None:
        predicted_xyz = np.asarray(
            predicted_foreground_point_current_camera, dtype=float
        ).reshape(-1)
        if predicted_xyz.shape != (3,) or not np.isfinite(predicted_xyz).all():
            return _missing(previous_point, current_observation, mode, "invalid_motion_prediction")
        if not has_history_motion_model:
            return _missing(previous_point, current_observation, mode, "motion_prediction_without_history")
        evidence_type = ReprojectionEvidenceType.DYNAMIC_RESIDUAL
    else:
        if mode == DynamicGeometryMode.STATIC_CAMERA_3D:
            predicted_xyz = previous_xyz.copy()
        elif mode == DynamicGeometryMode.ROTATION_COMPENSATED:
            predicted_xyz = transform[:3, :3] @ previous_xyz
        elif mode == DynamicGeometryMode.FULL_SE3_3D:
            predicted_xyz = (transform @ np.concatenate([previous_xyz, [1.0]]))[:3]
        else:
            return _missing(previous_point, current_observation, mode, "unsupported_geometry_mode")
        evidence_type = (
            ReprojectionEvidenceType.BACKGROUND_QA
            if is_background
            else ReprojectionEvidenceType.CAMERA_COMPENSATED_MOTION
        )
    predicted_uv = _project_array(predicted_xyz, matrix)
    if predicted_uv is None:
        return _missing(previous_point, current_observation, mode, "predicted_point_not_projectable")
    observed_uv = np.asarray(current_observation.pixel_uv, dtype=float)
    pixel_error = float(np.linalg.norm(observed_uv - predicted_uv))
    diagonal = math.hypot(image_width, image_height)
    normalized = pixel_error / diagonal
    quality = float(
        min(
            previous_point.reconstruction_quality,
            current_observation.tracking_confidence,
        )
    )
    source_ids = (
        previous_point.point_id,
        str(previous_point.frame_index),
        str(current_observation.frame_index),
    )
    diagnostic = ResidualEvidence.observed(
        evidence_type.value,
        normalized,
        quality=quality,
        source_ids=source_ids,
        metadata={
            "pixel_error": pixel_error,
            "background_qa": is_background,
            "not_forgery_by_itself": evidence_type != ReprojectionEvidenceType.DYNAMIC_RESIDUAL,
            "observed_uv_source": current_observation.source_tracker,
            "independent_observation": True,
        },
    )
    residual = (
        ResidualEvidence.observed(
            "r_dynamic_reprojection",
            normalized,
            quality=quality,
            source_ids=source_ids,
            metadata={
                "pixel_error": pixel_error,
                "history_motion_model": True,
                "observed_uv_source": current_observation.source_tracker,
                "anomaly_threshold_applied": False,
            },
        )
        if evidence_type == ReprojectionEvidenceType.DYNAMIC_RESIDUAL
        else ResidualEvidence.missing(
            "r_dynamic_reprojection",
            (
                "background_reprojection_is_geometry_qa"
                if is_background
                else "foreground_motion_model_unavailable"
            ),
            source_ids=source_ids,
            metadata={"diagnostic_value": normalized},
        )
    )
    return DynamicReprojectionResidual(
        point_id=previous_point.point_id,
        object_track_id=previous_point.object_track_id,
        source_frame_index=previous_point.frame_index,
        target_frame_index=current_observation.frame_index,
        geometry_mode=mode,
        evidence_type=evidence_type,
        predicted_uv=tuple(predicted_uv),
        observed_uv=tuple(observed_uv),
        pixel_error=pixel_error,
        normalized_pixel_error=normalized,
        quality=quality,
        diagnostic_evidence=diagnostic,
        residual_evidence=residual,
        valid=True,
        metadata={
            "translation_used": mode == DynamicGeometryMode.FULL_SE3_3D,
            "rotation_only_translation_assumed_zero": False,
            "current_observation_independent": True,
        },
    )
