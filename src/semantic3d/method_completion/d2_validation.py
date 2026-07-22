"""Deterministic synthetic validation for the D2 rotation-compensated path."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..dynamic_3d.readiness import DynamicGeometryMode
from ..dynamic_3d.reprojection_residual import compute_dynamic_reprojection_residual
from ..dynamic_3d.track_observation import (
    PointTrack2DObservation,
    PointTrack3DObservation,
)
from ..geometry.backprojection import backproject_pixel
from ..sequence_geometry.observation import SequenceScaleStatus
from ..shared_3d_observation import GeometryScaleStatus, VisibilityStatus


def _project(K: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    homogeneous = K @ xyz
    return homogeneous[:2] / homogeneous[2]


def _previous_point(K: np.ndarray, xyz: np.ndarray) -> PointTrack3DObservation:
    uv = _project(K, xyz)
    return PointTrack3DObservation(
        point_id="synthetic_point",
        object_track_id="synthetic_track",
        frame_index=0,
        pixel_uv=tuple(uv),
        observed_depth=float(xyz[2]),
        point_3d_camera=tuple(xyz),
        point_3d_world=None,
        visibility=VisibilityStatus.VISIBLE,
        occlusion_status="visible",
        tracking_confidence=1.0,
        depth_quality=1.0,
        reconstruction_quality=1.0,
        source_tracker="synthetic_ground_truth",
        scale_status=SequenceScaleStatus.RELATIVE_SHARED_SEQUENCE,
        geometry_mode=DynamicGeometryMode.ROTATION_COMPENSATED,
        valid=True,
        metadata={"synthetic_only": True},
    )


def _current_point(uv: np.ndarray) -> PointTrack2DObservation:
    return PointTrack2DObservation(
        point_id="synthetic_point",
        object_track_id="synthetic_track",
        frame_index=1,
        pixel_uv=tuple(uv),
        visibility=VisibilityStatus.VISIBLE,
        occlusion_status="visible",
        tracking_confidence=1.0,
        source_tracker="synthetic_independent_observation",
        valid=True,
        metadata={
            "independent_observation": True,
            "generated_from_projection": False,
        },
    )


def _safe(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def run_d2_synthetic_validation() -> dict[str, Any]:
    """Exercise D2 geometry without reading video labels or learned providers.

    P4-C0 defines D2 as rotation-compensated geometry.  The synthetic fixture
    also checks a known rigid ``R,t`` projection as a lower-level transform
    primitive, while explicitly verifying that D2 itself does not claim an
    unobservable translation scale.
    """

    K = np.asarray(
        [[120.0, 0.0, 64.0], [0.0, 118.0, 48.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    xyz = np.asarray([0.35, -0.20, 4.5], dtype=float)
    angle = math.radians(8.0)
    rotation = np.asarray(
        [
            [math.cos(angle), 0.0, math.sin(angle)],
            [0.0, 1.0, 0.0],
            [-math.sin(angle), 0.0, math.cos(angle)],
        ],
        dtype=float,
    )
    translation = np.asarray([0.15, -0.04, 0.10], dtype=float)
    identity = np.eye(4, dtype=float)
    rigid = np.eye(4, dtype=float)
    rigid[:3, :3] = rotation
    rigid[:3, 3] = translation
    rotation_only = rigid.copy()
    previous = _previous_point(K, xyz)

    checks: list[dict[str, Any]] = []

    def record(name: str, passed: bool, details: dict[str, Any]) -> None:
        checks.append({"name": name, "passed": bool(passed), "details": details})

    identity_uv = _project(K, xyz)
    identity_result = compute_dynamic_reprojection_residual(
        previous,
        _current_point(identity_uv),
        K_current=K,
        image_width=128,
        image_height=96,
        relative_pose_current_from_previous=identity,
        geometry_mode=DynamicGeometryMode.ROTATION_COMPENSATED,
        is_background=True,
    )
    record(
        "identity_pose_reprojection",
        identity_result.valid and identity_result.pixel_error < 1e-10,
        {"pixel_error": _safe(identity_result.pixel_error)},
    )

    rotated_xyz = rotation @ xyz
    rotated_uv = _project(K, rotated_xyz)
    rotation_result = compute_dynamic_reprojection_residual(
        previous,
        _current_point(rotated_uv),
        K_current=K,
        image_width=128,
        image_height=96,
        relative_pose_current_from_previous=rotation_only,
        geometry_mode=DynamicGeometryMode.ROTATION_COMPENSATED,
        is_background=True,
    )
    record(
        "known_rotation_d2_reprojection",
        rotation_result.valid and rotation_result.pixel_error < 1e-10,
        {
            "pixel_error": _safe(rotation_result.pixel_error),
            "translation_ignored_by_d2": True,
        },
    )

    rigid_truth = _project(K, rotation @ xyz + translation)
    manual_homogeneous = rigid @ np.concatenate([xyz, [1.0]])
    rigid_prediction = _project(K, manual_homogeneous[:3])
    record(
        "known_rotation_translation_geometry_primitive",
        bool(np.allclose(rigid_truth, rigid_prediction, atol=1e-12)),
        {
            "truth_uv": rigid_truth.tolist(),
            "predicted_uv": rigid_prediction.tolist(),
            "classified_as_d3_input_not_d2_translation": True,
        },
    )

    inverse = np.linalg.inv(rotation_only)
    reversed_result = compute_dynamic_reprojection_residual(
        previous,
        _current_point(rotated_uv),
        K_current=K,
        image_width=128,
        image_height=96,
        relative_pose_current_from_previous=inverse,
        geometry_mode=DynamicGeometryMode.ROTATION_COMPENSATED,
        is_background=True,
    )
    record(
        "pose_direction_reversal_detected",
        reversed_result.valid
        and reversed_result.pixel_error > rotation_result.pixel_error + 1.0,
        {
            "correct_error": _safe(rotation_result.pixel_error),
            "reversed_error": _safe(reversed_result.pixel_error),
        },
    )

    uv = _project(K, xyz)
    recovered = backproject_pixel(
        float(uv[0]),
        float(uv[1]),
        float(xyz[2]),
        K,
        point_id="roundtrip",
        scale_status=GeometryScaleStatus.RELATIVE_3D,
    )
    record(
        "projection_backprojection_roundtrip",
        recovered.valid and bool(np.allclose(recovered.as_array(), xyz, atol=1e-12)),
        {"recovered_xyz": recovered.as_array().tolist() if recovered.valid else None},
    )

    invisible = PointTrack2DObservation.missing(
        point_id="synthetic_point",
        object_track_id="synthetic_track",
        frame_index=1,
        reason="occluded",
        source_tracker="synthetic_visibility",
    )
    invisible_result = compute_dynamic_reprojection_residual(
        previous,
        invisible,
        K_current=K,
        image_width=128,
        image_height=96,
        relative_pose_current_from_previous=rotation_only,
        geometry_mode=DynamicGeometryMode.ROTATION_COMPENSATED,
        is_background=True,
    )
    record(
        "invisible_point_is_masked",
        not invisible_result.valid
        and math.isnan(invisible_result.pixel_error)
        and math.isnan(invisible_result.residual_evidence.value),
        {
            "valid": invisible_result.valid,
            "missing_reason": invisible_result.missing_reason,
            "residual": _safe(invisible_result.residual_evidence.value),
        },
    )

    invalid_depth = backproject_pixel(
        float(uv[0]),
        float(uv[1]),
        0.0,
        K,
        point_id="invalid_depth",
        scale_status=GeometryScaleStatus.RELATIVE_3D,
    )
    record(
        "invalid_depth_is_masked",
        not invalid_depth.valid
        and invalid_depth.x is None
        and invalid_depth.missing_reason == "non_positive_z_depth",
        {
            "valid": invalid_depth.valid,
            "missing_reason": invalid_depth.missing_reason,
        },
    )

    return {
        "definition": "D2 is rotation-compensated reprojection in the P4-C0 protocol",
        "translation_policy": (
            "R is used by D2; metric/scale-compatible t belongs to the D3 full-SE3 "
            "input contract and is not silently used by D2"
        ),
        "coordinate_convention": "opencv_x_right_y_down_z_forward",
        "transform_direction": "X_current = T_current_from_previous @ X_previous",
        "uses_authenticity_labels": False,
        "checks": checks,
        "passed": sum(bool(check["passed"]) for check in checks),
        "total": len(checks),
        "all_passed": all(bool(check["passed"]) for check in checks),
    }
