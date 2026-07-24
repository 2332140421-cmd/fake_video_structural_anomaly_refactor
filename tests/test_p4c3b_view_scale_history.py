from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from semantic3d.geometry.camera import CameraObservation, CoordinateConvention
from semantic3d.method_completion import (
    CameraMotionClass,
    CameraViewObservation,
    ObjectViewInput,
    PoseEstimateStatus,
    ProviderStatus,
    ScaleHistoryObservation,
    TemporalSameObjectScaleBranch,
    ViewpointClass,
    evaluate_object_view,
)
from semantic3d.method_completion.view_scale_smoke import (
    PROTOCOL_HASHES,
    STRICT_HASHES,
    run_view_scale_history_smoke,
)


ROOT = Path(__file__).resolve().parents[1]


def _view_input(**updates):
    values = {
        "object_id": "object",
        "track_id": "track",
        "class_name": "car",
        "bbox": (100.0, 80.0, 300.0, 280.0),
        "image_width": 640,
        "image_height": 480,
        "detection_confidence": 0.9,
        "mask_area": 32000.0,
        "bbox_area": 40000.0,
        "viewpoint_hint": ViewpointClass.FRONTAL,
        "pose_estimate_status": PoseEstimateStatus.RESOLVED,
        "view_confidence": 0.9,
    }
    values.update(updates)
    return ObjectViewInput(**values)


def _history(
    frame: int,
    size: float,
    *,
    track: str = "track",
    fingerprint: str = "K0",
    intrinsics=(500.0, 500.0, 320.0, 240.0),
    valid: bool = True,
    observable: bool = True,
    failure_reason: str = "",
    **updates,
):
    values = {
        "video_id": "video",
        "clip_id": "clip",
        "frame_id": f"frame_{frame}",
        "frame_index": frame,
        "track_id": track,
        "object_id": f"object_{frame}",
        "dimension_type": "camera_x_visible_extent_m",
        "size_value": size,
        "size_unit": "meter",
        "temporal_mode": "metric",
        "depth_provider": "metric_provider",
        "depth_definition": "z_depth",
        "intrinsics_fingerprint": fingerprint,
        "intrinsics_source": "model_predicted",
        "depth_scale_alignment_status": "metric_model_predicted_per_frame",
        "quality": 0.9,
        "valid": valid,
        "dimension_observable": observable,
        "failure_reason": failure_reason,
        "metadata": {"intrinsics_parameters": list(intrinsics)},
    }
    values.update(updates)
    return ScaleHistoryObservation(**values)


def test_camera_view_fov_and_static_motion_contract():
    camera = CameraObservation(
        K=np.asarray(
            [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]]
        ),
        distortion=None,
        T_world_camera=None,
        T_camera_world=None,
        image_width=640,
        image_height=480,
        coordinate_convention=CoordinateConvention.OPENCV,
        intrinsics_source="model_predicted",
        pose_source="unavailable",
        valid=True,
        quality=0.7,
    )
    view = CameraViewObservation.from_camera(
        "frame",
        camera,
        camera_motion_class=CameraMotionClass.STATIC,
        intrinsics_confidence=float("nan"),
    )
    assert view.valid
    assert view.fov_x == pytest.approx(65.238486, rel=1e-6)
    assert view.camera_motion_class == CameraMotionClass.STATIC
    assert view.pose_status == "unavailable"


def test_frontal_lateral_and_unknown_viewpoints_are_not_conflated():
    frontal = evaluate_object_view(_view_input())
    lateral = evaluate_object_view(
        _view_input(viewpoint_hint=ViewpointClass.LATERAL)
    )
    unknown = evaluate_object_view(
        _view_input(viewpoint_hint=ViewpointClass.UNKNOWN, view_confidence=float("nan"))
    )
    assert frontal.viewpoint_class == ViewpointClass.FRONTAL
    assert frontal.height_observable and frontal.width_observable
    assert not frontal.length_observable
    assert lateral.viewpoint_class == ViewpointClass.LATERAL
    assert lateral.height_observable and lateral.length_observable
    assert not lateral.width_observable
    assert unknown.viewpoint_class == ViewpointClass.UNKNOWN
    assert not any(
        (
            unknown.height_observable,
            unknown.width_observable,
            unknown.length_observable,
            unknown.depth_extent_observable,
        )
    )


def test_dimensions_are_independent_and_person_pose_is_gated():
    standing = evaluate_object_view(
        _view_input(
            class_name="person",
            pose_estimate_status=PoseEstimateStatus.UPRIGHT_FULL_BODY,
        )
    )
    sitting = evaluate_object_view(
        _view_input(
            class_name="person",
            pose_estimate_status=PoseEstimateStatus.SITTING,
        )
    )
    assert standing.height_observable
    assert not standing.width_observable
    assert not standing.length_observable
    assert not sitting.height_observable


def test_occlusion_and_border_contact_reject_only_affected_measurements():
    border = evaluate_object_view(
        _view_input(bbox=(0.0, 80.0, 300.0, 280.0))
    )
    occluded = evaluate_object_view(_view_input(occlusion_ratio=0.8))
    assert border.height_observable
    assert not border.width_observable
    assert not occluded.height_observable
    assert not occluded.width_observable


def test_temporal_constant_and_single_frame_jump():
    branch = TemporalSameObjectScaleBranch(reference_method="previous_valid")
    history = [_history(0, 2.0)]
    stable = branch.evaluate(_history(1, 2.0), history)
    jump = branch.evaluate(_history(1, 4.0), history)
    assert stable.valid and stable.residual_value == pytest.approx(0.0)
    assert jump.valid and jump.residual_value == pytest.approx(math.log(2.0))


def test_rolling_and_robust_track_median_resist_one_outlier():
    history = [_history(0, 2.0), _history(1, 2.0), _history(2, 20.0)]
    for method in ("rolling_median", "robust_track_median"):
        branch = TemporalSameObjectScaleBranch(
            reference_method=method, min_valid_history=2, reference_window=5
        )
        result = branch.evaluate(_history(3, 2.0), history)
        assert result.valid
        assert result.residual_value == pytest.approx(0.0)


def test_id_switch_intrinsics_change_and_track_gap_are_explicitly_blocked():
    id_switch = TemporalSameObjectScaleBranch(
        reference_method="previous_valid"
    ).evaluate(_history(1, 2.0), [_history(0, 2.0, track="other")])
    assert id_switch.failure_reason == "track_id_switch_or_mismatch"

    intrinsics = TemporalSameObjectScaleBranch(
        reference_method="previous_valid",
        max_intrinsics_relative_change=0.03,
    )
    compatible = intrinsics.evaluate(
        _history(
            1,
            2.0,
            fingerprint="K1",
            intrinsics=(505.0, 505.0, 320.0, 240.0),
        ),
        [_history(0, 2.0)],
    )
    incompatible = intrinsics.evaluate(
        _history(
            1,
            2.0,
            fingerprint="K2",
            intrinsics=(700.0, 700.0, 320.0, 240.0),
        ),
        [_history(0, 2.0)],
    )
    assert compatible.valid
    assert incompatible.failure_reason == "camera_intrinsics_changed"

    gap = TemporalSameObjectScaleBranch(
        reference_method="previous_valid", max_frame_gap=1
    ).evaluate(_history(3, 2.0), [_history(0, 2.0)])
    assert gap.failure_reason == "track_history_frame_gap_too_large"


def test_static_state_does_not_block_scale_branch_and_unobservable_is_nan():
    branch = TemporalSameObjectScaleBranch(reference_method="previous_valid")
    result = branch.evaluate(
        _history(1, 2.0, metadata={"camera_motion_class": "static"}),
        [_history(0, 2.0, metadata={"camera_motion_class": "static"})],
    )
    assert result.valid
    missing = _history(
        2,
        float("nan"),
        valid=False,
        observable=False,
        failure_reason="dimension_not_observable",
    )
    blocked = branch.evaluate(missing, [_history(1, 2.0)])
    assert not blocked.valid
    assert math.isnan(blocked.residual_value)
    assert blocked.failure_reason == "dimension_not_observable"


def test_m3_offline_smoke_writes_required_outputs(tmp_path):
    output = tmp_path / "m3"
    result = run_view_scale_history_smoke(
        config_path=ROOT / "configs/p4c3b_view_scale_history_v1.yaml",
        output_dir=output,
        project_root=ROOT,
    )
    required = {
        "camera_view_audit.csv",
        "object_view_audit.csv",
        "dimension_observability_audit.csv",
        "metric_single_object_execution.csv",
        "track_size_history.csv",
        "temporal_scale_execution.csv",
        "static_clip_branch_audit.csv",
        "eligibility_funnels.json",
        "validation_report.json",
        "VIEW_SCALE_HISTORY_REPORT.md",
    }
    assert all((output / name).exists() for name in required)
    assert result["camera_view_model_complete"]
    assert result["metric_single_object_real_smoke_verified"]
    assert result["temporal_metric_real_smoke_verified"]
    assert result["method_effectiveness_established"] is False
    rows = list(csv.DictReader((output / "temporal_scale_execution.csv").open()))
    assert any(row["valid"] == "True" for row in rows)
    assert all(
        row["residual"].lower() == "nan"
        for row in rows
        if row["valid"] == "False"
    )


def test_frozen_protocol_and_prior_hashes_are_unchanged():
    expected = {**STRICT_HASHES, **PROTOCOL_HASHES}
    for name, digest in expected.items():
        data = (ROOT / "configs" / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == digest
