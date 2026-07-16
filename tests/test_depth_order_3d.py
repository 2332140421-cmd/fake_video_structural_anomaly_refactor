from __future__ import annotations

import math

from semantic3d.static_3d import DepthOrder3DResidual, EvidenceRole

from synthetic_geometry import synthetic_object_3d, synthetic_shared_3d_frame


def _frame(front_z: float, back_z: float):
    front = synthetic_object_3d(
        "front", center=(0.0, 0.0, front_z), metric=False
    )
    back = synthetic_object_3d(
        "back", center=(0.1, 0.0, back_z), metric=False
    )
    return synthetic_shared_3d_frame((front, back), metric=False)


def test_center_order_is_diagnostic_not_anomaly() -> None:
    evidence = DepthOrder3DResidual().center_depth_order(_frame(3.0, 7.0), "front", "back")
    assert evidence.valid
    assert evidence.value == 4.0
    assert evidence.metadata["relation"] == "a_in_front_of_b"
    assert evidence.metadata["evidence_role"] == EvidenceRole.DIAGNOSTIC.value
    assert evidence.metadata["anomaly_residual"] is False


def test_occlusion_direction_consistent_has_zero_residual() -> None:
    evidence = DepthOrder3DResidual().occlusion_depth_consistency(
        _frame(3.0, 7.0),
        "front",
        "back",
        front_object_id="front",
        overlap_ratio=0.4,
    )
    assert evidence.valid
    assert evidence.value == 0.0
    assert evidence.metadata["consistent"] is True


def test_occlusion_direction_contradiction_has_positive_residual() -> None:
    evidence = DepthOrder3DResidual().occlusion_depth_consistency(
        _frame(7.0, 3.0),
        "front",
        "back",
        front_object_id="front",
        overlap_ratio=0.4,
    )
    assert evidence.valid
    assert evidence.value > 0.0
    assert evidence.metadata["consistent"] is False


def test_missing_occlusion_relation_is_nan() -> None:
    evidence = DepthOrder3DResidual().occlusion_depth_consistency(
        _frame(3.0, 7.0),
        "front",
        "back",
        front_object_id=None,
        overlap_ratio=None,
    )
    assert not evidence.valid
    assert math.isnan(evidence.value)
    assert evidence.missing_reason == "no_occlusion_relation_evidence"

