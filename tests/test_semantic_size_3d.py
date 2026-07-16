from __future__ import annotations

import hashlib
import math
from pathlib import Path

import pytest

from semantic3d.scale_depth import ScalePrior
from semantic3d.static_3d import SemanticSize3DResidual

from synthetic_geometry import synthetic_object_3d, synthetic_shared_3d_frame


PRIORS = {
    "person": ScalePrior(1.5, 2.0),
    "cup": ScalePrior(0.08, 0.12),
    "anchor": ScalePrior(0.9, 1.1),
}


def test_metric_semantic_scale_inside_interval_is_zero() -> None:
    obj = synthetic_object_3d(
        "person_1", label="person", observed_scale=1.75, metric=True
    )
    frame = synthetic_shared_3d_frame((obj,), metric=True)
    evidence = SemanticSize3DResidual(PRIORS).evaluate_metric(frame, "person_1")
    assert evidence.valid
    assert evidence.value == 0.0
    assert evidence.metadata["calibration_status"] == "metric_calibrated"
    assert evidence.metadata["expected_min"] == 1.5
    assert evidence.metadata["expected_max"] == 2.0


def test_metric_semantic_scale_outside_interval_has_correct_distance() -> None:
    obj = synthetic_object_3d(
        "person_1", label="person", observed_scale=2.4, metric=True
    )
    frame = synthetic_shared_3d_frame((obj,), metric=True)
    evidence = SemanticSize3DResidual(PRIORS).evaluate_metric(frame, "person_1")
    assert evidence.valid
    assert evidence.value == pytest.approx(0.4)


def test_relative_per_frame_cannot_directly_use_metric_interval() -> None:
    obj = synthetic_object_3d(
        "person_1", label="person", observed_scale=2.0, metric=False
    )
    frame = synthetic_shared_3d_frame((obj,), metric=False)
    evidence = SemanticSize3DResidual(PRIORS).evaluate_metric(frame, "person_1")
    assert not evidence.valid
    assert math.isnan(evidence.value)
    assert evidence.missing_reason == "metric_calibration_required"


def test_independent_anchor_calibrates_other_object_but_not_itself() -> None:
    anchor = synthetic_object_3d(
        "anchor_1", label="anchor", observed_scale=2.0, metric=False
    )
    target = synthetic_object_3d(
        "person_1", label="person", observed_scale=3.5, metric=False
    )
    frame = synthetic_shared_3d_frame((anchor, target), metric=False)
    residual = SemanticSize3DResidual(PRIORS)
    evidence = residual.evaluate_relative_with_anchors(
        frame,
        "person_1",
        ("anchor_1",),
        anchors_are_independent=True,
    )
    assert evidence.valid
    assert evidence.value == 0.0
    assert evidence.metadata["estimated_scene_scale"] == pytest.approx(0.5)
    assert evidence.metadata["target_used_as_anchor"] is False

    circular = residual.evaluate_relative_with_anchors(
        frame,
        "anchor_1",
        ("anchor_1",),
        anchors_are_independent=True,
    )
    assert not circular.valid
    assert math.isnan(circular.value)
    assert circular.missing_reason == "anchor_self_evaluation_forbidden"


def test_leave_one_out_excludes_target_and_missing_anchor_is_nan() -> None:
    anchor = synthetic_object_3d(
        "anchor_1", label="anchor", observed_scale=2.0, metric=False
    )
    target = synthetic_object_3d(
        "person_1", label="person", observed_scale=3.5, metric=False
    )
    frame = synthetic_shared_3d_frame((anchor, target), metric=False)
    residual = SemanticSize3DResidual(PRIORS)
    evidence = residual.evaluate_leave_one_out(
        frame, "person_1", ("person_1", "anchor_1")
    )
    assert evidence.valid
    assert evidence.metadata["leave_one_out_target_excluded"] is True
    assert "person_1" not in evidence.metadata["anchor_object_ids"]

    missing = residual.evaluate_leave_one_out(
        frame, "person_1", ("person_1",)
    )
    assert not missing.valid
    assert math.isnan(missing.value)
    assert missing.missing_reason == "no_independent_scale_anchor"


def test_same_frame_ratio_is_relative_not_unary_metric_claim() -> None:
    cup = synthetic_object_3d(
        "cup_1", label="cup", observed_scale=0.2, metric=False
    )
    person = synthetic_object_3d(
        "person_1", label="person", observed_scale=3.5, metric=False
    )
    frame = synthetic_shared_3d_frame((cup, person), metric=False)
    evidence = SemanticSize3DResidual(PRIORS).evaluate_same_frame_relative_ratio(
        frame, "cup_1", "person_1"
    )
    assert evidence.valid
    assert evidence.metadata["metric_unary_claim"] is False
    assert evidence.metadata["calibration_source"] == "same_frame_relative_ratio"


def test_scene_api_returns_one_evidence_per_object_not_pair_matrix() -> None:
    cup = synthetic_object_3d(
        "cup_1", label="cup", observed_scale=0.10, metric=True
    )
    person = synthetic_object_3d(
        "person_1", label="person", observed_scale=1.75, metric=True
    )
    frame = synthetic_shared_3d_frame((cup, person), metric=True)
    evidence_by_object = SemanticSize3DResidual(PRIORS).evaluate_metric_scene(frame)
    assert set(evidence_by_object) == {"cup_1", "person_1"}
    assert all(evidence.valid for evidence in evidence_by_object.values())
    assert all(len(evidence.source_ids) == 1 for evidence in evidence_by_object.values())


def test_frozen_strict_prior_hashes_remain_unchanged() -> None:
    root = Path(__file__).resolve().parents[1]
    expected = {
        "scale_priors_strict_v1.yaml": "e86466e19fe3e1663fa855fdf73843cf7ab8b5b6c8fa8771aa46172f6b726a6b",
        "scale_priors_strict_v2.yaml": "3c55c8eb42e19d2447b00794085d8e6fa233c37a2071ef51f23b447d23ef268b",
    }
    for filename, digest in expected.items():
        payload = (root / "configs" / filename).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == digest
