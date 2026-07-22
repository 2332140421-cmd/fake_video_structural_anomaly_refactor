from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from semantic3d.depth_provider import DepthScaleStatus
from semantic3d.method_completion.audit import build_method_completion_audit
from semantic3d.method_completion.d2_validation import run_d2_synthetic_validation
from semantic3d.method_completion.d3_relations import (
    D3RelationObservation,
    D3RelationType,
    compare_d3_relations,
)
from semantic3d.method_completion.eligibility import (
    EligibilityRecord,
    summarize_eligibility,
)
from semantic3d.method_completion.localization import (
    ObjectResidualLocation,
    PointResidualLocation,
    map_residual_evidence,
)
from semantic3d.method_completion.observability import (
    ClipMotionMeasurements,
    ClipObservability,
    classify_clip_observability,
)
from semantic3d.method_completion.semantic_geometry import (
    AbsoluteSemanticScaleBranch,
    CrossFrameScaleStabilityBranch,
    RelativeScaleDepthBranch,
)
from semantic3d.scale_depth import ObjectObservation, ScalePrior
from semantic3d.validity import ResidualEvidence

from synthetic_geometry import synthetic_object_3d, synthetic_shared_3d_frame


ROOT = Path(__file__).resolve().parents[1]
PRIORS = {"person": ScalePrior(1.5, 2.0), "cup": ScalePrior(0.08, 0.12)}


def test_three_semantic_geometry_routes_are_separate() -> None:
    relative = synthetic_object_3d(
        "person_1", label="person", observed_scale=1.75, metric=False
    )
    frame = synthetic_shared_3d_frame((relative,), metric=False)
    absolute = AbsoluteSemanticScaleBranch(PRIORS).evaluate(frame, "person_1")
    assert not absolute.valid and math.isnan(absolute.value)
    assert absolute.missing_reason == "blocked_by_metric_scale_unavailable"

    metric = synthetic_object_3d(
        "person_1", label="person", observed_scale=1.75, metric=True
    )
    metric_evidence = AbsoluteSemanticScaleBranch(PRIORS).evaluate(
        synthetic_shared_3d_frame((metric,), metric=True), "person_1"
    )
    assert metric_evidence.valid and metric_evidence.value == 0.0

    obj_a = ObjectObservation("a", "person", 100.0, 10000.0, 2.0)
    obj_b = ObjectObservation("b", "cup", 25.0, 10000.0, 20.0)
    pair = RelativeScaleDepthBranch(PRIORS).evaluate(obj_a, obj_b)
    assert pair.valid and math.isfinite(pair.value)
    assert pair.metadata["semantic_geometry_route"] == "same_frame_relative_pair"


def test_cross_frame_scale_stability_requires_shared_scale() -> None:
    base = synthetic_object_3d(
        "person_1", label="person", observed_scale=2.0, metric=False
    )
    per_frame = replace(base, track_id="track_1", frame_index=0)
    blocked = CrossFrameScaleStabilityBranch().evaluate(
        per_frame, replace(per_frame, frame_index=1, observed_scale_3d=2.2)
    )
    assert not blocked.valid and math.isnan(blocked.value)
    assert blocked.missing_reason == "blocked_by_cross_frame_scale_unavailable"

    previous = replace(
        per_frame, depth_scale_status=DepthScaleStatus.RELATIVE_SHARED_SEQUENCE
    )
    stable = replace(previous, frame_index=1, observed_scale_3d=2.01)
    changed = replace(previous, frame_index=1, observed_scale_3d=3.0)
    stable_result = CrossFrameScaleStabilityBranch(tolerance=0.02).evaluate(
        previous, stable
    )
    changed_result = CrossFrameScaleStabilityBranch(tolerance=0.02).evaluate(
        previous, changed
    )
    assert stable_result.valid and stable_result.value == 0.0
    assert changed_result.valid and changed_result.value > 0.0


@pytest.mark.parametrize(
    ("background", "objects", "expected"),
    [
        (0.1, 0.1, ClipObservability.STATIC),
        (0.6, 0.5, ClipObservability.LOW_MOTION),
        (0.2, 3.0, ClipObservability.OBJECT_MOTION),
        (3.0, 0.2, ClipObservability.CAMERA_MOTION),
        (3.0, 3.0, ClipObservability.MIXED_MOTION),
    ],
)
def test_clip_observability_classes(
    background: float, objects: float, expected: ClipObservability
) -> None:
    result = classify_clip_observability(
        ClipMotionMeasurements(
            "clip", background, objects, True, True, quality=1.0
        )
    )
    assert result.valid and not result.provider_failed
    assert result.observability == expected


def test_motion_unreliable_is_distinct_from_static_and_provider_failure() -> None:
    failed = classify_clip_observability(
        ClipMotionMeasurements(
            "failed", None, None, False, False, 0.0,
            provider_failed=True, missing_reason="tracker_provider_failed",
        )
    )
    assert not failed.valid and failed.provider_failed
    assert failed.observability == ClipObservability.MOTION_UNRELIABLE


def test_eligibility_funnel_keeps_terminal_outcomes_distinct() -> None:
    records = [
        EligibilityRecord("r", "valid", True, True, True, True, metadata={"unit": "clip"}),
        EligibilityRecord("r", "blocked", True, False, False, False, blocked=True, reason="no_pose", metadata={"unit": "clip"}),
        EligibilityRecord("r", "failed", True, False, False, False, provider_failed=True, reason="provider_failed", metadata={"unit": "clip"}),
        EligibilityRecord("r", "na", False, False, False, False, not_applicable=True, reason="no_event", metadata={"unit": "clip"}),
    ]
    result = summarize_eligibility(records)["r"]
    assert result["total"] == 4 and result["valid"] == 1
    assert result["provider_failed"] == 1
    assert result["blocked"] == 1 and result["not_applicable"] == 1


def test_d2_synthetic_geometry_closure() -> None:
    result = run_d2_synthetic_validation()
    assert result["all_passed"]
    assert result["passed"] == result["total"] >= 6
    assert "rotation-compensated" in result["definition"]


def test_d3_relation_formulas_exist_but_executor_is_not_claimed() -> None:
    common = dict(
        relation_id="a:b",
        relation_type=D3RelationType.OBJECT_RELATIVE_DISTANCE,
        source_ids=("a", "b"),
        unit="relative_unit",
        coordinate_system_id="synthetic_world",
        pose_compensated=True,
        valid=True,
        quality=1.0,
    )
    previous = D3RelationObservation(frame_index=0, values=(2.0,), **common)
    same = compare_d3_relations(
        previous, D3RelationObservation(frame_index=1, values=(2.0,), **common)
    )
    changed = compare_d3_relations(
        previous, D3RelationObservation(frame_index=1, values=(4.0,), **common)
    )
    assert same.valid and same.value == 0.0
    assert changed.valid and changed.value == pytest.approx(math.log(2.0))
    assert changed.metadata["code_status"] == "interface_only"


def test_localization_maps_valid_evidence_and_preserves_nan() -> None:
    mask = np.zeros((6, 6), dtype=bool)
    mask[2:4, 2:4] = True
    valid = ResidualEvidence.observed("valid", 0.5, source_ids=("valid",))
    failed = ResidualEvidence.missing("failed", "provider_failed", source_ids=("bad",))
    result = map_residual_evidence(
        image_shape=(6, 6),
        frame_indices=(0, 1),
        point_residuals=(
            PointResidualLocation(valid, 0, (1.0, 1.0), "p"),
            PointResidualLocation(failed, 0, (5.0, 5.0), "bad"),
        ),
        object_residuals=(ObjectResidualLocation(valid, 0, "o", mask),),
    )
    assert result.frame_residual_map[0][1, 1] == pytest.approx(0.5)
    assert result.object_scores["o"] == pytest.approx(0.5)
    assert math.isnan(float(result.spatial_evidence_map[0][5, 5]))
    assert math.isnan(result.temporal_evidence_sequence[1])
    assert result.skipped_source_reasons["provider_failed"] == 1


def test_method_completion_outputs_are_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    result = build_method_completion_audit(ROOT, first)
    repeated = build_method_completion_audit(ROOT, second)
    assert result == repeated
    expected = {
        "method_branch_inventory.csv",
        "observability_audit.csv",
        "residual_eligibility_funnels.json",
        "d2_synthetic_validation.json",
        "d3_definition_and_status.json",
        "localization_interface_audit.json",
        "blocked_features.json",
        "METHOD_COMPLETION_REPORT.md",
        "validation_report.json",
    }
    assert {path.name for path in first.iterdir()} == expected
    for name in expected:
        assert hashlib.sha256((first / name).read_bytes()).digest() == hashlib.sha256(
            (second / name).read_bytes()
        ).digest()
    validation = json.loads((first / "validation_report.json").read_text())
    assert validation["ready_for_git_freeze"] is True
    assert validation["d2_six_video_verified"] is False
    assert validation["d3_code_status"] == "interface_only"
    assert validation["method_effectiveness_established"] is False


def test_frozen_strict_prior_hashes_are_unchanged() -> None:
    expected = {
        "scale_priors_strict_v1.yaml": "e86466e19fe3e1663fa855fdf73843cf7ab8b5b6c8fa8771aa46172f6b726a6b",
        "scale_priors_strict_v2.yaml": "3c55c8eb42e19d2447b00794085d8e6fa233c37a2071ef51f23b447d23ef268b",
    }
    for filename, digest in expected.items():
        assert hashlib.sha256((ROOT / "configs" / filename).read_bytes()).hexdigest() == digest
