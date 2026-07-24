"""Tests for P4-C3B-M6 missing-aware fusion and localization."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from semantic3d.evidence_fusion import (
    EvidenceBranchGroup,
    FrameFusionEvidence,
    UnifiedEvidence,
    branch_dropout_audit,
    build_temporal_evidence_sequences,
    fuse_unified_evidence,
    map_unified_evidence_spatially,
    missingness_only_features,
    rank_object_and_track_evidence,
    route_evidence_branches,
)
from semantic3d.evidence_fusion import smoke as smoke_module


def _observed(
    evidence_id: str,
    value: float,
    group: EvidenceBranchGroup,
    *,
    confidence: float = 1.0,
    uncertainty: float = 0.0,
    frame_index: int = 1,
    object_id: str = "",
    track_id: str = "",
    spatial_reference: dict[str, object] | None = None,
) -> UnifiedEvidence:
    return UnifiedEvidence.observed(
        evidence_id=evidence_id,
        residual_value=value,
        confidence=confidence,
        uncertainty=uncertainty,
        provider_status="executed_valid",
        branch_name=f"{group.value}_residual",
        branch_group=group,
        video_id="video",
        clip_id="clip",
        frame_id=f"frame_{frame_index:06d}",
        frame_index=frame_index,
        object_id=object_id,
        track_id=track_id,
        spatial_reference=spatial_reference or {"kind": "reference_only"},
        temporal_reference={"frame_index": frame_index},
        provenance={"synthetic_test": True},
    )


def _missing(
    evidence_id: str,
    group: EvidenceBranchGroup,
    *,
    applicable: bool = True,
    provider_status: str = "blocked_by_input",
    reason: str = "input_missing",
) -> UnifiedEvidence:
    return UnifiedEvidence.unavailable(
        evidence_id=evidence_id,
        applicable=applicable,
        provider_status=provider_status,
        failure_reason=reason,
        branch_name=f"{group.value}_residual",
        branch_group=group,
        video_id="video",
        clip_id="clip",
        frame_id="frame_000001",
        frame_index=1,
        spatial_reference={"kind": "reference_only"},
        temporal_reference={"frame_index": 1},
        provenance={"synthetic_test": True},
    )


def test_unified_evidence_preserves_zero_and_missing_nan() -> None:
    normal = _observed(
        "normal",
        0.0,
        EvidenceBranchGroup.STATIC_METRIC_GEOMETRY,
    )
    missing = _missing(
        "missing",
        EvidenceBranchGroup.STATIC_RELATIVE_GEOMETRY,
    )
    assert normal.valid and normal.residual_value == 0.0
    assert not missing.valid and math.isnan(missing.residual_value)


def test_provider_failure_cannot_be_valid_evidence() -> None:
    with pytest.raises(ValueError, match="Provider failure"):
        UnifiedEvidence.observed(
            evidence_id="bad",
            residual_value=1.0,
            confidence=1.0,
            uncertainty=0.0,
            provider_status="provider_failed",
            branch_name="bad",
            branch_group=EvidenceBranchGroup.D2_POSE_REPROJECTION,
        )


def test_missingness_changes_confidence_not_existing_risk() -> None:
    observed = _observed(
        "observed",
        0.8,
        EvidenceBranchGroup.STATIC_METRIC_GEOMETRY,
    )
    complete = fuse_unified_evidence((observed,))
    incomplete = fuse_unified_evidence(
        (
            observed,
            _missing(
                "missing",
                EvidenceBranchGroup.STATIC_RELATIVE_GEOMETRY,
            ),
        )
    )
    assert incomplete.risk_score == pytest.approx(complete.risk_score)
    assert incomplete.evidence_confidence < complete.evidence_confidence
    assert incomplete.available_weight_ratio == pytest.approx(0.5)


def test_provider_failure_does_not_become_risk() -> None:
    failed = _missing(
        "failed",
        EvidenceBranchGroup.D2_POSE_REPROJECTION,
        provider_status="provider_failed",
        reason="optical_flow_provider_failed",
    )
    result = fuse_unified_evidence((failed,))
    features = missingness_only_features((failed,))
    assert not result.valid
    assert math.isnan(result.risk_score)
    assert features["provider_failure_count"] == 1
    assert features["used_in_risk"] == 0


def test_low_quality_branch_has_reduced_risk_influence() -> None:
    high_quality = _observed(
        "high_quality",
        0.2,
        EvidenceBranchGroup.STATIC_METRIC_GEOMETRY,
        confidence=1.0,
    )
    low_quality_high_residual = _observed(
        "low_quality",
        3.0,
        EvidenceBranchGroup.D2_POSE_REPROJECTION,
        confidence=0.05,
    )
    low_quality_result = fuse_unified_evidence(
        (high_quality, low_quality_high_residual)
    )
    full_quality_result = fuse_unified_evidence(
        (
            high_quality,
            _observed(
                "full_quality",
                3.0,
                EvidenceBranchGroup.D2_POSE_REPROJECTION,
                confidence=1.0,
            ),
        )
    )
    assert low_quality_result.risk_score < full_quality_result.risk_score
    assert all(
        item.quality_adjusted_weight <= item.configured_weight
        for item in low_quality_result.branch_contributions
    )


def test_static_and_dynamic_pose_routes_are_distinct() -> None:
    static = route_evidence_branches(
        "static",
        pose_available=False,
        event_observed=False,
    )
    assert static[EvidenceBranchGroup.STATIC_METRIC_GEOMETRY].status == "enabled"
    assert (
        static[EvidenceBranchGroup.D2_POSE_REPROJECTION].status
        == "not_applicable"
    )
    dynamic_without_pose = route_evidence_branches(
        "camera_motion",
        pose_available=False,
        event_observed=False,
    )
    assert dynamic_without_pose[EvidenceBranchGroup.D1_LOCAL_MOTION].status == "enabled"
    assert (
        dynamic_without_pose[EvidenceBranchGroup.D2_POSE_REPROJECTION].status
        == "blocked_by_input"
    )
    assert (
        dynamic_without_pose[EvidenceBranchGroup.D3_STRUCTURAL_RELATION].status
        == "blocked_by_input"
    )


def test_branch_dropout_is_diagnostic_and_deterministic() -> None:
    evidence = (
        _observed("a", 0.2, EvidenceBranchGroup.STATIC_METRIC_GEOMETRY),
        _observed("b", 0.6, EvidenceBranchGroup.TEMPORAL_SCALE),
    )
    first = branch_dropout_audit(evidence)
    second = branch_dropout_audit(evidence)
    assert first == second
    assert len(first) == len(EvidenceBranchGroup)
    assert {row["dropped_branch"] for row in first} == {
        group.value for group in EvidenceBranchGroup
    }


def test_temporal_sequence_preserves_missing_and_has_no_formal_threshold() -> None:
    frames = (
        FrameFusionEvidence(
            "video", "clip", 0, "f0", 0.1, 0.8, 1, 1.0, True, ""
        ),
        FrameFusionEvidence(
            "video",
            "clip",
            1,
            "f1",
            float("nan"),
            0.0,
            0,
            0.0,
            False,
            "no_valid_fusion_evidence",
        ),
        FrameFusionEvidence(
            "video", "clip", 2, "f2", 0.4, 0.7, 1, 1.0, True, ""
        ),
    )
    sequences = build_temporal_evidence_sequences(
        frames,
        smoothing_window=1,
        diagnostic_threshold=None,
    )
    assert len(sequences) == 1
    sequence = sequences[0]
    assert math.isnan(sequence.smoothed_risk[1])
    assert sequence.intervals == ()
    assert not sequence.formal_threshold_selected


def test_spatial_maps_and_rankings_preserve_native_references() -> None:
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:5, 3:6] = True
    evidence = (
        _observed(
            "object",
            0.7,
            EvidenceBranchGroup.STATIC_METRIC_GEOMETRY,
            object_id="obj",
            track_id="trk",
            spatial_reference={"kind": "object_mask", "mask": mask},
        ),
        _observed(
            "reference",
            0.9,
            EvidenceBranchGroup.TEMPORAL_SCALE,
            object_id="obj_ref",
            track_id="trk_ref",
            spatial_reference={"kind": "reference_only"},
        ),
        _missing(
            "missing",
            EvidenceBranchGroup.D2_POSE_REPROJECTION,
        ),
    )
    products = map_unified_evidence_spatially(evidence, image_shape=(8, 8))
    key = ("video", "clip", 1)
    assert np.isfinite(products.object_evidence_maps[key][mask]).all()
    assert np.isnan(products.object_evidence_maps[key][0, 0])
    rankings = rank_object_and_track_evidence(products)
    ids = {row["identity_id"] for row in rankings}
    assert {"obj", "trk", "obj_ref", "trk_ref"} <= ids
    reference_row = next(
        row for row in products.manifest_rows if row["evidence_id"] == "reference"
    )
    assert not reference_row["rasterized"]
    assert reference_row["failure_reason"] == "spatial_support_reference_only"


def test_video_specific_source_shape_keeps_portrait_support_in_frame(
    tmp_path: Path,
) -> None:
    point_csv = tmp_path / "point.csv"
    headers = [
        "evidence_id",
        "video_id",
        "clip_id",
        "frame_t",
        "frame_t1",
        "object_id",
        "track_id",
        "point_id",
        "point_reprojection_residual",
        "visibility_status",
        "pose_confidence",
        "point_confidence",
        "valid",
        "failure_reason",
        "provider_status",
        "metadata",
    ]
    with point_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerow(
            {
                "evidence_id": "portrait",
                "video_id": "portrait_video",
                "clip_id": "clip",
                "frame_t": 0,
                "frame_t1": 1,
                "object_id": "obj",
                "track_id": "trk",
                "point_id": "p",
                "point_reprojection_residual": 0.2,
                "visibility_status": "visible",
                "pose_confidence": 1.0,
                "point_confidence": 1.0,
                "valid": True,
                "failure_reason": "",
                "provider_status": "estimated_valid",
                "metadata": json.dumps({"predicted_uv": [360.0, 1200.0]}),
            }
        )
    from semantic3d.evidence_fusion.adapters import _adapt_d2

    evidence = _adapt_d2(
        {"point": point_csv, "boundary": tmp_path / "none.csv", "object": tmp_path / "none2.csv"},
        {},
        source_image_shapes={"portrait_video": (1280, 720)},
        fallback_source_image_shape=(720, 1280),
    )
    products = map_unified_evidence_spatially(evidence, image_shape=(180, 320))
    assert sum(row["rasterized"] for row in products.manifest_rows) == 1


def test_smoke_writes_required_outputs_without_provider_inference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:6, 2:6] = True
    persisted = (
        _observed(
            "static",
            0.4,
            EvidenceBranchGroup.STATIC_METRIC_GEOMETRY,
            object_id="obj",
            track_id="trk",
            spatial_reference={"kind": "object_mask", "mask": mask},
        ),
    )
    monkeypatch.setattr(
        smoke_module,
        "load_persisted_unified_evidence",
        lambda *_args, **_kwargs: (
            persisted,
            {
                "loaded_evidence_count": 1,
                "provider_inference_executed": False,
                "authenticity_labels_read": False,
            },
        ),
    )
    config = {
        "schema_version": "test.m6",
        "inputs": {},
        "output_dir": "outputs/m6",
        "clip_aliases": {},
        "clip_routes": [
            {
                "video_id": "video",
                "clip_id": "clip",
                "observability": "static",
                "pose_available": False,
                "event_observed": False,
            },
            {
                "video_id": "dynamic",
                "clip_id": "dynamic_clip",
                "observability": "camera_motion",
                "pose_available": False,
                "event_observed": False,
            },
        ],
        "branch_weights": {group.value: 1.0 for group in EvidenceBranchGroup},
        "fusion": {"top_k": 3},
        "temporal_localization": {
            "smoothing_window": 1,
            "diagnostic_threshold": None,
            "max_gap": 1,
            "minimum_duration": 1,
        },
        "localization": {
            "source_image_shape": [8, 8],
            "audit_map_shape": [8, 8],
        },
        "coverage_groups": {"video": "source_a", "dynamic": "source_b"},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    validation = smoke_module.run_evidence_fusion_smoke(tmp_path, config_path)
    output = tmp_path / "outputs/m6"
    required = {
        "evidence_schema_audit.json",
        "branch_availability_audit.csv",
        "risk_confidence_baseline.csv",
        "branch_contribution_audit.csv",
        "temporal_evidence_sequences.json",
        "spatial_evidence_manifest.csv",
        "object_track_rankings.csv",
        "missingness_shortcut_audit.json",
        "localization_interface_audit.json",
        "validation_report.json",
        "EVIDENCE_FUSION_REPORT.md",
    }
    assert required <= {path.name for path in output.iterdir()}
    assert validation["missing_aware_fusion_complete"]
    assert validation["static_video_detection_supported"]
    assert validation["pose_missing_fallback_supported"]
    assert not validation["provider_inference_executed"]
    assert not validation["formal_threshold_selected"]
    assert not validation["formal_training_executed"]
    assert not validation["method_effectiveness_established"]
