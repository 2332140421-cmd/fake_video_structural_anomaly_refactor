from __future__ import annotations

import math

import numpy as np

from semantic3d.method_completion import (
    PairResidualLocation,
    PairBranchPolicy,
    ProviderStatus,
    ScaleBranchName,
    ScaleEvidenceRole,
    ScaleEvidenceRouter,
    ScaleGeometryEvidence,
    map_residual_evidence,
    rank_object_scale_evidence,
)
from semantic3d.validity import ResidualEvidence


def evidence(branch: ScaleBranchName, *, valid: bool, role: ScaleEvidenceRole, failed=False):
    base = dict(
        video_id="v", clip_id="c", frame_id="f", object_id="o", track_id="t",
        branch_name=branch,
        branch_priority={
            ScaleBranchName.METRIC_SINGLE_OBJECT: 1,
            ScaleBranchName.TEMPORAL_SAME_OBJECT: 2,
            ScaleBranchName.RELATIVE_PAIR: 3,
        }[branch],
        evidence_role=role,
        residual_name="R",
        depth_type="metric" if branch == ScaleBranchName.METRIC_SINGLE_OBJECT else "relative",
        depth_unit="meter" if branch == ScaleBranchName.METRIC_SINGLE_OBJECT else "relative_local_unit",
        depth_definition="z_depth", coordinate_system="camera",
        localization_reference="object:o", provenance={}, config_sha256="", software_commit="",
    )
    if valid:
        return ScaleGeometryEvidence.observed(residual_value=0.2, confidence=0.8, **base)
    return ScaleGeometryEvidence.missing(
        failure_reason="provider_failed" if failed else "unavailable",
        provider_status=ProviderStatus.PROVIDER_FAILED if failed else ProviderStatus.BLOCKED,
        **base,
    )


def test_metric_is_primary_and_fallback_only_does_not_enumerate_pairs():
    calls = 0

    def supplier():
        nonlocal calls
        calls += 1
        return [evidence(ScaleBranchName.RELATIVE_PAIR, valid=True, role=ScaleEvidenceRole.FALLBACK)]

    result = ScaleEvidenceRouter("fallback_only").route(
        metric_evidence=evidence(ScaleBranchName.METRIC_SINGLE_OBJECT, valid=True, role=ScaleEvidenceRole.PRIMARY),
        pair_supplier=supplier,
        clip_observability="static",
    )
    assert result.selected_primary_branch == "metric_single_object_scale"
    assert calls == 0
    assert "relative_pair_scale_depth" not in result.executed_branches


def test_temporal_secondary_precedes_pair_fallback():
    calls = 0

    def supplier():
        nonlocal calls
        calls += 1
        return []

    result = ScaleEvidenceRouter().route(
        metric_evidence=evidence(ScaleBranchName.METRIC_SINGLE_OBJECT, valid=False, role=ScaleEvidenceRole.PRIMARY),
        temporal_evidence=evidence(ScaleBranchName.TEMPORAL_SAME_OBJECT, valid=True, role=ScaleEvidenceRole.TEMPORAL_SUPPORT),
        pair_supplier=supplier,
    )
    assert result.selected_primary_branch == "temporal_same_object_scale"
    assert calls == 0


def test_pair_fallback_runs_only_when_higher_priorities_unavailable():
    pair = evidence(ScaleBranchName.RELATIVE_PAIR, valid=True, role=ScaleEvidenceRole.FALLBACK)
    result = ScaleEvidenceRouter().route(
        metric_evidence=evidence(ScaleBranchName.METRIC_SINGLE_OBJECT, valid=False, role=ScaleEvidenceRole.PRIMARY),
        temporal_evidence=evidence(ScaleBranchName.TEMPORAL_SAME_OBJECT, valid=False, role=ScaleEvidenceRole.TEMPORAL_SUPPORT),
        pair_supplier=lambda: [pair],
    )
    assert result.selected_primary_branch == "relative_pair_scale_depth"
    assert result.fallback_used


def test_no_branch_is_nan_and_provider_failure_is_not_high_residual():
    result = ScaleEvidenceRouter().route(
        metric_evidence=evidence(
            ScaleBranchName.METRIC_SINGLE_OBJECT, valid=False,
            role=ScaleEvidenceRole.PRIMARY, failed=True,
        ),
        pair_supplier=lambda: [],
    )
    assert result.selected_primary_branch == "no_scale_evidence"
    assert result.evidence_confidence == 0.0
    assert all(not item.valid and math.isnan(item.residual_value) for item in result.evidences)


def test_static_clip_still_selects_metric_primary():
    result = ScaleEvidenceRouter().route(
        metric_evidence=evidence(ScaleBranchName.METRIC_SINGLE_OBJECT, valid=True, role=ScaleEvidenceRole.PRIMARY),
        clip_observability="static",
    )
    assert result.selected_primary_branch == "metric_single_object_scale"


def test_pair_evidence_maps_to_both_masks_relation_edge_and_ranking():
    mask_a = np.zeros((8, 8), bool)
    mask_b = np.zeros((8, 8), bool)
    mask_a[1:3, 1:3] = True
    mask_b[5:7, 5:7] = True
    pair = PairResidualLocation(
        ResidualEvidence.observed("R_sd_pair", 0.7, source_ids=("a", "b")),
        0,
        "a:b",
        "a",
        "b",
        mask_a,
        mask_b,
        (1.5, 1.5),
        (5.5, 5.5),
    )
    result = map_residual_evidence(
        image_shape=(8, 8), frame_indices=(0,), pair_residuals=(pair,)
    )
    assert result.object_scores == {"a": 0.7, "b": 0.7}
    assert result.relation_edges["a:b"]["residual"] == 0.7
    assert np.nanmax(result.frame_residual_map[0]) == 0.7
    assert rank_object_scale_evidence(result.object_scores) == (("a", 0.7), ("b", 0.7))
