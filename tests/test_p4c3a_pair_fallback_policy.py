from __future__ import annotations

import hashlib
from pathlib import Path

from semantic3d.method_completion import (
    PairBranchPolicy,
    ScaleBranchName,
    ScaleEvidenceRole,
    ScaleEvidenceRouter,
    ScaleGeometryEvidence,
    strict_v2_row_to_scale_evidence,
)


ROOT = Path(__file__).resolve().parents[1]
STRICT_HASHES = {
    "scale_priors_strict_v1.yaml": "e86466e19fe3e1663fa855fdf73843cf7ab8b5b6c8fa8771aa46172f6b726a6b",
    "scale_priors_strict_v2.yaml": "3c55c8eb42e19d2447b00794085d8e6fa233c37a2071ef51f23b447d23ef268b",
}


def missing_metric():
    return ScaleGeometryEvidence.missing(
        video_id="v", clip_id="c", frame_id="f", object_id="o", track_id="t",
        branch_name=ScaleBranchName.METRIC_SINGLE_OBJECT, branch_priority=1,
        evidence_role=ScaleEvidenceRole.PRIMARY, residual_name="R_metric_abs",
        failure_reason="metric_scale_unavailable", depth_type="relative",
        depth_unit="relative_local_unit", depth_definition="z_depth",
        coordinate_system="camera", localization_reference="object:o", provenance={},
        config_sha256="", software_commit="",
    )


def pair_evidence():
    return strict_v2_row_to_scale_evidence(
        {
            "object_a_id": "a", "object_b_id": "b", "valid": True,
            "rsd_log": 0.25, "gate_score_a": 0.9, "gate_score_b": 0.8,
            "prior_source": "physical",
        },
        video_id="v", clip_id="c", frame_id="f",
    )


def test_audit_only_pair_cannot_become_primary():
    result = ScaleEvidenceRouter(PairBranchPolicy.AUDIT_ONLY).route(
        metric_evidence=missing_metric(), pair_supplier=lambda: [pair_evidence()]
    )
    assert result.selected_primary_branch == "no_scale_evidence"
    pair = result.evidences[-1]
    assert pair.evidence_role == ScaleEvidenceRole.AUDIT_CROSSCHECK
    assert not pair.eligible_for_primary_aggregation
    assert result.audit_crosscheck_used


def test_disabled_policy_never_calls_pair_supplier():
    calls = 0

    def supplier():
        nonlocal calls
        calls += 1
        return [pair_evidence()]

    ScaleEvidenceRouter("disabled").route(metric_evidence=missing_metric(), pair_supplier=supplier)
    assert calls == 0


def test_strict_v1_v2_frozen_hashes_are_unchanged():
    for name, expected in STRICT_HASHES.items():
        actual = hashlib.sha256((ROOT / "configs" / name).read_bytes()).hexdigest()
        assert actual == expected


def test_strict_v2_adapter_reuses_row_without_recomputing_formula():
    evidence = pair_evidence()
    assert evidence.residual_value == 0.25
    assert evidence.provenance["strict_v2_row_reused"] is True
    assert evidence.provenance["strict_v2_formula_recomputed"] is False
