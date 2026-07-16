from __future__ import annotations

import math

from semantic3d.static_3d import (
    EvidenceRole,
    ReconstructionQualityEvidence,
    Static3DContext,
    reprojection_cycle_evidence,
)

from synthetic_geometry import synthetic_object_3d, synthetic_shared_3d_frame


def test_reconstruction_quality_is_gate_evidence_not_anomaly_score() -> None:
    obj = synthetic_object_3d("person_1", label="person", metric=False, quality=0.8)
    frame = synthetic_shared_3d_frame(
        (obj,), metric=False, approximate_intrinsics=True
    )

    evidence = ReconstructionQualityEvidence().evaluate(
        frame, "person_1", reprojection_cycle_error=0.0
    )

    assert evidence.valid
    assert evidence.metadata["evidence_role"] == EvidenceRole.QUALITY.value
    assert evidence.metadata["quality_is_probability"] is False
    assert evidence.metadata["approximate_intrinsics_quality"] == 0.5
    assert evidence.metadata["valid_3d_point_ratio"] == 1.0
    assert 0.0 <= evidence.value <= 1.0
    assert Static3DContext(frame).object_by_id("person_1") is obj


def test_invalid_reconstruction_quality_is_nan() -> None:
    obj = synthetic_object_3d("missing", valid=False)
    frame = synthetic_shared_3d_frame((obj,))
    evidence = ReconstructionQualityEvidence().evaluate(frame, "missing")
    assert not evidence.valid
    assert math.isnan(evidence.value)
    assert evidence.missing_reason == "synthetic_missing_object"


def test_same_point_reprojection_cycle_is_qa_only() -> None:
    evidence = reprojection_cycle_evidence(1e-9, source_ids=("person_1",))
    assert evidence.valid
    assert evidence.name == "reconstruction_cycle_error"
    assert evidence.metadata["evidence_role"] == EvidenceRole.QA.value
    assert evidence.metadata["anomaly_residual"] is False
    assert evidence.metadata["independent_observation"] is False

