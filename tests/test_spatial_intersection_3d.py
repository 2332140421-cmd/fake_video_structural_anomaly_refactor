from __future__ import annotations

import math

from semantic3d.static_3d import EvidenceRole, SpatialIntersection3DResidual

from synthetic_geometry import synthetic_object_3d, synthetic_shared_3d_frame


def test_aabb_overlap_is_diagnostic_not_definite_penetration() -> None:
    first = synthetic_object_3d("a", center=(0.0, 0.0, 5.0), metric=False)
    second = synthetic_object_3d("b", center=(0.2, 0.0, 5.0), metric=False)
    frame = synthetic_shared_3d_frame((first, second), metric=False)
    evidence = SpatialIntersection3DResidual().evaluate(frame, "a", "b")
    assert evidence.valid
    assert evidence.value > 0.0
    assert evidence.metadata["evidence_role"] == EvidenceRole.DIAGNOSTIC.value
    assert evidence.metadata["approximation"] == "aabb_or_sparse_points"
    assert evidence.metadata["definite_physical_penetration"] is False
    assert evidence.metadata["recommended_weight"] <= 0.1


def test_insufficient_spatial_points_is_nan() -> None:
    first = synthetic_object_3d("a", metric=False)
    second = synthetic_object_3d("b", metric=False)
    frame = synthetic_shared_3d_frame((first, second), metric=False)
    evidence = SpatialIntersection3DResidual(minimum_points=20).evaluate(
        frame, "a", "b"
    )
    assert not evidence.valid
    assert math.isnan(evidence.value)
    assert evidence.missing_reason == "insufficient_3d_spatial_evidence"

