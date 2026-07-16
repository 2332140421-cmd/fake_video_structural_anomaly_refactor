"""Common types and invariants for static 3D ResidualEvidence producers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from ..shared_3d_observation import Object3DObservation, Shared3DFrameObservation
from ..validity import ResidualEvidence


class EvidenceRole(str, Enum):
    """Interpretation boundary for values emitted by static 3D modules."""

    QUALITY = "quality"
    QA = "qa"
    DIAGNOSTIC = "diagnostic"
    ANOMALY_RESIDUAL = "anomaly_residual"


@dataclass(frozen=True)
class Static3DContext:
    """Read-only access to one shared frame used by every static branch."""

    frame: Shared3DFrameObservation

    def object_by_id(self, object_id: str) -> Object3DObservation:
        """Return one object without copying or modifying the observation."""

        matches = [
            obj for obj in self.frame.objects if obj.source_object_2d_id == object_id
        ]
        if len(matches) != 1:
            raise KeyError(
                f"Expected one object with source_object_2d_id={object_id!r}, "
                f"found {len(matches)}."
            )
        return matches[0]


def reprojection_cycle_evidence(
    error_px: float,
    *,
    source_ids: tuple[str, ...] = (),
    independent_observation: bool = False,
) -> ResidualEvidence:
    """Represent projection closure as QA, never as a static anomaly residual.

    The same points used for back-projection and projection are not independent
    evidence. ``independent_observation=True`` only reserves metadata for a
    future held-out mask/boundary/model observation; this function still does
    not promote the value to a trained anomaly score.
    """

    if not math.isfinite(float(error_px)) or error_px < 0.0:
        return ResidualEvidence.missing(
            "reconstruction_cycle_error",
            "invalid_reprojection_cycle_error",
            source_ids=source_ids,
            metadata={"independent_observation": independent_observation},
        )
    quality = 1.0 / (1.0 + float(error_px))
    return ResidualEvidence.observed(
        "reconstruction_cycle_error",
        float(error_px),
        quality=quality,
        source_ids=source_ids,
        metadata={
            "evidence_role": EvidenceRole.QA.value,
            "independent_observation": independent_observation,
            "anomaly_residual": False,
            "allowed_future_static_residual": bool(independent_observation),
            "interpretation": "same-point projection/back-projection cycle QA",
        },
    )
