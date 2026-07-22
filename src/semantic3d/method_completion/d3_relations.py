"""D3 higher-order 3D relation contracts and formula-level diagnostics.

This module does not claim an integrated D3 detector.  It defines the
pose-compensated relation measurements and comparison formulas needed by a
future full-SE3 executor.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

import numpy as np

from ..validity import ResidualEvidence


class D3RelationType(str, Enum):
    """Higher-order relationships available after full pose compensation."""

    OBJECT_RELATIVE_DISTANCE = "object_relative_distance"
    DEPTH_ORDER = "depth_order"
    STRUCTURE_EDGE_LENGTH = "structure_edge_length"
    LOCAL_RIGIDITY = "local_rigidity"
    BEARING_RELATION = "bearing_relation"


@dataclass(frozen=True)
class D3RelationObservation:
    """One pose-compensated relation measurement at one frame."""

    relation_id: str
    relation_type: D3RelationType | str
    frame_index: int
    source_ids: tuple[str, ...]
    values: tuple[float, ...]
    unit: str
    coordinate_system_id: str
    pose_compensated: bool
    valid: bool
    quality: float
    missing_reason: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        relation_type = D3RelationType(self.relation_type)
        values = tuple(float(value) for value in self.values)
        quality = float(self.quality)
        if not math.isfinite(quality) or not 0.0 <= quality <= 1.0:
            raise ValueError("D3 relation quality must be in [0, 1].")
        if self.valid:
            if not values or not all(math.isfinite(value) for value in values):
                raise ValueError("Valid D3 relation requires finite values.")
            if not self.pose_compensated:
                raise ValueError("Valid D3 relation must be pose compensated.")
            if not self.unit or not self.coordinate_system_id or self.missing_reason:
                raise ValueError("Valid D3 relation requires unit and coordinate system.")
        elif not self.missing_reason:
            raise ValueError("Invalid D3 relation requires missing_reason.")
        object.__setattr__(self, "relation_type", relation_type)
        object.__setattr__(self, "source_ids", tuple(self.source_ids))
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "quality", quality)
        object.__setattr__(self, "metadata", dict(self.metadata))


class D3HigherOrderResidual(ABC):
    """Interface for a future integrated full-SE3 D3 branch."""

    @abstractmethod
    def evaluate(
        self,
        previous: Sequence[D3RelationObservation],
        current: Sequence[D3RelationObservation],
    ) -> tuple[ResidualEvidence, ...]:
        """Compare matched full-SE3 relations without fitting a threshold."""


def d3_formula_definitions() -> dict[str, str]:
    """Return frozen formula interfaces used by future D3 implementations."""

    return {
        D3RelationType.OBJECT_RELATIVE_DISTANCE.value: (
            "abs(log((d_t + eps) / (d_prev + eps)))"
        ),
        D3RelationType.DEPTH_ORDER.value: (
            "0 if sign(z_i-z_j)_t == sign(z_i-z_j)_prev else 1"
        ),
        D3RelationType.STRUCTURE_EDGE_LENGTH.value: (
            "abs(log((edge_t + eps) / (edge_prev + eps)))"
        ),
        D3RelationType.LOCAL_RIGIDITY.value: (
            "abs(rigidity_stat_t - rigidity_stat_prev)"
        ),
        D3RelationType.BEARING_RELATION.value: (
            "acos(clip(dot(unit_bearing_t, unit_bearing_prev), -1, 1)) / pi"
        ),
    }


def compare_d3_relations(
    previous: D3RelationObservation,
    current: D3RelationObservation,
    *,
    eps: float = 1e-8,
) -> ResidualEvidence:
    """Compare two already pose-compensated relation measurements.

    This is a formula-level interface.  Building trustworthy D3 relations from
    full-SE3 video observations remains outside P4-C3A-M.
    """

    source_ids = (*previous.source_ids, *current.source_ids)
    if not previous.valid or not current.valid:
        return ResidualEvidence.missing(
            "d3_relation_change",
            previous.missing_reason or current.missing_reason or "invalid_d3_relation",
            source_ids=source_ids,
        )
    if (
        previous.relation_id != current.relation_id
        or previous.relation_type != current.relation_type
    ):
        return ResidualEvidence.missing(
            "d3_relation_change", "d3_relation_mismatch", source_ids=source_ids
        )
    if (
        previous.coordinate_system_id != current.coordinate_system_id
        or previous.unit != current.unit
    ):
        return ResidualEvidence.missing(
            "d3_relation_change", "d3_coordinate_or_unit_mismatch", source_ids=source_ids
        )
    previous_values = np.asarray(previous.values, dtype=float)
    current_values = np.asarray(current.values, dtype=float)
    if previous_values.shape != current_values.shape:
        return ResidualEvidence.missing(
            "d3_relation_change", "d3_relation_shape_mismatch", source_ids=source_ids
        )
    relation_type = previous.relation_type
    if relation_type in {
        D3RelationType.OBJECT_RELATIVE_DISTANCE,
        D3RelationType.STRUCTURE_EDGE_LENGTH,
    }:
        if np.any(previous_values <= 0.0) or np.any(current_values <= 0.0):
            return ResidualEvidence.missing(
                "d3_relation_change", "non_positive_d3_length", source_ids=source_ids
            )
        value = float(
            np.mean(np.abs(np.log((current_values + eps) / (previous_values + eps))))
        )
    elif relation_type == D3RelationType.DEPTH_ORDER:
        value = float(np.mean(np.sign(previous_values) != np.sign(current_values)))
    elif relation_type == D3RelationType.LOCAL_RIGIDITY:
        value = float(np.mean(np.abs(current_values - previous_values)))
    else:
        previous_norm = float(np.linalg.norm(previous_values))
        current_norm = float(np.linalg.norm(current_values))
        if previous_norm <= eps or current_norm <= eps:
            return ResidualEvidence.missing(
                "d3_relation_change", "invalid_bearing_vector", source_ids=source_ids
            )
        cosine = float(
            np.clip(
                np.dot(previous_values / previous_norm, current_values / current_norm),
                -1.0,
                1.0,
            )
        )
        value = math.acos(cosine) / math.pi
    return ResidualEvidence.observed(
        "d3_relation_change",
        value,
        quality=min(previous.quality, current.quality),
        source_ids=source_ids,
        metadata={
            "relation_id": previous.relation_id,
            "relation_type": relation_type.value,
            "formula": d3_formula_definitions()[relation_type.value],
            "full_d3_executor_available": False,
            "code_status": "interface_only",
        },
    )
