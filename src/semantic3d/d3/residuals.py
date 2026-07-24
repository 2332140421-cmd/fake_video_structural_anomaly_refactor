"""Concrete M5 executor for pose-compensated D3 graph relations."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from ..method_completion.d3_relations import (
    D3HigherOrderResidual,
    D3RelationObservation,
    D3RelationType,
    d3_formula_definitions,
)
from ..validity import ResidualEvidence
from .contracts import (
    D3FrameGraph,
    D3GraphRelation,
    D3RelationResidual,
    D3TransitionContext,
)


_RESIDUAL_NAMES = {
    D3RelationType.OBJECT_RELATIVE_DISTANCE: "R_relative_distance",
    D3RelationType.DEPTH_ORDER: "R_depth_order",
    D3RelationType.STRUCTURE_EDGE_LENGTH: "R_edge_length",
    D3RelationType.LOCAL_RIGIDITY: "R_local_rigidity",
    D3RelationType.BEARING_RELATION: "R_bearing_relation",
    D3RelationType.RELATIVE_ORIENTATION: "R_relative_orientation",
    D3RelationType.OBJECT_BOUNDARY_RELATION: "R_object_boundary_relation",
    D3RelationType.CONTAINMENT_OR_OVERLAP: "R_containment_or_overlap",
    D3RelationType.SUPPORT_OR_CONTACT: "R_support_or_contact",
}


def _missing(
    *,
    context: D3TransitionContext,
    relation_type: D3RelationType,
    relation_id: str,
    reason: str,
    previous: D3GraphRelation | None = None,
    current: D3GraphRelation | None = None,
) -> D3RelationResidual:
    source_nodes = tuple(
        dict.fromkeys(
            (
                *(previous.source_node_ids if previous is not None else ()),
                *(current.source_node_ids if current is not None else ()),
            )
        )
    )
    source_edge = (
        current.source_edge_id
        if current is not None
        else previous.source_edge_id
        if previous is not None
        else ""
    )
    localization = (
        current.localization_reference
        if current is not None
        else previous.localization_reference
        if previous is not None
        else {}
    )
    return D3RelationResidual(
        residual_id=f"{context.clip_id}:{context.frame_t}:{context.frame_t1}:{relation_id}",
        residual_name=_RESIDUAL_NAMES[relation_type],
        relation_type=relation_type,
        video_id=context.video_id,
        clip_id=context.clip_id,
        frame_t=context.frame_t,
        frame_t1=context.frame_t1,
        source_nodes=source_nodes,
        source_edge=source_edge,
        coordinate_frame="clip_local_aligned",
        reference_relation=(
            tuple(float("nan") for _ in previous.values)
            if previous is not None
            else (float("nan"),)
        ),
        observed_relation=(
            tuple(float("nan") for _ in current.values)
            if current is not None
            else (float("nan"),)
        ),
        value=float("nan"),
        confidence=0.0,
        valid=False,
        failure_reason=reason,
        localization_reference=localization,
        metadata={
            "pose_status": context.pose_status.value,
            "provider_failure_is_anomaly": False,
            "missing_is_zero": False,
        },
    )


def _value(
    relation_type: D3RelationType,
    previous: np.ndarray,
    current: np.ndarray,
    *,
    eps: float,
) -> float | None:
    if previous.shape != current.shape:
        return None
    if relation_type in {
        D3RelationType.OBJECT_RELATIVE_DISTANCE,
        D3RelationType.STRUCTURE_EDGE_LENGTH,
        D3RelationType.OBJECT_BOUNDARY_RELATION,
    }:
        if np.any(previous <= 0.0) or np.any(current <= 0.0):
            return None
        return float(np.mean(np.abs(np.log((current + eps) / (previous + eps)))))
    if relation_type == D3RelationType.DEPTH_ORDER:
        if np.any(np.abs(previous) <= eps) or np.any(np.abs(current) <= eps):
            return None
        return float(np.mean(np.sign(previous) != np.sign(current)))
    if relation_type == D3RelationType.LOCAL_RIGIDITY:
        return float(np.mean(np.abs(current - previous)))
    if relation_type == D3RelationType.RELATIVE_ORIENTATION:
        return float(np.mean(np.abs(current - previous)))
    if relation_type == D3RelationType.BEARING_RELATION:
        previous_norm = float(np.linalg.norm(previous))
        current_norm = float(np.linalg.norm(current))
        if previous_norm <= eps or current_norm <= eps:
            return None
        cosine = float(
            np.clip(
                np.dot(previous / previous_norm, current / current_norm),
                -1.0,
                1.0,
            )
        )
        return math.acos(cosine) / math.pi
    return float(np.mean(np.abs(current - previous)))


class D3StructureResidualExecutor(D3HigherOrderResidual):
    """Evaluate matched D3 relations after pose and identity eligibility gates."""

    def compare_graphs(
        self,
        previous_graph: D3FrameGraph,
        current_graph: D3FrameGraph,
        context: D3TransitionContext,
        *,
        eps: float = 1e-8,
    ) -> tuple[D3RelationResidual, ...]:
        """Return one auditable result for every unioned relation identity."""

        previous_by_id = {
            relation.relation_id: relation for relation in previous_graph.relations
        }
        current_by_id = {
            relation.relation_id: relation for relation in current_graph.relations
        }
        relation_ids = sorted(set(previous_by_id) | set(current_by_id))
        output: list[D3RelationResidual] = []
        for relation_id in relation_ids:
            previous = previous_by_id.get(relation_id)
            current = current_by_id.get(relation_id)
            relation_type = (
                previous.relation_type if previous is not None else current.relation_type
            )
            assert relation_type is not None
            if not context.valid:
                output.append(
                    _missing(
                        context=context,
                        relation_type=relation_type,
                        relation_id=relation_id,
                        reason="blocked_by_pose_or_correspondence",
                        previous=previous,
                        current=current,
                    )
                )
                continue
            if previous is None or current is None:
                output.append(
                    _missing(
                        context=context,
                        relation_type=relation_type,
                        relation_id=relation_id,
                        reason="relation_not_observed_in_both_frames",
                        previous=previous,
                        current=current,
                    )
                )
                continue
            if previous.relation_type != current.relation_type:
                output.append(
                    _missing(
                        context=context,
                        relation_type=relation_type,
                        relation_id=relation_id,
                        reason="d3_relation_type_mismatch",
                        previous=previous,
                        current=current,
                    )
                )
                continue
            if not previous.valid or not current.valid:
                output.append(
                    _missing(
                        context=context,
                        relation_type=relation_type,
                        relation_id=relation_id,
                        reason=previous.failure_reason
                        or current.failure_reason
                        or "invalid_d3_relation",
                        previous=previous,
                        current=current,
                    )
                )
                continue
            if not previous.identity_reliable or not current.identity_reliable:
                output.append(
                    _missing(
                        context=context,
                        relation_type=relation_type,
                        relation_id=relation_id,
                        reason="blocked_by_pose_or_correspondence",
                        previous=previous,
                        current=current,
                    )
                )
                continue
            if previous.unit != current.unit:
                output.append(
                    _missing(
                        context=context,
                        relation_type=relation_type,
                        relation_id=relation_id,
                        reason="d3_relation_unit_mismatch",
                        previous=previous,
                        current=current,
                    )
                )
                continue
            previous_values = np.asarray(previous.values, dtype=float)
            current_values = np.asarray(current.values, dtype=float)
            value = _value(
                relation_type, previous_values, current_values, eps=eps
            )
            if value is None or not math.isfinite(value):
                output.append(
                    _missing(
                        context=context,
                        relation_type=relation_type,
                        relation_id=relation_id,
                        reason="d3_relation_values_not_comparable",
                        previous=previous,
                        current=current,
                    )
                )
                continue
            confidence = min(
                context.pose_confidence,
                previous.confidence,
                current.confidence,
            )
            output.append(
                D3RelationResidual(
                    residual_id=(
                        f"{context.clip_id}:{context.frame_t}:"
                        f"{context.frame_t1}:{relation_id}"
                    ),
                    residual_name=_RESIDUAL_NAMES[relation_type],
                    relation_type=relation_type,
                    video_id=context.video_id,
                    clip_id=context.clip_id,
                    frame_t=context.frame_t,
                    frame_t1=context.frame_t1,
                    source_nodes=tuple(
                        dict.fromkeys(
                            (*previous.source_node_ids, *current.source_node_ids)
                        )
                    ),
                    source_edge=current.source_edge_id,
                    coordinate_frame="clip_local_aligned",
                    reference_relation=previous.values,
                    observed_relation=current.values,
                    value=value,
                    confidence=confidence,
                    valid=True,
                    failure_reason="",
                    localization_reference=current.localization_reference,
                    metadata={
                        "formula": d3_formula_definitions()[relation_type.value],
                        "pose_status": context.pose_status.value,
                        "pose_confidence": context.pose_confidence,
                        "authenticity_threshold_applied": False,
                        "residual_is_authenticity_decision": False,
                    },
                )
            )
        return tuple(output)

    def evaluate(
        self,
        previous: Sequence[D3RelationObservation],
        current: Sequence[D3RelationObservation],
    ) -> tuple[ResidualEvidence, ...]:
        """Retain the P4-C3A-M abstract interface for formula-level callers."""

        current_by_id = {item.relation_id: item for item in current}
        output: list[ResidualEvidence] = []
        for reference in previous:
            observed = current_by_id.get(reference.relation_id)
            source_ids = reference.source_ids + (
                observed.source_ids if observed is not None else ()
            )
            if observed is None:
                output.append(
                    ResidualEvidence.missing(
                        "d3_relation_change",
                        "relation_not_observed_in_both_frames",
                        source_ids=source_ids,
                    )
                )
                continue
            value = _value(
                reference.relation_type,
                np.asarray(reference.values, dtype=float),
                np.asarray(observed.values, dtype=float),
                eps=1e-8,
            )
            if (
                not reference.valid
                or not observed.valid
                or value is None
                or not math.isfinite(value)
            ):
                output.append(
                    ResidualEvidence.missing(
                        "d3_relation_change",
                        reference.missing_reason
                        or observed.missing_reason
                        or "d3_relation_values_not_comparable",
                        source_ids=source_ids,
                    )
                )
            else:
                output.append(
                    ResidualEvidence.observed(
                        "d3_relation_change",
                        value,
                        quality=min(reference.quality, observed.quality),
                        source_ids=source_ids,
                        metadata={
                            "concrete_m5_executor": True,
                            "authenticity_threshold_applied": False,
                        },
                    )
                )
        return tuple(output)
