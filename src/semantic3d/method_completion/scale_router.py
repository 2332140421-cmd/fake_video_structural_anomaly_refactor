"""Priority router for metric, temporal, and strict-v2 pair scale evidence."""

from __future__ import annotations

import math
from dataclasses import replace
from enum import Enum
from typing import Callable, Optional, Sequence

from .scale_evidence import (
    ProviderStatus,
    ScaleBranchName,
    ScaleEvidenceRole,
    ScaleGeometryEvidence,
    ScaleRouteDecision,
)


class PairBranchPolicy(str, Enum):
    """When same-frame object pairs may be enumerated."""

    DISABLED = "disabled"
    FALLBACK_ONLY = "fallback_only"
    AUDIT_ONLY = "audit_only"
    FALLBACK_AND_AUDIT = "fallback_and_audit"
    ALWAYS = "always"


PairEvidenceSupplier = Callable[[], Sequence[ScaleGeometryEvidence]]


class ScaleEvidenceRouter:
    """Select scale evidence without letting fallback overwrite metric evidence."""

    def __init__(
        self, pair_branch_policy: PairBranchPolicy | str = PairBranchPolicy.FALLBACK_ONLY
    ) -> None:
        self.pair_branch_policy = PairBranchPolicy(pair_branch_policy)

    def route(
        self,
        *,
        metric_evidence: ScaleGeometryEvidence,
        temporal_evidence: Optional[ScaleGeometryEvidence] = None,
        pair_supplier: Optional[PairEvidenceSupplier] = None,
        clip_observability: str = "unknown",
    ) -> ScaleRouteDecision:
        """Route one object/track through the fixed MD2 branch priority."""

        if metric_evidence.branch_name != ScaleBranchName.METRIC_SINGLE_OBJECT:
            raise ValueError("metric_evidence must come from metric_single_object_scale.")
        evidences: list[ScaleGeometryEvidence] = [metric_evidence]
        executed = [ScaleBranchName.METRIC_SINGLE_OBJECT.value]
        available: list[str] = []
        skipped: dict[str, str] = {}
        metric_valid = metric_evidence.valid
        temporal_valid = bool(temporal_evidence is not None and temporal_evidence.valid)
        if metric_valid:
            available.append(ScaleBranchName.METRIC_SINGLE_OBJECT.value)
        if temporal_evidence is not None:
            if temporal_evidence.branch_name != ScaleBranchName.TEMPORAL_SAME_OBJECT:
                raise ValueError("temporal_evidence must come from temporal_same_object_scale.")
            evidences.append(temporal_evidence)
            executed.append(ScaleBranchName.TEMPORAL_SAME_OBJECT.value)
            if temporal_valid:
                available.append(ScaleBranchName.TEMPORAL_SAME_OBJECT.value)
        else:
            skipped[ScaleBranchName.TEMPORAL_SAME_OBJECT.value] = "temporal_evidence_not_supplied"

        fallback_condition = not metric_valid and not temporal_valid
        run_pair = False
        pair_role = ScaleEvidenceRole.FALLBACK
        policy = self.pair_branch_policy
        if policy == PairBranchPolicy.DISABLED:
            skipped[ScaleBranchName.RELATIVE_PAIR.value] = "pair_policy_disabled"
        elif policy == PairBranchPolicy.FALLBACK_ONLY:
            run_pair = fallback_condition
            if not run_pair:
                skipped[ScaleBranchName.RELATIVE_PAIR.value] = "higher_priority_scale_evidence_available"
        elif policy == PairBranchPolicy.AUDIT_ONLY:
            run_pair = True
            pair_role = ScaleEvidenceRole.AUDIT_CROSSCHECK
        elif policy == PairBranchPolicy.FALLBACK_AND_AUDIT:
            run_pair = True
            pair_role = (
                ScaleEvidenceRole.FALLBACK
                if fallback_condition
                else ScaleEvidenceRole.AUDIT_CROSSCHECK
            )
        elif policy == PairBranchPolicy.ALWAYS:
            run_pair = True
            pair_role = (
                ScaleEvidenceRole.FALLBACK
                if fallback_condition
                else ScaleEvidenceRole.AUDIT_CROSSCHECK
            )

        pair_items: list[ScaleGeometryEvidence] = []
        if run_pair:
            if pair_supplier is None:
                skipped[ScaleBranchName.RELATIVE_PAIR.value] = "pair_supplier_unavailable"
            else:
                supplied = tuple(pair_supplier())
                executed.append(ScaleBranchName.RELATIVE_PAIR.value)
                for item in supplied:
                    if item.branch_name != ScaleBranchName.RELATIVE_PAIR:
                        raise ValueError("pair_supplier returned non-pair evidence.")
                    pair_items.append(replace(item, evidence_role=pair_role))
                evidences.extend(pair_items)
                if any(item.valid for item in pair_items):
                    available.append(ScaleBranchName.RELATIVE_PAIR.value)
                elif not pair_items:
                    skipped[ScaleBranchName.RELATIVE_PAIR.value] = "no_pair_candidates"

        pair_valid = next((item for item in pair_items if item.valid), None)
        if metric_valid:
            selected = ScaleBranchName.METRIC_SINGLE_OBJECT.value
            selected_evidence = metric_evidence
            reason = "metric_single_object_scale_valid"
        elif temporal_valid and temporal_evidence is not None:
            selected = ScaleBranchName.TEMPORAL_SAME_OBJECT.value
            selected_evidence = temporal_evidence
            reason = "metric_unavailable_temporal_scale_valid"
        elif pair_valid is not None and pair_role == ScaleEvidenceRole.FALLBACK:
            selected = ScaleBranchName.RELATIVE_PAIR.value
            selected_evidence = pair_valid
            reason = "higher_priority_scale_evidence_unavailable_pair_fallback_valid"
        else:
            selected = ScaleBranchName.NONE.value
            selected_evidence = None
            failures = [item.failure_reason for item in evidences if not item.valid]
            reason = "no_scale_evidence:" + "|".join(dict.fromkeys(filter(None, failures)))
        confidence = 0.0 if selected_evidence is None else selected_evidence.confidence
        return ScaleRouteDecision(
            selected_primary_branch=selected,
            available_branches=tuple(dict.fromkeys(available)),
            executed_branches=tuple(dict.fromkeys(executed)),
            skipped_branches=skipped,
            fallback_used=selected == ScaleBranchName.RELATIVE_PAIR.value,
            audit_crosscheck_used=any(
                item.evidence_role == ScaleEvidenceRole.AUDIT_CROSSCHECK for item in pair_items
            ),
            routing_reason=reason,
            evidence_confidence=confidence,
            evidences=tuple(evidences),
        )


def strict_v2_row_to_scale_evidence(
    row: dict[str, object],
    *,
    video_id: str,
    clip_id: str,
    frame_id: str,
    config_sha256: str = "",
    software_commit: str = "",
) -> ScaleGeometryEvidence:
    """Adapt an unchanged strict-v2 result row to the unified router schema."""

    object_a = str(row.get("object_a_id", ""))
    object_b = str(row.get("object_b_id", ""))
    base = {
        "video_id": video_id,
        "clip_id": clip_id,
        "frame_id": frame_id,
        "object_id": f"{object_a}|{object_b}",
        "track_id": "",
        "branch_name": ScaleBranchName.RELATIVE_PAIR,
        "branch_priority": 3,
        "evidence_role": ScaleEvidenceRole.FALLBACK,
        "residual_name": "R_sd_pair",
        "depth_type": "relative",
        "depth_unit": "relative_local_unit",
        "depth_definition": "larger_is_farther",
        "coordinate_system": "same_frame_dimension_aligned_2p5d",
        "localization_reference": f"object_pair:{object_a}|{object_b}",
        "provenance": {
            "strict_v2_row_reused": True,
            "strict_v2_formula_recomputed": False,
            "prior_source": row.get("prior_source", "physical"),
            "object_a_id": object_a,
            "object_b_id": object_b,
        },
        "config_sha256": config_sha256,
        "software_commit": software_commit,
    }
    value = float(row.get("rsd_log", float("nan")))
    if bool(row.get("valid", False)) and math.isfinite(value):
        return ScaleGeometryEvidence.observed(
            residual_value=value,
            confidence=min(
                float(row.get("gate_score_a", 1.0)),
                float(row.get("gate_score_b", 1.0)),
            ),
            uncertainty=0.0,
            **base,
        )
    return ScaleGeometryEvidence.missing(
        failure_reason=str(row.get("skip_reason", "invalid_strict_v2_pair")),
        **base,
    )
