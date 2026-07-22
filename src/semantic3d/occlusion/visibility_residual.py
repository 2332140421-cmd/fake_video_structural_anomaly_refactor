"""Explain disappearance and appearance before emitting anomaly evidence."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from ..validity import ResidualEvidence
from .visibility_state import ObjectVisibilityObservation, VisibilityState


class VisibilityExplanation(str, Enum):
    EXPLAINED_OCCLUSION = "explained_occlusion"
    EXPLAINED_OUT_OF_FRAME = "explained_out_of_frame"
    UNEXPLAINED_DISAPPEARANCE = "unexplained_disappearance"
    UNEXPLAINED_APPEARANCE = "unexplained_appearance"
    DETECTOR_UNCERTAIN = "detector_uncertain"
    NO_EVENT = "no_visibility_event"


@dataclass(frozen=True)
class VisibilityExplanationResidual:
    """Diagnostic explanation plus formal evidence only for unexplained events."""

    object_track_id: str
    frame_index: int
    explanation: VisibilityExplanation | str
    diagnostic_evidence: ResidualEvidence
    residual_evidence: ResidualEvidence
    valid: bool
    missing_reason: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        explanation = VisibilityExplanation(self.explanation)
        if self.valid and not self.diagnostic_evidence.valid:
            raise ValueError("Valid visibility explanation requires diagnostic evidence.")
        object.__setattr__(self, "explanation", explanation)
        object.__setattr__(self, "metadata", dict(self.metadata))


def compute_visibility_explanation_residual(observation: ObjectVisibilityObservation) -> VisibilityExplanationResidual:
    """Separate explained visibility changes, detector uncertainty, and anomalies."""

    source_ids = (observation.object_track_id, str(observation.frame_index))
    state, previous = observation.current_state, observation.previous_state
    legacy_diagnostic = bool(observation.metadata.get("legacy_bbox_fallback", False))
    if not observation.valid and not legacy_diagnostic:
        if state == VisibilityState.DETECTOR_MISSING:
            explanation = VisibilityExplanation.DETECTOR_UNCERTAIN
        else:
            explanation = VisibilityExplanation.NO_EVENT
        reason = observation.missing_reason or "visibility_observation_missing"
        diagnostic = ResidualEvidence.missing(
            "visibility_explanation_diagnostic", reason, source_ids=source_ids,
            metadata={"state": state.value},
        )
        residual = ResidualEvidence.missing(
            "r_visibility_explanation", reason, source_ids=source_ids,
            metadata={"no_event_distinct_from_missing": True},
        )
        return VisibilityExplanationResidual(
            observation.object_track_id, observation.frame_index, explanation,
            diagnostic, residual, False, reason,
            {"detection_miss_is_anomaly": False},
        )
    if state == VisibilityState.FULLY_OCCLUDED and observation.possible_occluder_ids:
        explanation, diagnostic_value = VisibilityExplanation.EXPLAINED_OCCLUSION, 0.0
    elif state == VisibilityState.OUT_OF_FRAME:
        explanation, diagnostic_value = VisibilityExplanation.EXPLAINED_OUT_OF_FRAME, 0.0
    elif state == VisibilityState.DETECTOR_MISSING:
        explanation, diagnostic_value = VisibilityExplanation.DETECTOR_UNCERTAIN, 0.0
    elif state == VisibilityState.REAPPEARED and previous not in {VisibilityState.FULLY_OCCLUDED, VisibilityState.OUT_OF_FRAME}:
        explanation, diagnostic_value = VisibilityExplanation.UNEXPLAINED_APPEARANCE, 1.0
    elif state == VisibilityState.UNCERTAIN and observation.in_frame_ratio > 0.5 and not observation.possible_occluder_ids:
        explanation, diagnostic_value = VisibilityExplanation.UNEXPLAINED_DISAPPEARANCE, 1.0
    else:
        explanation, diagnostic_value = VisibilityExplanation.NO_EVENT, 0.0
    diagnostic = ResidualEvidence.observed("visibility_explanation_diagnostic", diagnostic_value, quality=observation.state_quality, source_ids=source_ids, metadata={"explanation": explanation.value})
    if explanation in {VisibilityExplanation.UNEXPLAINED_DISAPPEARANCE, VisibilityExplanation.UNEXPLAINED_APPEARANCE} and observation.valid:
        residual = ResidualEvidence.observed("r_visibility_explanation", diagnostic_value, quality=observation.state_quality, source_ids=source_ids, metadata={"explanation": explanation.value})
    else:
        reason = "explained_visibility_change" if explanation in {VisibilityExplanation.EXPLAINED_OCCLUSION, VisibilityExplanation.EXPLAINED_OUT_OF_FRAME, VisibilityExplanation.NO_EVENT} else observation.missing_reason or "detector_uncertain"
        residual = ResidualEvidence.missing("r_visibility_explanation", reason, source_ids=source_ids, metadata={"diagnostic_value": diagnostic_value})
    return VisibilityExplanationResidual(observation.object_track_id, observation.frame_index, explanation, diagnostic, residual, True, metadata={"detection_miss_is_anomaly": False})
