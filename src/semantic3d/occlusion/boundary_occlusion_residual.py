"""Occlusion contact-boundary motion diagnostics and formal evidence."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import cv2
import numpy as np

from ..validity import ResidualEvidence
from .mask_observation import InstanceMaskObservation, PredictedObjectSupport
from .occlusion_graph import OcclusionRelation


@dataclass(frozen=True)
class BoundaryOcclusionResidual:
    """Predicted versus independently observed object-contact boundary."""

    foreground_object_id: str
    background_object_id: str
    frame_index: int
    predicted_contact_boundary: Optional[np.ndarray]
    observed_contact_boundary: Optional[np.ndarray]
    boundary_distance: float
    boundary_motion_consistency: float
    diagnostic_evidence: ResidualEvidence
    residual_evidence: ResidualEvidence
    valid: bool
    missing_reason: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.valid:
            if self.predicted_contact_boundary is None or self.observed_contact_boundary is None or not math.isfinite(float(self.boundary_distance)):
                raise ValueError("Valid boundary result requires predicted/observed boundaries.")
            if not self.diagnostic_evidence.valid:
                raise ValueError("Valid boundary result requires diagnostic evidence.")
        elif not self.missing_reason:
            raise ValueError("Invalid boundary result requires missing_reason.")
        object.__setattr__(self, "metadata", dict(self.metadata))


def _contact(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    second_boundary = second ^ cv2.erode(second.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    return second_boundary & cv2.dilate(first.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)


def _distance(first: np.ndarray, second: np.ndarray) -> float:
    if not np.any(first) or not np.any(second):
        return float("nan")
    to_second = cv2.distanceTransform((~second).astype(np.uint8), cv2.DIST_L2, 3)
    to_first = cv2.distanceTransform((~first).astype(np.uint8), cv2.DIST_L2, 3)
    return float(0.5 * (np.mean(to_second[first]) + np.mean(to_first[second])))


def compute_boundary_occlusion_residual(
    relation: OcclusionRelation,
    *,
    predicted_foreground: PredictedObjectSupport,
    predicted_background: PredictedObjectSupport,
    observed_foreground: Optional[InstanceMaskObservation],
    observed_background: Optional[InstanceMaskObservation],
) -> BoundaryOcclusionResidual:
    """Compare contact boundaries; bbox-only sources remain diagnostic-only."""

    ids = (relation.foreground_object_id, relation.background_object_id, str(relation.frame_index))
    reason = ""
    if not relation.valid:
        reason = relation.missing_reason or "invalid_occlusion_relation"
    elif not predicted_foreground.valid or not predicted_background.valid:
        reason = "missing_predicted_contact_support"
    elif observed_foreground is None or observed_background is None or not observed_foreground.valid or not observed_background.valid:
        reason = "missing_observed_contact_masks"
    elif observed_foreground.is_legacy_bbox_fallback or observed_background.is_legacy_bbox_fallback:
        reason = "bbox_boundary_diagnostic_only"
    if reason:
        missing = ResidualEvidence.missing("r_boundary_occlusion", reason, source_ids=ids)
        return BoundaryOcclusionResidual(relation.foreground_object_id, relation.background_object_id, relation.frame_index, None, None, float("nan"), float("nan"), missing, missing, False, reason)
    assert predicted_foreground.support_mask is not None and predicted_background.support_mask is not None
    assert observed_foreground is not None and observed_foreground.visible_mask is not None
    assert observed_background is not None and observed_background.visible_mask is not None
    predicted_contact = _contact(predicted_background.support_mask, predicted_foreground.support_mask)
    observed_contact = _contact(observed_background.visible_mask, observed_foreground.visible_mask)
    distance = _distance(predicted_contact, observed_contact)
    if not math.isfinite(distance):
        missing = ResidualEvidence.missing("r_boundary_occlusion", "contact_boundary_unavailable", source_ids=ids)
        return BoundaryOcclusionResidual(relation.foreground_object_id, relation.background_object_id, relation.frame_index, None, None, float("nan"), float("nan"), missing, missing, False, "contact_boundary_unavailable")
    diagonal = math.hypot(*predicted_foreground.image_shape)
    normalized = distance / max(diagonal, 1e-8)
    quality = min(relation.occlusion_confidence, predicted_foreground.quality, predicted_background.quality, observed_foreground.confidence, observed_background.confidence)
    diagnostic = ResidualEvidence.observed("boundary_occlusion_diagnostic", normalized, quality=quality, source_ids=ids, metadata={"boundary_distance_px": distance})
    residual = ResidualEvidence.observed("r_boundary_occlusion", normalized, quality=quality, source_ids=ids, metadata={"current_boundary_independent": True, "prediction_uses_current_boundary": False})
    return BoundaryOcclusionResidual(relation.foreground_object_id, relation.background_object_id, relation.frame_index, predicted_contact, observed_contact, distance, max(0.0, 1.0 - normalized), diagnostic, residual, True, metadata={"current_observed_boundary_used_for_prediction": False})
