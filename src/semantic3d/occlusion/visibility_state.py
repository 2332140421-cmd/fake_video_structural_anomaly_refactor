"""Object visibility state machine with explicit uncertainty semantics."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .mask_observation import InstanceMaskObservation, PredictedObjectSupport


class VisibilityState(str, Enum):
    FULLY_VISIBLE = "fully_visible"
    PARTIALLY_OCCLUDED = "partially_occluded"
    FULLY_OCCLUDED = "fully_occluded"
    OUT_OF_FRAME = "out_of_frame"
    DETECTOR_MISSING = "detector_missing"
    REAPPEARED = "reappeared"
    SCENE_CUT = "scene_cut"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True)
class ObjectVisibilityObservation:
    """One state transition backed by predicted and independently visible area."""

    object_track_id: str
    frame_index: int
    previous_state: VisibilityState | str
    current_state: VisibilityState | str
    predicted_support_area: float
    observed_visible_area: float
    visible_ratio: float
    occluded_ratio: float
    in_frame_ratio: float
    possible_occluder_ids: tuple[str, ...]
    state_quality: float
    valid: bool
    missing_reason: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        previous, current = VisibilityState(self.previous_state), VisibilityState(self.current_state)
        quality = float(self.state_quality)
        ratios = (float(self.visible_ratio), float(self.occluded_ratio), float(self.in_frame_ratio))
        if not math.isfinite(quality) or not 0.0 <= quality <= 1.0:
            raise ValueError("state_quality must be in [0, 1].")
        if self.valid and any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in ratios):
            raise ValueError("Valid visibility ratios must be in [0, 1].")
        if not self.valid and any(
            not (math.isnan(value) or (math.isfinite(value) and 0.0 <= value <= 1.0))
            for value in ratios
        ):
            raise ValueError("Invalid visibility ratios must be NaN or diagnostic values in [0, 1].")
        if self.valid and self.missing_reason:
            raise ValueError("Valid visibility state cannot have missing_reason.")
        if not self.valid and not self.missing_reason:
            raise ValueError("Invalid visibility state requires missing_reason.")
        object.__setattr__(self, "previous_state", previous)
        object.__setattr__(self, "current_state", current)
        object.__setattr__(self, "possible_occluder_ids", tuple(self.possible_occluder_ids))
        object.__setattr__(self, "state_quality", quality)
        object.__setattr__(self, "metadata", dict(self.metadata))


def infer_visibility_state(
    prediction: PredictedObjectSupport,
    observed: Optional[InstanceMaskObservation],
    *,
    previous_state: VisibilityState | str = VisibilityState.UNCERTAIN,
    nearer_object_masks: Optional[Mapping[str, np.ndarray]] = None,
    detector_confidence: float = 1.0,
    scene_cut: bool = False,
    detection_confirmed_absent: bool = False,
    appearance_without_history_is_event: bool = False,
    full_visibility_ratio: float = 0.80,
    full_occlusion_ratio: float = 0.80,
    out_of_frame_ratio: float = 0.20,
) -> ObjectVisibilityObservation:
    """Infer state without treating detection absence as occlusion by default."""

    previous = VisibilityState(previous_state)
    if scene_cut:
        return ObjectVisibilityObservation(
            prediction.object_track_id, prediction.target_frame_index, previous,
            VisibilityState.SCENE_CUT, float("nan"), float("nan"),
            float("nan"), float("nan"), float("nan"), (), 0.0,
            False, "scene_cut_breaks_visibility_history",
            {"state_history_continued": False},
        )
    if prediction.geometry_mode.value == "unavailable":
        return ObjectVisibilityObservation(
            prediction.object_track_id, prediction.target_frame_index, previous,
            VisibilityState.UNCERTAIN, float("nan"), float("nan"),
            float("nan"), float("nan"), float("nan"), (), 0.0, False,
            prediction.missing_reason or "dynamic_geometry_unavailable",
            {"fully_occluded_asserted": False, "formal_occlusion_evidence": False},
        )
    if not prediction.valid or prediction.support_mask is None:
        if observed is not None and observed.valid and observed.visible_mask is not None:
            legacy = observed.is_legacy_bbox_fallback
            reappeared = previous in {
                VisibilityState.FULLY_OCCLUDED,
                VisibilityState.OUT_OF_FRAME,
            } or appearance_without_history_is_event
            return ObjectVisibilityObservation(
                observed.object_track_id, observed.frame_index, previous,
                VisibilityState.REAPPEARED if reappeared else VisibilityState.UNCERTAIN,
                0.0 if reappeared else float("nan"), float(observed.mask_area),
                1.0 if reappeared else float("nan"),
                0.0 if reappeared else float("nan"), 1.0, (),
                observed.confidence if reappeared else 0.0,
                bool(reappeared and not legacy),
                (
                    "legacy_bbox_mask_visibility_diagnostic_only"
                    if reappeared and legacy
                    else "insufficient_history_for_appearance_explanation"
                ) if not (reappeared and not legacy) else "",
                metadata={
                    "reappearance_without_history": bool(
                        appearance_without_history_is_event
                    ),
                    "detection_missing_auto_occlusion": False,
                    "legacy_bbox_fallback": legacy,
                },
            )
        return ObjectVisibilityObservation(
            prediction.object_track_id, prediction.target_frame_index, previous,
            VisibilityState.UNCERTAIN, float("nan"), float("nan"),
            float("nan"), float("nan"), float("nan"),
            (), 0.0, False, "missing_predicted_support",
            {"fully_occluded_asserted": False},
        )
    predicted_area = float(np.count_nonzero(prediction.support_mask))
    observed_mask = None if observed is None or not observed.valid else observed.visible_mask
    observed_area = float(np.count_nonzero(observed_mask)) if observed_mask is not None else 0.0
    visible_ratio = min(1.0, observed_area / max(prediction.predicted_area, 1.0))
    occluders, covered = [], np.zeros(prediction.image_shape, dtype=bool)
    for object_id, mask in (nearer_object_masks or {}).items():
        candidate = np.asarray(mask, dtype=bool)
        if candidate.shape != prediction.image_shape:
            raise ValueError("Occluder masks must match predicted support shape.")
        overlap = prediction.support_mask & candidate
        if np.any(overlap):
            occluders.append(str(object_id))
            covered |= overlap
    occluded_ratio = float(np.count_nonzero(covered) / max(predicted_area, 1.0))
    if observed_mask is not None:
        if previous == VisibilityState.FULLY_OCCLUDED:
            state = VisibilityState.REAPPEARED
        elif visible_ratio >= full_visibility_ratio:
            state = VisibilityState.FULLY_VISIBLE
        elif visible_ratio > 0.0:
            state = VisibilityState.PARTIALLY_OCCLUDED
        else:
            state = VisibilityState.UNCERTAIN
        valid = not observed.is_legacy_bbox_fallback
        reason = "" if valid else "legacy_bbox_mask_visibility_diagnostic_only"
        quality = min(prediction.quality, observed.confidence)
    elif prediction.in_frame_ratio <= out_of_frame_ratio:
        state, valid, reason, quality = VisibilityState.OUT_OF_FRAME, True, "", prediction.quality
    elif occluded_ratio >= full_occlusion_ratio and occluders:
        state, valid, reason, quality = VisibilityState.FULLY_OCCLUDED, True, "", prediction.quality
    elif detection_confirmed_absent and detector_confidence >= 0.5:
        state, valid, reason, quality = VisibilityState.UNCERTAIN, True, "", min(prediction.quality, detector_confidence)
    elif detector_confidence < 0.5:
        state, valid, reason, quality = VisibilityState.DETECTOR_MISSING, False, "low_detector_confidence", 0.0
    else:
        state, valid, reason, quality = VisibilityState.DETECTOR_MISSING, False, "missing_detection_is_not_occlusion", 0.0
    if state == VisibilityState.DETECTOR_MISSING:
        observed_area = float("nan")
        visible_ratio = float("nan")
    if valid and bool(prediction.metadata.get("legacy_bbox_fallback", False)):
        valid = False
        reason = "legacy_bbox_mask_visibility_diagnostic_only"
    return ObjectVisibilityObservation(
        object_track_id=prediction.object_track_id,
        frame_index=prediction.target_frame_index,
        previous_state=previous,
        current_state=state,
        predicted_support_area=prediction.predicted_area,
        observed_visible_area=observed_area,
        visible_ratio=visible_ratio,
        occluded_ratio=occluded_ratio,
        in_frame_ratio=prediction.in_frame_ratio,
        possible_occluder_ids=tuple(occluders),
        state_quality=quality,
        valid=valid,
        missing_reason=reason,
        metadata={
            "detection_missing_auto_occlusion": False,
            "legacy_bbox_fallback": False if observed is None else observed.is_legacy_bbox_fallback,
            "predicted_support_quality": prediction.quality,
        },
    )
