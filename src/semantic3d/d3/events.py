"""Conservative M5 occlusion/disappearance classification and reappearance QA."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

import numpy as np


class OcclusionEventType(str, Enum):
    """Mutually explicit visibility/identity event classes."""

    PARTIAL_OCCLUSION = "partial_occlusion"
    FULL_OCCLUSION = "full_occlusion"
    OUT_OF_FRAME = "out_of_frame"
    DETECTOR_MISS = "detector_miss"
    TRACK_FAILURE = "track_failure"
    TRUE_DISAPPEARANCE = "true_disappearance"
    REAPPEARANCE = "reappearance"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class OcclusionEventInputsV2:
    """Observation predicates used without authenticity labels or residual thresholds."""

    video_id: str
    clip_id: str
    frame_index: int
    object_track_id: str
    previous_event_type: OcclusionEventType | str
    formal_visible_mask_available: bool
    history_prediction_available: bool
    observed_object_available: bool
    candidate_object_available: bool
    identity_consistent: bool
    mask_overlap_ratio: float
    depth_order_supported: bool
    visible_ratio: float
    predicted_in_frame_ratio: float
    trajectory_prediction_quality: float
    d2_reprojection_supported: bool
    detector_attempted: bool
    detector_reliable: bool
    detection_confirmed_absent: bool
    tracker_failed: bool
    persistent_absence_frames: int
    possible_occluder_ids: tuple[str, ...] = ()
    scene_cut: bool = False
    provider_failed: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        previous = OcclusionEventType(self.previous_event_type)
        for name in (
            "mask_overlap_ratio",
            "visible_ratio",
            "predicted_in_frame_ratio",
            "trajectory_prediction_quality",
        ):
            value = float(getattr(self, name))
            if not (math.isnan(value) or (math.isfinite(value) and 0.0 <= value <= 1.0)):
                raise ValueError(f"{name} must be NaN or in [0, 1].")
            object.__setattr__(self, name, value)
        if self.persistent_absence_frames < 0:
            raise ValueError("persistent_absence_frames cannot be negative.")
        object.__setattr__(self, "previous_event_type", previous)
        object.__setattr__(
            self, "possible_occluder_ids", tuple(self.possible_occluder_ids)
        )
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class OcclusionEventEvidence:
    """One event classification; no-event and missing observation stay invalid."""

    event_id: str
    video_id: str
    clip_id: str
    frame_index: int
    object_track_id: str
    event_type: OcclusionEventType | str
    status: str
    confidence: float
    valid: bool
    failure_reason: str
    possible_occluder_ids: tuple[str, ...]
    localization_reference: Mapping[str, Any]
    diagnostic_details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        event_type = OcclusionEventType(self.event_type)
        confidence = float(self.confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("Event confidence must be in [0, 1].")
        if self.valid:
            if (
                event_type == OcclusionEventType.UNKNOWN
                or self.status != "executed_valid"
                or self.failure_reason
            ):
                raise ValueError("Valid event requires a concrete executed_valid class.")
        elif not self.failure_reason or self.status == "executed_valid":
            raise ValueError("Invalid event requires non-valid status and reason.")
        object.__setattr__(self, "event_type", event_type)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(
            self, "possible_occluder_ids", tuple(self.possible_occluder_ids)
        )
        object.__setattr__(
            self, "localization_reference", dict(self.localization_reference)
        )
        object.__setattr__(self, "diagnostic_details", dict(self.diagnostic_details))


def _event(
    inputs: OcclusionEventInputsV2,
    event_type: OcclusionEventType,
    confidence: float,
) -> OcclusionEventEvidence:
    return OcclusionEventEvidence(
        event_id=(
            f"{inputs.clip_id}:{inputs.frame_index}:"
            f"{inputs.object_track_id}:{event_type.value}"
        ),
        video_id=inputs.video_id,
        clip_id=inputs.clip_id,
        frame_index=inputs.frame_index,
        object_track_id=inputs.object_track_id,
        event_type=event_type,
        status="executed_valid",
        confidence=float(np.clip(confidence, 0.0, 1.0)),
        valid=True,
        failure_reason="",
        possible_occluder_ids=inputs.possible_occluder_ids,
        localization_reference={
            "level": "object_track",
            "track_id": inputs.object_track_id,
            "frame_index": inputs.frame_index,
            "possible_occluder_ids": list(inputs.possible_occluder_ids),
        },
        diagnostic_details={
            "mask_overlap_ratio": inputs.mask_overlap_ratio,
            "visible_ratio": inputs.visible_ratio,
            "predicted_in_frame_ratio": inputs.predicted_in_frame_ratio,
            "depth_order_supported": inputs.depth_order_supported,
            "d2_reprojection_supported": inputs.d2_reprojection_supported,
            "authenticity_label_used": False,
            "event_is_authenticity_decision": False,
            "input_metadata": dict(inputs.metadata),
        },
    )


def _missing_event(
    inputs: OcclusionEventInputsV2,
    reason: str,
    *,
    status: str,
) -> OcclusionEventEvidence:
    return OcclusionEventEvidence(
        event_id=(
            f"{inputs.clip_id}:{inputs.frame_index}:"
            f"{inputs.object_track_id}:unknown"
        ),
        video_id=inputs.video_id,
        clip_id=inputs.clip_id,
        frame_index=inputs.frame_index,
        object_track_id=inputs.object_track_id,
        event_type=OcclusionEventType.UNKNOWN,
        status=status,
        confidence=0.0,
        valid=False,
        failure_reason=reason,
        possible_occluder_ids=inputs.possible_occluder_ids,
        localization_reference={
            "level": "object_track",
            "track_id": inputs.object_track_id,
            "frame_index": inputs.frame_index,
        },
        diagnostic_details={
            "provider_failure_is_anomaly": False,
            "missing_event_is_zero_residual": False,
            "input_metadata": dict(inputs.metadata),
        },
    )


def classify_occlusion_event(
    inputs: OcclusionEventInputsV2,
    *,
    partial_visible_ratio: float = 0.80,
    full_overlap_ratio: float = 0.75,
    out_of_frame_ratio: float = 0.20,
    minimum_track_quality: float = 0.50,
    true_disappearance_confirmation_frames: int = 2,
) -> OcclusionEventEvidence:
    """Classify one event only when its required observation predicates exist."""

    if inputs.provider_failed:
        return _missing_event(inputs, "event_provider_failed", status="provider_failed")
    if inputs.scene_cut:
        return _missing_event(
            inputs, "scene_cut_breaks_event_history", status="not_applicable"
        )
    if not inputs.history_prediction_available:
        return _missing_event(
            inputs, "history_prediction_unavailable", status="blocked_by_input"
        )
    quality = inputs.trajectory_prediction_quality
    if not math.isfinite(quality) or quality < minimum_track_quality:
        return _missing_event(
            inputs, "track_prediction_quality_insufficient", status="blocked_by_input"
        )

    if inputs.candidate_object_available:
        if not inputs.identity_consistent:
            return _event(inputs, OcclusionEventType.TRACK_FAILURE, quality)
        if inputs.previous_event_type in {
            OcclusionEventType.FULL_OCCLUSION,
            OcclusionEventType.OUT_OF_FRAME,
            OcclusionEventType.TRUE_DISAPPEARANCE,
        }:
            return _event(inputs, OcclusionEventType.REAPPEARANCE, quality)

    if (
        inputs.observed_object_available
        and inputs.formal_visible_mask_available
        and math.isfinite(inputs.visible_ratio)
        and 0.0 < inputs.visible_ratio < partial_visible_ratio
        and math.isfinite(inputs.mask_overlap_ratio)
        and inputs.mask_overlap_ratio > 0.0
        and inputs.depth_order_supported
        and inputs.d2_reprojection_supported
        and inputs.possible_occluder_ids
    ):
        return _event(
            inputs,
            OcclusionEventType.PARTIAL_OCCLUSION,
            min(quality, max(inputs.mask_overlap_ratio, 0.0)),
        )
    if inputs.observed_object_available:
        return _missing_event(
            inputs, "not_applicable_no_event", status="not_applicable"
        )

    if (
        math.isfinite(inputs.predicted_in_frame_ratio)
        and inputs.predicted_in_frame_ratio <= out_of_frame_ratio
    ):
        return _event(inputs, OcclusionEventType.OUT_OF_FRAME, quality)
    if (
        inputs.formal_visible_mask_available
        and math.isfinite(inputs.mask_overlap_ratio)
        and inputs.mask_overlap_ratio >= full_overlap_ratio
        and inputs.depth_order_supported
        and inputs.d2_reprojection_supported
        and inputs.possible_occluder_ids
    ):
        return _event(
            inputs,
            OcclusionEventType.FULL_OCCLUSION,
            min(quality, inputs.mask_overlap_ratio),
        )
    if inputs.tracker_failed:
        return _event(inputs, OcclusionEventType.TRACK_FAILURE, quality)
    if (
        inputs.detector_attempted
        and inputs.detector_reliable
        and inputs.detection_confirmed_absent
        and not inputs.possible_occluder_ids
        and inputs.persistent_absence_frames >= true_disappearance_confirmation_frames
    ):
        return _event(inputs, OcclusionEventType.TRUE_DISAPPEARANCE, quality)
    if (
        inputs.detector_attempted
        and not inputs.detector_reliable
        and inputs.d2_reprojection_supported
        and math.isfinite(inputs.predicted_in_frame_ratio)
        and inputs.predicted_in_frame_ratio > out_of_frame_ratio
    ):
        return _event(inputs, OcclusionEventType.DETECTOR_MISS, quality)
    return _missing_event(
        inputs, "event_observation_unresolved", status="blocked_by_input"
    )


@dataclass(frozen=True)
class ReappearanceResidual:
    """Multi-cue reappearance residual emitted only for a reappearance event."""

    event_id: str
    previous_track_id: str
    current_track_id: str
    frame_index: int
    identity_residual: float
    position_residual: float
    depth_residual: float
    physical_scale_residual: float
    structure_residual: float
    motion_trend_residual: float
    combined_residual: float
    confidence: float
    valid: bool
    failure_reason: str
    localization_reference: Mapping[str, Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = (
            self.identity_residual,
            self.position_residual,
            self.depth_residual,
            self.physical_scale_residual,
            self.structure_residual,
            self.motion_trend_residual,
            self.combined_residual,
        )
        if self.valid:
            if not all(math.isfinite(float(value)) and value >= 0.0 for value in values):
                raise ValueError("Valid reappearance residuals must be non-negative.")
            if self.failure_reason:
                raise ValueError("Valid reappearance cannot have failure reason.")
        elif any(math.isfinite(float(value)) for value in values) or not self.failure_reason:
            raise ValueError("Invalid reappearance must retain NaN values and a reason.")
        object.__setattr__(
            self, "localization_reference", dict(self.localization_reference)
        )
        object.__setattr__(self, "metadata", dict(self.metadata))


def compute_reappearance_residual(
    *,
    event: OcclusionEventEvidence,
    previous_track_id: str,
    current_track_id: str,
    identity_consistent: bool,
    predicted_position_error_normalized: float,
    previous_depth_m: float,
    current_depth_m: float,
    previous_physical_scale_m: float,
    current_physical_scale_m: float,
    structure_change: float,
    motion_trend_change: float,
    confidence: float,
    eps: float = 1e-8,
) -> ReappearanceResidual:
    """Check identity, predicted location, depth, scale, structure, and motion."""

    nan_values = {
        "identity_residual": float("nan"),
        "position_residual": float("nan"),
        "depth_residual": float("nan"),
        "physical_scale_residual": float("nan"),
        "structure_residual": float("nan"),
        "motion_trend_residual": float("nan"),
        "combined_residual": float("nan"),
    }
    localization = {
        "level": "object_track_reappearance",
        "previous_track_id": previous_track_id,
        "current_track_id": current_track_id,
        "frame_index": event.frame_index,
    }
    if not event.valid or event.event_type != OcclusionEventType.REAPPEARANCE:
        return ReappearanceResidual(
            event.event_id,
            previous_track_id,
            current_track_id,
            event.frame_index,
            **nan_values,
            confidence=0.0,
            valid=False,
            failure_reason="not_applicable_no_event",
            localization_reference=localization,
        )
    values = np.asarray(
        [
            predicted_position_error_normalized,
            previous_depth_m,
            current_depth_m,
            previous_physical_scale_m,
            current_physical_scale_m,
            structure_change,
            motion_trend_change,
            confidence,
        ],
        dtype=float,
    )
    if not np.isfinite(values).all() or min(previous_depth_m, current_depth_m) <= 0.0:
        return ReappearanceResidual(
            event.event_id,
            previous_track_id,
            current_track_id,
            event.frame_index,
            **nan_values,
            confidence=0.0,
            valid=False,
            failure_reason="reappearance_measurement_unavailable",
            localization_reference=localization,
        )
    if min(previous_physical_scale_m, current_physical_scale_m) <= 0.0:
        return ReappearanceResidual(
            event.event_id,
            previous_track_id,
            current_track_id,
            event.frame_index,
            **nan_values,
            confidence=0.0,
            valid=False,
            failure_reason="reappearance_physical_scale_unavailable",
            localization_reference=localization,
        )
    if not identity_consistent:
        return ReappearanceResidual(
            event.event_id,
            previous_track_id,
            current_track_id,
            event.frame_index,
            **nan_values,
            confidence=0.0,
            valid=False,
            failure_reason="reappearance_identity_inconsistent",
            localization_reference=localization,
        )
    identity_residual = 0.0
    position_residual = max(0.0, float(predicted_position_error_normalized))
    depth_residual = abs(
        math.log((float(current_depth_m) + eps) / (float(previous_depth_m) + eps))
    )
    scale_residual = abs(
        math.log(
            (float(current_physical_scale_m) + eps)
            / (float(previous_physical_scale_m) + eps)
        )
    )
    structure_residual = max(0.0, float(structure_change))
    motion_residual = max(0.0, float(motion_trend_change))
    combined = float(
        np.mean(
            [
                identity_residual,
                position_residual,
                depth_residual,
                scale_residual,
                structure_residual,
                motion_residual,
            ]
        )
    )
    return ReappearanceResidual(
        event.event_id,
        previous_track_id,
        current_track_id,
        event.frame_index,
        identity_residual,
        position_residual,
        depth_residual,
        scale_residual,
        structure_residual,
        motion_residual,
        combined,
        float(np.clip(confidence, 0.0, 1.0)),
        True,
        "",
        localization,
        {
            "aggregation": "mean_of_six_unthresholded_components",
            "authenticity_threshold_applied": False,
            "residual_is_authenticity_decision": False,
        },
    )
