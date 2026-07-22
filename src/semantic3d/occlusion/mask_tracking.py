"""Mask propagation and independent current-mask validation."""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional, Sequence

import cv2
import numpy as np

from .mask_observation import InstanceMaskObservation, PredictedObjectSupport, TrackedMaskObservation


def _iou(first: np.ndarray, second: np.ndarray) -> float:
    union = np.count_nonzero(first | second)
    return float(np.count_nonzero(first & second) / union) if union else 1.0


def _boundary_distance(first: np.ndarray, second: np.ndarray) -> float:
    first_boundary = first ^ cv2.erode(first.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    second_boundary = second ^ cv2.erode(second.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    if not np.any(first_boundary) or not np.any(second_boundary):
        return float("nan")
    distance_to_second = cv2.distanceTransform((~second_boundary).astype(np.uint8), cv2.DIST_L2, 3)
    distance_to_first = cv2.distanceTransform((~first_boundary).astype(np.uint8), cv2.DIST_L2, 3)
    return float(0.5 * (np.mean(distance_to_second[first_boundary]) + np.mean(distance_to_first[second_boundary])))


class MaskTracker:
    """Validate a history-only support prediction against an independent mask."""

    provider_name = "history_mask_tracker"

    def track(
        self,
        prediction: PredictedObjectSupport,
        observed: Optional[InstanceMaskObservation],
        *,
        optical_flow_or_point_warp: Optional[np.ndarray] = None,
        propagation_source: Optional[str] = None,
        object_dynamic_observation: Optional[Any] = None,
        independent_point_track_ids: Sequence[str] = (),
        camera_motion_metadata: Optional[Mapping[str, Any]] = None,
        object_history_motion_model: Optional[str] = None,
    ) -> TrackedMaskObservation:
        """Validate a propagated mask against current independent segmentation.

        ``optical_flow_or_point_warp`` must already be generated from prior-mask
        evidence and an independently estimated warp. The current observed mask
        is consumed only after propagation, for IoU and boundary validation.
        """

        shape = prediction.image_shape
        propagated = prediction.support_mask
        source = prediction.prediction_method
        if optical_flow_or_point_warp is not None:
            candidate = np.asarray(optical_flow_or_point_warp, dtype=bool)
            if candidate.shape != shape:
                raise ValueError("optical_flow_or_point_warp must match image shape.")
            if not np.any(candidate):
                propagated = None
            else:
                propagated = candidate
                source = propagation_source or "optical_flow_or_point_warp"
        common = dict(video_id=prediction.video_id, object_track_id=prediction.object_track_id, frame_index=prediction.target_frame_index, image_shape=shape, propagation_source=source)
        if object_dynamic_observation is not None and (
            str(getattr(object_dynamic_observation, "object_track_id", ""))
            != prediction.object_track_id
            or int(getattr(object_dynamic_observation, "frame_index", -1))
            != prediction.target_frame_index
        ):
            return TrackedMaskObservation(**common, propagated_mask=None, observed_mask=None, predicted_support_mask=None, mask_iou=float("nan"), boundary_distance=float("nan"), track_quality=0.0, valid=False, missing_reason="object_dynamic_observation_identity_or_frame_mismatch")
        if not prediction.valid or prediction.support_mask is None:
            return TrackedMaskObservation(**common, propagated_mask=None, observed_mask=None, predicted_support_mask=None, mask_iou=float("nan"), boundary_distance=float("nan"), track_quality=0.0, valid=False, missing_reason=prediction.missing_reason or "invalid_support_prediction")
        if observed is None or not observed.valid or observed.visible_mask is None:
            return TrackedMaskObservation(**common, propagated_mask=None, observed_mask=None, predicted_support_mask=None, mask_iou=float("nan"), boundary_distance=float("nan"), track_quality=0.0, valid=False, missing_reason="independent_current_mask_missing")
        if observed.frame_index != prediction.target_frame_index or observed.object_track_id != prediction.object_track_id:
            return TrackedMaskObservation(**common, propagated_mask=None, observed_mask=None, predicted_support_mask=None, mask_iou=float("nan"), boundary_distance=float("nan"), track_quality=0.0, valid=False, missing_reason="mask_identity_or_frame_mismatch")
        if propagated is None:
            return TrackedMaskObservation(**common, propagated_mask=None, observed_mask=None, predicted_support_mask=None, mask_iou=float("nan"), boundary_distance=float("nan"), track_quality=0.0, valid=False, missing_reason="propagated_mask_unavailable")
        distance = _boundary_distance(propagated, observed.visible_mask)
        if not math.isfinite(distance):
            return TrackedMaskObservation(**common, propagated_mask=None, observed_mask=None, predicted_support_mask=None, mask_iou=float("nan"), boundary_distance=float("nan"), track_quality=0.0, valid=False, missing_reason="boundary_validation_unavailable")
        iou = _iou(propagated, observed.visible_mask)
        diagonal = math.hypot(*shape)
        boundary_score = max(0.0, 1.0 - distance / max(diagonal, 1e-8))
        quality = min(prediction.quality, observed.confidence, 0.5 * (iou + boundary_score))
        return TrackedMaskObservation(
            **common,
            propagated_mask=propagated,
            observed_mask=observed.visible_mask,
            predicted_support_mask=prediction.support_mask,
            mask_iou=iou,
            boundary_distance=distance,
            track_quality=quality,
            valid=True,
            metadata={
                "current_observed_mask_used_for_prediction": False,
                "current_observed_mask_used_for_validation_only": True,
                "mask_intersection_validation": True,
                "boundary_consistency_validation": True,
                "object_motion_prediction_available": prediction.valid,
                "optical_flow_or_point_warp_used": optical_flow_or_point_warp is not None,
                "object_dynamic_observation_used": object_dynamic_observation is not None,
                "independent_point_track_ids": tuple(str(value) for value in independent_point_track_ids),
                "camera_motion_metadata": dict(camera_motion_metadata or {}),
                "object_history_motion_model": object_history_motion_model,
                "legacy_bbox_fallback": observed.is_legacy_bbox_fallback or bool(prediction.metadata.get("legacy_bbox_fallback", False)),
            },
        )


TrackedMaskProvider = MaskTracker
