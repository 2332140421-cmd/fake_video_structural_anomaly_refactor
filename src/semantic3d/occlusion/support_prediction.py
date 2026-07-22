"""History-only target-frame object support prediction."""

from __future__ import annotations

import math
from typing import Sequence

import cv2
import numpy as np

from ..dynamic_3d import DynamicGeometryMode
from .mask_observation import InstanceMaskObservation, PredictedObjectSupport


def _centroid(mask: np.ndarray) -> np.ndarray:
    rows, columns = np.nonzero(mask)
    return np.asarray([float(columns.mean()), float(rows.mean())])


def predict_object_support(
    history: Sequence[InstanceMaskObservation],
    *,
    target_frame_index: int,
    geometry_mode: DynamicGeometryMode | str,
    camera_rotation_affine: np.ndarray | None = None,
    object_motion_translation_uv: tuple[float, float] | None = None,
    projected_full_se3_mask: np.ndarray | None = None,
) -> PredictedObjectSupport:
    """Warp the latest visible mask using only t-2/t-1 historical motion."""

    mode = DynamicGeometryMode(geometry_mode)
    past = sorted((item for item in history if item.valid and item.frame_index < target_frame_index and item.visible_mask is not None), key=lambda item: item.frame_index)
    if not past:
        return PredictedObjectSupport.missing(video_id="unknown", object_track_id="unknown", target_frame_index=target_frame_index, image_shape=(1, 1), geometry_mode=mode, reason="missing_mask_history")
    latest = past[-1]
    common = dict(video_id=latest.video_id, object_track_id=latest.object_track_id, target_frame_index=target_frame_index, image_shape=latest.image_shape, geometry_mode=mode)
    if mode == DynamicGeometryMode.UNAVAILABLE:
        return PredictedObjectSupport.missing(**common, reason="dynamic_geometry_unavailable")
    if len(past) < 2:
        return PredictedObjectSupport.missing(**common, reason="insufficient_mask_history")
    previous, latest = past[-2:]
    assert previous.visible_mask is not None and latest.visible_mask is not None
    if mode == DynamicGeometryMode.FULL_SE3_3D:
        if projected_full_se3_mask is None:
            return PredictedObjectSupport.missing(
                **common, reason="full_se3_support_projector_unavailable"
            )
        predicted = np.asarray(projected_full_se3_mask, dtype=bool)
        if predicted.shape != latest.image_shape or not np.any(predicted):
            return PredictedObjectSupport.missing(
                **common, reason="invalid_full_se3_projected_support"
            )
        expected_area = float(np.count_nonzero(latest.visible_mask))
        quality = min(previous.confidence, latest.confidence)
        legacy = previous.is_legacy_bbox_fallback or latest.is_legacy_bbox_fallback
        if legacy:
            quality = min(quality, 0.25)
        return PredictedObjectSupport(
            **common,
            support_mask=predicted,
            predicted_area=expected_area,
            in_frame_ratio=min(1.0, float(np.count_nonzero(predicted)) / max(expected_area, 1.0)),
            history_frames=(previous.frame_index, latest.frame_index),
            prediction_method="full_se3_history_support_projection",
            quality=quality,
            valid=True,
            metadata={
                "current_frame_used_for_prediction": False,
                "observed_target_mask_used": False,
                "complete_3d_volume_claimed": False,
                "full_se3_projection_path": True,
                "legacy_bbox_fallback": legacy,
            },
        )
    delta = (
        np.asarray(object_motion_translation_uv, dtype=float)
        if object_motion_translation_uv is not None
        else _centroid(latest.visible_mask) - _centroid(previous.visible_mask)
    )
    if delta.shape != (2,) or not np.isfinite(delta).all():
        return PredictedObjectSupport.missing(**common, reason="invalid_object_motion_translation")
    history_gap = latest.frame_index - previous.frame_index
    target_gap = target_frame_index - latest.frame_index
    if history_gap <= 0 or target_gap <= 0:
        return PredictedObjectSupport.missing(**common, reason="invalid_mask_history_order")
    shift = delta * (target_gap / history_gap)
    affine = np.asarray([[1.0, 0.0, shift[0]], [0.0, 1.0, shift[1]]], dtype=float)
    method = "history_mask_constant_translation"
    if mode == DynamicGeometryMode.ROTATION_COMPENSATED and camera_rotation_affine is not None:
        rotation = np.asarray(camera_rotation_affine, dtype=float)
        if rotation.shape != (2, 3) or not np.isfinite(rotation).all():
            return PredictedObjectSupport.missing(**common, reason="invalid_rotation_affine")
        homogeneous = np.vstack([affine, [0.0, 0.0, 1.0]])
        affine = (np.vstack([rotation, [0.0, 0.0, 1.0]]) @ homogeneous)[:2]
        method = "rotation_compensated_history_mask_warp"
    elif mode == DynamicGeometryMode.ROTATION_COMPENSATED:
        method = "rotation_mode_2d_history_motion_only"
    height, width = latest.image_shape
    predicted = cv2.warpAffine(latest.visible_mask.astype(np.uint8), affine, (width, height), flags=cv2.INTER_NEAREST, borderValue=0).astype(bool)
    expected_area = float(np.count_nonzero(latest.visible_mask))
    in_frame_area = float(np.count_nonzero(predicted))
    ratio = min(1.0, in_frame_area / expected_area) if expected_area > 0.0 else 0.0
    if not np.any(predicted):
        return PredictedObjectSupport.missing(**common, reason="predicted_support_outside_frame")
    quality = min(previous.confidence, latest.confidence)
    if previous.is_legacy_bbox_fallback or latest.is_legacy_bbox_fallback:
        quality = min(quality, 0.25)
    return PredictedObjectSupport(
        **common,
        support_mask=predicted,
        predicted_area=expected_area,
        in_frame_ratio=ratio,
        history_frames=(previous.frame_index, latest.frame_index),
        prediction_method=method,
        quality=quality,
        valid=True,
        metadata={
            "current_frame_used_for_prediction": False,
            "observed_target_mask_used": False,
            "translation_uv": tuple(float(value) for value in shift),
            "object_motion_model_supplied": object_motion_translation_uv is not None,
            "camera_rotation_compensation_applied": camera_rotation_affine is not None,
            "complete_3d_volume_claimed": False,
            "legacy_bbox_fallback": previous.is_legacy_bbox_fallback or latest.is_legacy_bbox_fallback,
        },
    )
