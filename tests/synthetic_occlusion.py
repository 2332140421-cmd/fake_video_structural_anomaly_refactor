"""Deterministic mask truth for P3-C visibility and occlusion tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from semantic3d.dynamic_3d import DynamicGeometryMode
from semantic3d.occlusion import InstanceMaskObservation, PredictedObjectSupport


SHAPE = (64, 64)


def rectangle(x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    mask = np.zeros(SHAPE, dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask


def observed(track_id: str, frame: int, mask: np.ndarray, *, label: str = "object", legacy: bool = False) -> InstanceMaskObservation:
    return InstanceMaskObservation.from_visible_mask(
        video_id="synthetic", frame_index=frame, object_track_id=track_id,
        semantic_label=label, mask=mask, confidence=0.25 if legacy else 1.0,
        source_provider="legacy_bbox" if legacy else "synthetic_truth",
        metadata={"legacy_bbox_fallback": legacy, "formal_mask_evidence": not legacy},
    )


def support(track_id: str, frame: int, mask: np.ndarray, *, mode=DynamicGeometryMode.STATIC_CAMERA_3D, in_frame_ratio: float = 1.0, legacy: bool = False) -> PredictedObjectSupport:
    return PredictedObjectSupport(
        video_id="synthetic", object_track_id=track_id, target_frame_index=frame,
        image_shape=SHAPE, support_mask=mask, predicted_area=float(np.count_nonzero(mask)) / max(in_frame_ratio, 1e-8),
        in_frame_ratio=in_frame_ratio, history_frames=(frame - 2, frame - 1),
        geometry_mode=mode, prediction_method="synthetic_history_prediction",
        quality=0.25 if legacy else 1.0, valid=True,
        metadata={"current_frame_used_for_prediction": False, "legacy_bbox_fallback": legacy},
    )


@dataclass(frozen=True)
class NormalOcclusionScene:
    background_support: PredictedObjectSupport
    foreground_support: PredictedObjectSupport
    background_visible: InstanceMaskObservation
    foreground_visible: InstanceMaskObservation


def normal_occlusion_scene() -> NormalOcclusionScene:
    background_full = rectangle(16, 16, 48, 48)
    foreground = rectangle(32, 12, 52, 52)
    background_visible = background_full & ~foreground
    return NormalOcclusionScene(
        support("background", 2, background_full),
        support("foreground", 2, foreground),
        observed("background", 2, background_visible),
        observed("foreground", 2, foreground),
    )


@dataclass(frozen=True)
class SyntheticOcclusionScenario:
    """Named synthetic situation used to audit state-machine coverage."""

    name: str
    geometry_mode: DynamicGeometryMode
    scene_cut: bool = False
    detector_confidence: float = 1.0
    has_mask: bool = True
    description: str = ""


def synthetic_occlusion_scenarios() -> dict[str, SyntheticOcclusionScenario]:
    """Return the required A-N scenario catalogue without truth-label leakage."""

    specifications = (
        ("normal_foreground_occlusion", DynamicGeometryMode.STATIC_CAMERA_3D, False, 1.0, True),
        ("reversed_depth_order", DynamicGeometryMode.STATIC_CAMERA_3D, False, 1.0, True),
        ("partial_occlusion", DynamicGeometryMode.STATIC_CAMERA_3D, False, 1.0, True),
        ("full_occlusion_then_reappearance", DynamicGeometryMode.STATIC_CAMERA_3D, False, 1.0, True),
        ("object_leaves_frame", DynamicGeometryMode.STATIC_CAMERA_3D, False, 1.0, True),
        ("single_frame_detector_miss", DynamicGeometryMode.STATIC_CAMERA_3D, False, 0.1, True),
        ("unexplained_disappearance", DynamicGeometryMode.STATIC_CAMERA_3D, False, 1.0, True),
        ("unexplained_appearance", DynamicGeometryMode.STATIC_CAMERA_3D, False, 1.0, True),
        ("crossing_objects", DynamicGeometryMode.STATIC_CAMERA_3D, False, 1.0, True),
        ("wrong_reidentification", DynamicGeometryMode.STATIC_CAMERA_3D, False, 1.0, True),
        ("rotation_only_camera", DynamicGeometryMode.ROTATION_COMPENSATED, False, 1.0, True),
        ("scene_cut", DynamicGeometryMode.UNAVAILABLE, True, 1.0, True),
        ("missing_mask", DynamicGeometryMode.STATIC_CAMERA_3D, False, 1.0, False),
        ("bbox_overlap_without_contour_contact", DynamicGeometryMode.STATIC_CAMERA_3D, False, 1.0, True),
    )
    return {
        name: SyntheticOcclusionScenario(
            name=name,
            geometry_mode=mode,
            scene_cut=cut,
            detector_confidence=confidence,
            has_mask=has_mask,
            description=name.replace("_", " "),
        )
        for name, mode, cut, confidence, has_mask in specifications
    }
