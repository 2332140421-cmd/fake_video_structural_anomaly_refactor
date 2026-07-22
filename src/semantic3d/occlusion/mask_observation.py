"""Instance-mask, boundary, tracking, and predicted-support contracts."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import cv2
import numpy as np

from ..dynamic_3d import DynamicGeometryMode


def _mask_or_none(value: Optional[np.ndarray], image_shape: tuple[int, int], name: str) -> Optional[np.ndarray]:
    if value is None:
        return None
    mask = np.asarray(value, dtype=bool)
    if mask.shape != image_shape:
        raise ValueError(f"{name} shape {mask.shape} must equal image shape {image_shape}.")
    output = mask.copy()
    output.setflags(write=False)
    return output


def mask_bbox(mask: np.ndarray) -> Optional[tuple[float, float, float, float]]:
    """Return an exclusive xyxy bbox for a non-empty binary mask."""

    rows, columns = np.nonzero(mask)
    if not len(rows):
        return None
    return float(columns.min()), float(rows.min()), float(columns.max() + 1), float(rows.max() + 1)


def mask_boundary_points(mask: np.ndarray) -> tuple[tuple[float, float], ...]:
    """Extract deterministic external contour points from one binary mask."""

    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    points = []
    for contour in contours:
        points.extend((float(value[0][0]), float(value[0][1])) for value in contour)
    return tuple(points)


@dataclass(frozen=True)
class InstanceMaskObservation:
    """One visible or amodal object mask with explicit source provenance."""

    video_id: str
    frame_index: int
    object_track_id: str
    semantic_label: str
    image_shape: tuple[int, int]
    visible_mask: Optional[np.ndarray]
    amodal_mask: Optional[np.ndarray]
    mask_area: float
    mask_bbox: Optional[tuple[float, float, float, float]]
    boundary_points: tuple[tuple[float, float], ...]
    confidence: float
    source_provider: str
    is_visible_mask: bool
    is_amodal_mask: bool
    valid: bool
    missing_reason: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        shape = tuple(int(value) for value in self.image_shape)
        if len(shape) != 2 or any(value <= 0 for value in shape):
            raise ValueError("image_shape must be positive (height, width).")
        visible = _mask_or_none(self.visible_mask, shape, "visible_mask")
        amodal = _mask_or_none(self.amodal_mask, shape, "amodal_mask")
        confidence = float(self.confidence)
        area = float(self.mask_area)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("Mask confidence must be in [0, 1].")
        if self.is_visible_mask == self.is_amodal_mask:
            raise ValueError("A mask observation must be explicitly visible or amodal, not both.")
        selected = visible if self.is_visible_mask else amodal
        if self.valid:
            if selected is None or not np.any(selected) or self.missing_reason:
                raise ValueError("Valid mask observation requires a non-empty selected mask.")
            actual_area = float(np.count_nonzero(selected))
            if not math.isclose(area, actual_area, rel_tol=0.0, abs_tol=1e-6):
                raise ValueError("mask_area must equal the selected binary-mask area.")
            expected_bbox = mask_bbox(selected)
            if self.mask_bbox is None or tuple(self.mask_bbox) != expected_bbox:
                raise ValueError("mask_bbox must be derived from the selected mask.")
            if not self.source_provider.strip():
                raise ValueError("Valid mask requires source_provider.")
        else:
            if selected is not None or not math.isnan(area) or self.mask_bbox is not None or not self.missing_reason:
                raise ValueError("Invalid mask requires no mask/bbox, NaN area, and a reason.")
        object.__setattr__(self, "image_shape", shape)
        object.__setattr__(self, "visible_mask", visible)
        object.__setattr__(self, "amodal_mask", amodal)
        object.__setattr__(self, "boundary_points", tuple(tuple(map(float, point)) for point in self.boundary_points))
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "mask_area", area)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def from_visible_mask(
        cls,
        *,
        video_id: str,
        frame_index: int,
        object_track_id: str,
        semantic_label: str,
        mask: np.ndarray,
        confidence: float,
        source_provider: str,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "InstanceMaskObservation":
        """Create a visible-mask observation without implying amodal support."""

        binary = np.asarray(mask, dtype=bool)
        return cls(
            video_id, frame_index, object_track_id, semantic_label, binary.shape,
            binary, None, float(np.count_nonzero(binary)), mask_bbox(binary),
            mask_boundary_points(binary), confidence, source_provider, True, False,
            True, metadata=dict(metadata or {}),
        )

    @classmethod
    def missing(
        cls,
        *,
        video_id: str,
        frame_index: int,
        object_track_id: str,
        semantic_label: str,
        image_shape: tuple[int, int],
        reason: str,
        source_provider: str,
        is_amodal: bool = False,
    ) -> "InstanceMaskObservation":
        """Create an invalid mask without fabricating a bbox-filled region."""

        return cls(
            video_id, frame_index, object_track_id, semantic_label, image_shape,
            None, None, float("nan"), None, (), 0.0, source_provider,
            not is_amodal, is_amodal, False, reason,
        )

    @property
    def selected_mask(self) -> Optional[np.ndarray]:
        """Return the declared mask representation only."""

        return self.visible_mask if self.is_visible_mask else self.amodal_mask

    @property
    def is_legacy_bbox_fallback(self) -> bool:
        """Return whether this is an explicitly low-quality bbox diagnostic."""

        return bool(self.metadata.get("legacy_bbox_fallback", False))


@dataclass(frozen=True)
class MaskBoundaryObservation:
    """Independent visible boundary evidence for one object and frame."""

    object_track_id: str
    frame_index: int
    boundary_points: tuple[tuple[float, float], ...]
    source_provider: str
    independent_observation: bool
    quality: float
    valid: bool
    missing_reason: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        quality = float(self.quality)
        if not math.isfinite(quality) or not 0.0 <= quality <= 1.0:
            raise ValueError("Boundary quality must be in [0, 1].")
        if self.valid and (not self.boundary_points or not self.independent_observation or self.missing_reason):
            raise ValueError("Valid observed boundary must be independent and non-empty.")
        if not self.valid and not self.missing_reason:
            raise ValueError("Invalid boundary requires missing_reason.")
        object.__setattr__(self, "quality", quality)
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class PredictedObjectSupport:
    """History-only prediction of an object's target-frame support region."""

    video_id: str
    object_track_id: str
    target_frame_index: int
    image_shape: tuple[int, int]
    support_mask: Optional[np.ndarray]
    predicted_area: float
    in_frame_ratio: float
    history_frames: tuple[int, ...]
    geometry_mode: DynamicGeometryMode | str
    prediction_method: str
    quality: float
    valid: bool
    missing_reason: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        mode = DynamicGeometryMode(self.geometry_mode)
        shape = tuple(int(value) for value in self.image_shape)
        mask = _mask_or_none(self.support_mask, shape, "support_mask")
        quality, ratio, area = float(self.quality), float(self.in_frame_ratio), float(self.predicted_area)
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in (quality, ratio)):
            raise ValueError("Support quality and in_frame_ratio must be in [0, 1].")
        if any(frame >= self.target_frame_index for frame in self.history_frames):
            raise ValueError("Predicted support cannot consume target/current frame evidence.")
        if self.valid:
            if mask is None or area <= 0.0 or len(self.history_frames) < 2 or self.missing_reason:
                raise ValueError("Valid support prediction requires mask and two history frames.")
            if bool(self.metadata.get("current_frame_used_for_prediction", True)):
                raise ValueError("Current-frame mask cannot be used to predict current support.")
        else:
            if mask is not None or not math.isnan(area) or not self.missing_reason:
                raise ValueError("Invalid support prediction requires no mask, NaN area, and reason.")
        object.__setattr__(self, "geometry_mode", mode)
        object.__setattr__(self, "image_shape", shape)
        object.__setattr__(self, "support_mask", mask)
        object.__setattr__(self, "history_frames", tuple(self.history_frames))
        object.__setattr__(self, "quality", quality)
        object.__setattr__(self, "in_frame_ratio", ratio)
        object.__setattr__(self, "predicted_area", area)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def missing(cls, *, video_id: str, object_track_id: str, target_frame_index: int, image_shape: tuple[int, int], geometry_mode: DynamicGeometryMode | str, reason: str) -> "PredictedObjectSupport":
        return cls(video_id, object_track_id, target_frame_index, image_shape, None, float("nan"), 0.0, (), geometry_mode, "unavailable", 0.0, False, reason)


@dataclass(frozen=True)
class TrackedMaskObservation:
    """Comparison of history-propagated support and independent current mask."""

    video_id: str
    object_track_id: str
    frame_index: int
    image_shape: tuple[int, int]
    propagated_mask: Optional[np.ndarray]
    observed_mask: Optional[np.ndarray]
    predicted_support_mask: Optional[np.ndarray]
    mask_iou: float
    boundary_distance: float
    track_quality: float
    propagation_source: str
    valid: bool
    missing_reason: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        shape = tuple(self.image_shape)
        propagated = _mask_or_none(self.propagated_mask, shape, "propagated_mask")
        observed = _mask_or_none(self.observed_mask, shape, "observed_mask")
        predicted = _mask_or_none(self.predicted_support_mask, shape, "predicted_support_mask")
        iou, distance, quality = float(self.mask_iou), float(self.boundary_distance), float(self.track_quality)
        if not math.isfinite(quality) or not 0.0 <= quality <= 1.0:
            raise ValueError("track_quality must be in [0, 1].")
        if self.valid:
            if any(mask is None for mask in (propagated, observed, predicted)):
                raise ValueError("Valid tracked mask requires propagated, predicted, and observed masks.")
            if not (math.isfinite(iou) and 0.0 <= iou <= 1.0 and math.isfinite(distance) and distance >= 0.0):
                raise ValueError("Valid mask tracking metrics are invalid.")
            if self.missing_reason or bool(self.metadata.get("current_observed_mask_used_for_prediction", True)):
                raise ValueError("Tracked mask prediction must be independent of current observed mask.")
        else:
            if any(mask is not None for mask in (propagated, observed, predicted)) or not math.isnan(iou) or not math.isnan(distance) or not self.missing_reason:
                raise ValueError("Invalid tracked mask requires missing masks/NaN metrics and a reason.")
        object.__setattr__(self, "propagated_mask", propagated)
        object.__setattr__(self, "observed_mask", observed)
        object.__setattr__(self, "predicted_support_mask", predicted)
        object.__setattr__(self, "mask_iou", iou)
        object.__setattr__(self, "boundary_distance", distance)
        object.__setattr__(self, "track_quality", quality)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def independently_observed_mask(self) -> Optional[np.ndarray]:
        """Return the current segmentation used only for independent validation."""

        return self.observed_mask

    @property
    def history_predicted_mask(self) -> Optional[np.ndarray]:
        """Return the target support predicted without the current mask."""

        return self.predicted_support_mask

    @property
    def area_change_ratio(self) -> float:
        """Return observed/predicted area ratio, or NaN when evidence is missing."""

        if not self.valid or self.observed_mask is None or self.predicted_support_mask is None:
            return float("nan")
        predicted_area = float(np.count_nonzero(self.predicted_support_mask))
        return float(np.count_nonzero(self.observed_mask) / predicted_area) if predicted_area > 0.0 else float("nan")

    @property
    def assignment_consistency(self) -> float:
        """Return identity consistency for this accepted one-to-one validation."""

        return 1.0 if self.valid else float("nan")
