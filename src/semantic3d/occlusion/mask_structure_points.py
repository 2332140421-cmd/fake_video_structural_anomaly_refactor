"""Stable ordinary-object points seeded strictly inside formal visible masks."""

from __future__ import annotations

from collections import defaultdict
from typing import Mapping, Sequence

import cv2
import numpy as np

from ..dynamic_3d.track_observation import PointTrack2DObservation
from ..shared_3d_observation import VisibilityStatus
from .mask_observation import InstanceMaskObservation


def adaptive_erosion_pixels(
    mask: np.ndarray,
    *,
    minimum: int = 1,
    maximum: int = 4,
) -> int:
    """Choose erosion solely from mask pixel size, never class or truth label."""

    if minimum < 1 or maximum < minimum:
        raise ValueError("Invalid adaptive erosion bounds.")
    area = int(np.count_nonzero(np.asarray(mask, dtype=bool)))
    if area <= 0:
        return minimum
    characteristic_pixels = float(np.sqrt(area))
    return int(np.clip(round(0.02 * characteristic_pixels), minimum, maximum))


def eroded_mask_interior(mask: np.ndarray, erosion_pixels: int | None = 4) -> np.ndarray:
    """Return a boundary-excluded mask used for formal stable-point seeding."""

    if erosion_pixels is None:
        erosion_pixels = adaptive_erosion_pixels(mask)
    if erosion_pixels < 1:
        raise ValueError("erosion_pixels must be positive.")
    binary = np.asarray(mask, dtype=bool)
    kernel_size = 2 * erosion_pixels + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    return cv2.erode(binary.astype(np.uint8), kernel, iterations=1).astype(bool)


def select_formal_mask_internal_points(
    image: np.ndarray,
    observation: InstanceMaskObservation,
    *,
    max_points: int = 24,
    erosion_pixels: int | None = 4,
    quality_level: float = 0.01,
    min_distance: float = 5.0,
) -> np.ndarray:
    """Select image corners only inside an eroded, non-bbox formal mask.

    The function deliberately returns an empty array when formal mask evidence
    is unavailable. It never falls back to the object's bounding box.
    """

    if max_points < 1:
        raise ValueError("max_points must be positive.")
    if (
        not observation.valid
        or observation.visible_mask is None
        or observation.is_legacy_bbox_fallback
        or not bool(observation.metadata.get("formal_mask_evidence", False))
    ):
        return np.empty((0, 2), dtype=np.float32)
    if image.shape[:2] != observation.image_shape:
        raise ValueError("image and instance mask shapes must match.")
    interior = eroded_mask_interior(observation.visible_mask, erosion_pixels)
    if not np.any(interior):
        return np.empty((0, 2), dtype=np.float32)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=max_points,
        qualityLevel=quality_level,
        minDistance=min_distance,
        mask=interior.astype(np.uint8) * 255,
    )
    if corners is None:
        return np.empty((0, 2), dtype=np.float32)
    points = corners.reshape(-1, 2).astype(np.float32)
    rows = np.clip(np.rint(points[:, 1]).astype(int), 0, interior.shape[0] - 1)
    columns = np.clip(np.rint(points[:, 0]).astype(int), 0, interior.shape[1] - 1)
    return points[interior[rows, columns]]


def track_formal_mask_internal_points(
    images: Mapping[int, np.ndarray],
    masks: Sequence[InstanceMaskObservation],
    *,
    scene_cut_flags: Mapping[int, bool] | None = None,
    max_points: int = 24,
    erosion_pixels: int | None = 4,
) -> tuple[PointTrack2DObservation, ...]:
    """Track stable IDs while using each current mask for validation only.

    KLT predicts the current point from the previous image and point. The
    independently inferred current mask is consulted only after optical flow,
    so no current-mask information leaks into the point prediction.
    """

    valid_masks = [
        item for item in masks
        if item.valid and item.visible_mask is not None
        and not item.is_legacy_bbox_fallback
        and bool(item.metadata.get("formal_mask_evidence", False))
    ]
    if not valid_masks:
        return ()
    track_ids = {item.object_track_id for item in valid_masks}
    if len(track_ids) != 1:
        raise ValueError("One point-tracking call must contain exactly one object track.")
    by_frame = {item.frame_index: item for item in valid_masks}
    frame_indices = sorted(index for index in by_frame if index in images)
    if not frame_indices:
        return ()
    cuts = dict(scene_cut_flags or {})
    first_index = frame_indices[0]
    seeds = select_formal_mask_internal_points(
        images[first_index], by_frame[first_index],
        max_points=max_points, erosion_pixels=erosion_pixels,
    )
    if not len(seeds):
        return ()
    object_track_id = next(iter(track_ids))
    output: list[PointTrack2DObservation] = []
    active: dict[str, np.ndarray] = {
        f"{object_track_id}:mask_point:{index:03d}": point
        for index, point in enumerate(seeds)
    }
    for point_id, point in active.items():
        output.append(PointTrack2DObservation(
            point_id=point_id,
            object_track_id=object_track_id,
            frame_index=first_index,
            pixel_uv=tuple(float(value) for value in point),
            visibility=VisibilityStatus.VISIBLE,
            occlusion_status="visible",
            tracking_confidence=by_frame[first_index].confidence,
            source_tracker="klt_formal_mask_internal",
            valid=True,
            metadata={
                "independent_observation": True,
                "generated_from_projection": False,
                "point_source": "formal_instance_mask_internal",
                "current_mask_used_for_prediction": False,
                "current_mask_used_for_validation_only": True,
            },
        ))
    previous_index = first_index
    for frame_index in frame_indices[1:]:
        if cuts.get(frame_index, False):
            break
        if not active:
            break
        previous_image = images[previous_index]
        current_image = images[frame_index]
        previous_gray = cv2.cvtColor(previous_image, cv2.COLOR_BGR2GRAY) if previous_image.ndim == 3 else previous_image
        current_gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY) if current_image.ndim == 3 else current_image
        point_ids = tuple(active)
        previous_points = np.asarray([active[item] for item in point_ids], dtype=np.float32).reshape(-1, 1, 2)
        current_points, status, error = cv2.calcOpticalFlowPyrLK(
            previous_gray, current_gray, previous_points, None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if current_points is None or status is None:
            break
        interior = eroded_mask_interior(by_frame[frame_index].visible_mask, erosion_pixels)
        next_active: dict[str, np.ndarray] = {}
        errors = np.zeros(len(point_ids), dtype=float) if error is None else error.reshape(-1)
        for position, point_id in enumerate(point_ids):
            point = current_points[position, 0]
            row, column = int(round(float(point[1]))), int(round(float(point[0])))
            valid = (
                bool(status[position, 0])
                and np.isfinite(point).all()
                and 0 <= row < interior.shape[0]
                and 0 <= column < interior.shape[1]
                and bool(interior[row, column])
            )
            if not valid:
                output.append(PointTrack2DObservation.missing(
                    point_id=point_id, object_track_id=object_track_id,
                    frame_index=frame_index, reason="point_left_formal_mask_interior",
                    source_tracker="klt_formal_mask_internal",
                ))
                continue
            confidence = float(np.exp(-max(float(errors[position]), 0.0) / 20.0))
            confidence = min(confidence, by_frame[frame_index].confidence)
            next_active[point_id] = point
            output.append(PointTrack2DObservation(
                point_id=point_id,
                object_track_id=object_track_id,
                frame_index=frame_index,
                pixel_uv=(float(point[0]), float(point[1])),
                visibility=VisibilityStatus.VISIBLE,
                occlusion_status="visible",
                tracking_confidence=confidence,
                source_tracker="klt_formal_mask_internal",
                valid=True,
                metadata={
                    "independent_observation": True,
                    "generated_from_projection": False,
                    "point_source": "formal_instance_mask_internal",
                    "current_mask_used_for_prediction": False,
                    "current_mask_used_for_validation_only": True,
                },
            ))
        active = next_active
        previous_index = frame_index
    return tuple(output)


def group_mask_points_by_track(
    points: Sequence[PointTrack2DObservation],
) -> Mapping[str, tuple[PointTrack2DObservation, ...]]:
    """Group formal mask points by object track for coverage reporting."""

    grouped: dict[str, list[PointTrack2DObservation]] = defaultdict(list)
    for point in points:
        grouped[point.object_track_id].append(point)
    return {
        key: tuple(sorted(value, key=lambda item: (item.frame_index, item.point_id)))
        for key, value in grouped.items()
    }
