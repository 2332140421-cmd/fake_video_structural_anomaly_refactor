"""Spatial mapping of valid unified evidence to native supports."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import UnifiedEvidence


@dataclass(frozen=True)
class SpatialEvidenceProducts:
    """Separate object, boundary, point, track, and fused frame maps."""

    object_evidence_maps: Mapping[tuple[str, str, int], np.ndarray]
    boundary_evidence_maps: Mapping[tuple[str, str, int], np.ndarray]
    point_evidence_maps: Mapping[tuple[str, str, int], np.ndarray]
    track_evidence_maps: Mapping[tuple[str, str, int], np.ndarray]
    frame_spatial_evidence_maps: Mapping[tuple[str, str, int], np.ndarray]
    object_scores: Mapping[tuple[str, str, str], float]
    track_scores: Mapping[tuple[str, str, str], float]
    manifest_rows: tuple[Mapping[str, Any], ...]
    skipped_reason_counts: Mapping[str, int]
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _place(array: np.ndarray, x: float, y: float, value: float) -> bool:
    column, row = int(round(x)), int(round(y))
    if row < 0 or column < 0 or row >= array.shape[0] or column >= array.shape[1]:
        return False
    current = float(array[row, column])
    array[row, column] = value if math.isnan(current) else max(current, value)
    return True


def _scaled_xy(
    xy: Sequence[float],
    reference: Mapping[str, Any],
    target_shape: tuple[int, int],
) -> tuple[float, float]:
    x, y = float(xy[0]), float(xy[1])
    source_shape = reference.get("source_image_shape")
    if source_shape and len(source_shape) == 2:
        source_height, source_width = float(source_shape[0]), float(source_shape[1])
        if source_height > 0 and source_width > 0:
            x *= target_shape[1] / source_width
            y *= target_shape[0] / source_height
    return x, y


def _update_score(
    scores: dict[tuple[str, str, str], float],
    identity: tuple[str, str, str],
    value: float,
) -> None:
    if identity:
        scores[identity] = max(scores.get(identity, -math.inf), value)


def map_unified_evidence_spatially(
    evidences: Sequence[UnifiedEvidence],
    *,
    image_shape: tuple[int, int],
) -> SpatialEvidenceProducts:
    """Rasterize only evidence with explicit pixel/Mask support."""

    height, width = int(image_shape[0]), int(image_shape[1])
    if height <= 0 or width <= 0:
        raise ValueError("image_shape must be positive.")
    keys = sorted(
        {
            (item.video_id, item.clip_id, int(item.frame_index))
            for item in evidences
            if item.frame_index is not None
        }
    )
    def blank() -> dict[tuple[str, str, int], np.ndarray]:
        return {
            key: np.full((height, width), np.nan, dtype=np.float32)
            for key in keys
        }

    object_maps, boundary_maps = blank(), blank()
    point_maps, track_maps, frame_maps = blank(), blank(), blank()
    object_scores: dict[tuple[str, str, str], float] = {}
    track_scores: dict[tuple[str, str, str], float] = {}
    manifest: list[dict[str, Any]] = []
    skipped: dict[str, int] = {}
    for item in evidences:
        reference = item.spatial_reference
        kind = str(reference.get("kind", "reference_only"))
        rasterized = False
        reason = ""
        if not item.valid:
            reason = item.failure_reason
        elif item.frame_index is None:
            reason = "spatial_frame_index_unavailable"
        else:
            key = (item.video_id, item.clip_id, item.frame_index)
            target: np.ndarray | None = None
            points: list[tuple[float, float]] = []
            if item.object_id:
                _update_score(
                    object_scores,
                    (item.video_id, item.clip_id, item.object_id),
                    item.residual_value,
                )
            if item.track_id:
                _update_score(
                    track_scores,
                    (item.video_id, item.clip_id, item.track_id),
                    item.residual_value,
                )
            if kind == "point":
                target = point_maps[key]
                xy = reference.get("xy")
                if xy is not None:
                    points = [_scaled_xy(xy, reference, image_shape)]
            elif kind == "boundary":
                target = boundary_maps[key]
                boundary = reference.get("boundary_xy")
                if boundary:
                    points = [
                        _scaled_xy(xy, reference, image_shape) for xy in boundary
                    ]
                elif reference.get("xy") is not None:
                    points = [
                        _scaled_xy(reference["xy"], reference, image_shape)
                    ]
            elif kind in {"object_mask", "object_support_points"}:
                target = object_maps[key]
                if kind == "object_mask" and reference.get("mask") is not None:
                    mask = np.asarray(reference["mask"], dtype=bool)
                    if mask.shape != image_shape:
                        reason = "object_mask_shape_mismatch"
                    elif np.any(mask):
                        target[mask] = np.fmax(target[mask], item.residual_value)
                        rasterized = True
                else:
                    points = [
                        _scaled_xy(xy, reference, image_shape)
                        for xy in reference.get("xy_points", ())
                    ]
            elif kind == "track":
                target = track_maps[key]
                xy = reference.get("xy")
                if xy is not None:
                    points = [_scaled_xy(xy, reference, image_shape)]
            else:
                reason = "spatial_support_reference_only"
            if target is not None and points:
                rasterized = any(
                    _place(target, x, y, item.residual_value) for x, y in points
                ) or rasterized
            if rasterized:
                source_map = target
                assert source_map is not None
                finite = np.isfinite(source_map)
                frame_maps[key][finite] = np.fmax(
                    frame_maps[key][finite], source_map[finite]
                )
            elif not reason:
                reason = "spatial_support_out_of_frame_or_empty"
        if reason:
            skipped[reason] = skipped.get(reason, 0) + 1
        manifest.append(
            {
                "evidence_id": item.evidence_id,
                "video_id": item.video_id,
                "clip_id": item.clip_id,
                "frame_id": item.frame_id,
                "frame_index": item.frame_index,
                "branch_group": item.branch_group.value,
                "object_id": item.object_id,
                "track_id": item.track_id,
                "localization_kind": kind,
                "spatial_reference": dict(reference),
                "rasterized": rasterized,
                "failure_reason": reason,
            }
        )
    return SpatialEvidenceProducts(
        object_evidence_maps=object_maps,
        boundary_evidence_maps=boundary_maps,
        point_evidence_maps=point_maps,
        track_evidence_maps=track_maps,
        frame_spatial_evidence_maps=frame_maps,
        object_scores=dict(sorted(object_scores.items())),
        track_scores=dict(sorted(track_scores.items())),
        manifest_rows=tuple(manifest),
        skipped_reason_counts=dict(sorted(skipped.items())),
        metadata={
            "map_shape": image_shape,
            "missing_pixels_are_nan": True,
            "reference_only_evidence_is_not_rasterized": True,
            "formal_threshold_selected": False,
            "classification_output": False,
        },
    )


def rank_object_and_track_evidence(
    products: SpatialEvidenceProducts,
) -> tuple[Mapping[str, Any], ...]:
    """Return deterministic object/track rankings without a decision threshold."""

    rows = [
        {
            "video_id": key[0],
            "clip_id": key[1],
            "identity_type": "object",
            "identity_id": key[2],
            "score": value,
        }
        for key, value in products.object_scores.items()
    ]
    rows.extend(
        {
            "video_id": key[0],
            "clip_id": key[1],
            "identity_type": "track",
            "identity_id": key[2],
            "score": value,
        }
        for key, value in products.track_scores.items()
    )
    return tuple(
        sorted(
            rows,
            key=lambda row: (
                -float(row["score"]),
                row["video_id"],
                row["clip_id"],
                row["identity_type"],
                row["identity_id"],
            ),
        )
    )
