"""Boundary-depth consistency from one shared canonical depth observation."""

from __future__ import annotations

import math
from typing import Optional, Sequence

import cv2
import numpy as np

from ..shared_3d_observation import Shared3DFrameObservation
from ..validity import ResidualEvidence
from .residual_types import EvidenceRole, Static3DContext


def _depth_value(
    depth_map: np.ndarray, valid_mask: np.ndarray, row: int, column: int
) -> float | None:
    if (
        row < 0
        or row >= depth_map.shape[0]
        or column < 0
        or column >= depth_map.shape[1]
        or not valid_mask[row, column]
    ):
        return None
    value = float(depth_map[row, column])
    return value if math.isfinite(value) and value > 0.0 else None


def _mask_pairs(mask: np.ndarray, max_samples: int) -> list[tuple[int, int, int, int]]:
    """Pair inner contour pixels with nearby outer-mask pixels."""

    kernel = np.ones((3, 3), dtype=np.uint8)
    mask_u8 = mask.astype(np.uint8)
    eroded = cv2.erode(mask_u8, kernel, iterations=1).astype(bool)
    dilated = cv2.dilate(mask_u8, kernel, iterations=1).astype(bool)
    inner = mask & ~eroded
    outer = dilated & ~mask
    coordinates = np.argwhere(inner)
    if coordinates.shape[0] > max_samples:
        indices = np.linspace(0, coordinates.shape[0] - 1, max_samples).astype(int)
        coordinates = coordinates[indices]
    pairs: list[tuple[int, int, int, int]] = []
    for row, column in coordinates:
        r0, r1 = max(0, row - 1), min(mask.shape[0], row + 2)
        c0, c1 = max(0, column - 1), min(mask.shape[1], column + 2)
        neighbours = np.argwhere(outer[r0:r1, c0:c1])
        if neighbours.size == 0:
            continue
        outer_row, outer_col = neighbours[0]
        pairs.append((int(row), int(column), int(r0 + outer_row), int(c0 + outer_col)))
    return pairs


def _bbox_fallback_pairs(
    bbox: Sequence[float], width: int, height: int
) -> list[tuple[int, int, int, int]]:
    """Sample only sparse middle edge segments; a bbox is not an instance contour."""

    if len(bbox) != 4:
        return []
    x1, y1, x2, y2 = (int(round(float(value))) for value in bbox)
    x1, x2 = max(1, x1), min(width - 2, x2)
    y1, y2 = max(1, y1), min(height - 2, y2)
    if x2 <= x1 + 2 or y2 <= y1 + 2:
        return []
    fractions = (0.35, 0.5, 0.65)
    pairs: list[tuple[int, int, int, int]] = []
    for fraction in fractions:
        column = int(round(x1 + fraction * (x2 - x1)))
        row = int(round(y1 + fraction * (y2 - y1)))
        pairs.extend(
            [
                (y1 + 1, column, y1 - 1, column),
                (y2 - 1, column, y2 + 1, column),
                (row, x1 + 1, row, x1 - 1),
                (row, x2 - 1, row, x2 + 1),
            ]
        )
    return pairs


class BoundaryDepth3DResidual:
    """Measure depth discontinuity alignment at a mask or low-quality bbox boundary."""

    def __init__(self, minimum_relative_jump: float = 0.03, max_samples: int = 512) -> None:
        if minimum_relative_jump < 0.0:
            raise ValueError("minimum_relative_jump must be non-negative.")
        if max_samples <= 0:
            raise ValueError("max_samples must be positive.")
        self.minimum_relative_jump = float(minimum_relative_jump)
        self.max_samples = int(max_samples)

    def evaluate(
        self,
        frame: Shared3DFrameObservation,
        object_id: str,
        *,
        instance_mask: Optional[np.ndarray] = None,
        bbox: Optional[Sequence[float]] = None,
    ) -> ResidualEvidence:
        """Return contradictory boundary-depth ratio with explicit source quality."""

        try:
            obj = Static3DContext(frame).object_by_id(object_id)
        except KeyError:
            return ResidualEvidence.missing(
                "boundary_depth_3d", "object_not_found", source_ids=(object_id,)
            )
        try:
            depth_map = frame.depth.require_geometry_depth()
        except ValueError as error:
            return ResidualEvidence.missing(
                "boundary_depth_3d",
                "invalid_geometry_depth",
                source_ids=(object_id,),
                metadata={"error": str(error)},
            )
        valid_mask = (
            np.asarray(frame.depth.valid_mask, dtype=bool)
            if frame.depth.valid_mask is not None
            else np.isfinite(depth_map) & (depth_map > 0.0)
        )
        boundary_source: str
        source_quality: float
        if instance_mask is not None:
            mask = np.asarray(instance_mask, dtype=bool)
            if mask.shape != depth_map.shape or not np.any(mask):
                return ResidualEvidence.missing(
                    "boundary_depth_3d",
                    "invalid_instance_mask",
                    source_ids=(object_id,),
                )
            pairs = _mask_pairs(mask, self.max_samples)
            boundary_source = "instance_mask"
            source_quality = 1.0
        else:
            fallback_bbox = (
                bbox if bbox is not None else obj.metadata.get("source_bbox")
            )
            if fallback_bbox is None:
                return ResidualEvidence.missing(
                    "boundary_depth_3d",
                    "no_boundary_observation",
                    source_ids=(object_id,),
                )
            pairs = _bbox_fallback_pairs(
                fallback_bbox, frame.image_width, frame.image_height
            )
            boundary_source = "bbox_sparse_fallback"
            source_quality = 0.30
        if not pairs:
            return ResidualEvidence.missing(
                "boundary_depth_3d",
                "no_boundary_sample_pairs",
                source_ids=(object_id,),
                metadata={"boundary_source": boundary_source},
            )

        signed_jumps: list[float] = []
        relative_magnitudes: list[float] = []
        missing = 0
        for inner_row, inner_col, outer_row, outer_col in pairs:
            inner = _depth_value(depth_map, valid_mask, inner_row, inner_col)
            outer = _depth_value(depth_map, valid_mask, outer_row, outer_col)
            if inner is None or outer is None:
                missing += 1
                continue
            jump = outer - inner
            normalizer = max(abs(inner), abs(outer), 1e-12)
            signed_jumps.append(jump / normalizer)
            relative_magnitudes.append(abs(jump) / normalizer)
        valid_count = len(signed_jumps)
        if valid_count == 0:
            return ResidualEvidence.missing(
                "boundary_depth_3d",
                "missing_depth_at_boundary",
                source_ids=(object_id,),
                metadata={
                    "boundary_source": boundary_source,
                    "missing_depth_boundary_ratio": 1.0,
                },
            )
        jumps = np.asarray(signed_jumps, dtype=float)
        magnitudes = np.asarray(relative_magnitudes, dtype=float)
        aligned = magnitudes >= self.minimum_relative_jump
        aligned_ratio = float(np.mean(aligned))
        if np.any(aligned):
            dominant_sign = float(np.sign(np.median(jumps[aligned])))
            contradictory = aligned & (np.sign(jumps) != dominant_sign)
        else:
            dominant_sign = 0.0
            contradictory = np.zeros_like(aligned)
        contradictory_ratio = float(np.mean(contradictory))
        missing_ratio = missing / len(pairs)
        contrast_quality = float(
            min(1.0, np.median(magnitudes) / max(self.minimum_relative_jump, 1e-12))
        )
        quality = float(
            np.clip(
                source_quality
                * (valid_count / len(pairs))
                * contrast_quality
                * obj.reconstruction_quality,
                0.0,
                1.0,
            )
        )
        return ResidualEvidence.observed(
            "boundary_depth_3d",
            contradictory_ratio,
            quality=quality,
            source_ids=(object_id,),
            metadata={
                "evidence_role": EvidenceRole.ANOMALY_RESIDUAL.value,
                "boundary_source": boundary_source,
                "bbox_is_not_instance_contour": boundary_source != "instance_mask",
                "aligned_boundary_ratio": aligned_ratio,
                "missing_depth_boundary_ratio": missing_ratio,
                "contradictory_depth_ratio": contradictory_ratio,
                "mean_relative_depth_jump": float(np.mean(magnitudes)),
                "median_relative_depth_jump": float(np.median(magnitudes)),
                "dominant_depth_jump_direction": dominant_sign,
                "boundary_point_quality": quality,
                "sample_pair_count": len(pairs),
                "valid_pair_count": valid_count,
                "relative_depth_supported": True,
            },
        )
