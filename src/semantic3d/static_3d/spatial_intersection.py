"""Low-confidence AABB and sparse-point intersection diagnostics."""

from __future__ import annotations

import math

import numpy as np

from ..shared_3d_observation import Object3DObservation, Shared3DFrameObservation
from ..validity import ResidualEvidence
from .residual_types import EvidenceRole, Static3DContext


def _valid_xyz(obj: Object3DObservation) -> np.ndarray:
    points = [point.as_array() for point in obj.structure_points_3d if point.valid]
    return np.stack(points) if points else np.empty((0, 3), dtype=float)


class SpatialIntersection3DResidual:
    """Approximate overlap evidence that must not be called definite penetration."""

    def __init__(self, minimum_points: int = 4, default_weight: float = 0.10) -> None:
        if minimum_points < 2:
            raise ValueError("minimum_points must be at least 2.")
        if not 0.0 <= default_weight <= 1.0:
            raise ValueError("default_weight must be in [0, 1].")
        self.minimum_points = int(minimum_points)
        self.default_weight = float(default_weight)

    def evaluate(
        self,
        frame: Shared3DFrameObservation,
        object_a_id: str,
        object_b_id: str,
    ) -> ResidualEvidence:
        """Return AABB/sparse proximity diagnostic evidence from one shared frame."""

        try:
            context = Static3DContext(frame)
            object_a = context.object_by_id(object_a_id)
            object_b = context.object_by_id(object_b_id)
        except KeyError:
            return ResidualEvidence.missing(
                "spatial_intersection_3d",
                "object_not_found",
                source_ids=(object_a_id, object_b_id),
            )
        points_a, points_b = _valid_xyz(object_a), _valid_xyz(object_b)
        if (
            not object_a.valid
            or not object_b.valid
            or points_a.shape[0] < self.minimum_points
            or points_b.shape[0] < self.minimum_points
            or min(object_a.reconstruction_quality, object_b.reconstruction_quality) <= 0.0
        ):
            return ResidualEvidence.missing(
                "spatial_intersection_3d",
                "insufficient_3d_spatial_evidence",
                source_ids=(object_a_id, object_b_id),
                metadata={
                    "points_a": int(points_a.shape[0]),
                    "points_b": int(points_b.shape[0]),
                },
            )
        if object_a.scale_status != object_b.scale_status:
            return ResidualEvidence.missing(
                "spatial_intersection_3d",
                "incompatible_scale_domains",
                source_ids=(object_a_id, object_b_id),
            )

        min_a, max_a = np.min(points_a, axis=0), np.max(points_a, axis=0)
        min_b, max_b = np.min(points_b, axis=0), np.max(points_b, axis=0)
        overlap_extent = np.maximum(0.0, np.minimum(max_a, max_b) - np.maximum(min_a, min_b))
        volume_a = float(np.prod(np.maximum(0.0, max_a - min_a)))
        volume_b = float(np.prod(np.maximum(0.0, max_b - min_b)))
        overlap_volume = float(np.prod(overlap_extent))
        denominator = min(volume_a, volume_b)
        overlap_ratio = overlap_volume / denominator if denominator > 1e-12 else 0.0
        center_a = object_a.center_3d_camera
        center_b = object_b.center_3d_camera
        center_a_inside_b = bool(
            center_a is not None
            and center_a.valid
            and np.all(center_a.as_array() >= min_b)
            and np.all(center_a.as_array() <= max_b)
        )
        center_b_inside_a = bool(
            center_b is not None
            and center_b.valid
            and np.all(center_b.as_array() >= min_a)
            and np.all(center_b.as_array() <= max_a)
        )
        pairwise_distances = np.linalg.norm(
            points_a[:, None, :] - points_b[None, :, :], axis=2
        )
        minimum_sparse_distance = float(np.min(pairwise_distances))
        scale_normalizer = max(
            float(object_a.observed_scale_3d or 0.0),
            float(object_b.observed_scale_3d or 0.0),
            1e-12,
        )
        normalized_proximity = minimum_sparse_distance / scale_normalizer
        diagnostic_value = float(
            np.clip(
                max(
                    overlap_ratio,
                    1.0 - min(1.0, normalized_proximity),
                    1.0 if center_a_inside_b or center_b_inside_a else 0.0,
                ),
                0.0,
                1.0,
            )
        )
        quality = float(
            min(object_a.reconstruction_quality, object_b.reconstruction_quality)
            * 0.5
        )
        return ResidualEvidence.observed(
            "spatial_intersection_3d",
            diagnostic_value,
            quality=quality,
            source_ids=(object_a_id, object_b_id),
            metadata={
                "evidence_role": EvidenceRole.DIAGNOSTIC.value,
                "approximation": "aabb_or_sparse_points",
                "definite_physical_penetration": False,
                "recommended_weight": self.default_weight,
                "aabb_overlap_extent": overlap_extent.tolist(),
                "overlap_volume": overlap_volume,
                "overlap_volume_ratio": overlap_ratio,
                "center_a_inside_b_extent": center_a_inside_b,
                "center_b_inside_a_extent": center_b_inside_a,
                "minimum_sparse_point_distance": minimum_sparse_distance,
                "normalized_sparse_proximity": normalized_proximity,
                "scale_status": object_a.scale_status.value,
            },
        )
