"""Reconstruction quality evidence used to gate, not score, anomalies."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from ..shared_3d_observation import Object3DObservation, Shared3DFrameObservation
from ..validity import ResidualEvidence
from .residual_types import EvidenceRole, Static3DContext


def _valid_ratio(points: tuple) -> float:
    if not points:
        return math.nan
    return float(np.mean([bool(point.valid) for point in points]))


class ReconstructionQualityEvidence:
    """Summarize whether a reconstructed object can support later residuals."""

    def __init__(self, minimum_quality: float = 0.20) -> None:
        if not 0.0 <= minimum_quality <= 1.0:
            raise ValueError("minimum_quality must be in [0, 1].")
        self.minimum_quality = float(minimum_quality)

    def evaluate(
        self,
        frame: Shared3DFrameObservation,
        object_id: str,
        *,
        reprojection_cycle_error: Optional[float] = None,
    ) -> ResidualEvidence:
        """Return a heuristic quality score, explicitly not a probability."""

        try:
            obj = Static3DContext(frame).object_by_id(object_id)
        except KeyError:
            return ResidualEvidence.missing(
                "reconstruction_quality", "object_not_found", source_ids=(object_id,)
            )
        if not obj.valid:
            return ResidualEvidence.missing(
                "reconstruction_quality",
                obj.missing_reason or "invalid_object_reconstruction",
                source_ids=(object_id,),
            )

        valid_3d_ratio = float(obj.metadata.get("valid_point_ratio", math.nan))
        keypoint_ratio = _valid_ratio(obj.keypoints_3d)
        boundary_ratio = _valid_ratio(obj.boundary_points_3d)
        valid_points = [point for point in obj.structure_points_3d if point.valid]
        point_qualities = [float(point.metadata.get("point_quality", point.confidence)) for point in valid_points]
        depth_iqrs = [
            float(point.metadata["local_depth_iqr"])
            for point in valid_points
            if point.metadata.get("local_depth_iqr") is not None
            and math.isfinite(float(point.metadata["local_depth_iqr"]))
        ]
        mean_point_quality = float(np.mean(point_qualities)) if point_qualities else 0.0
        mean_depth_iqr = float(np.mean(depth_iqrs)) if depth_iqrs else math.nan
        median_depth = float(np.median([point.z for point in valid_points])) if valid_points else math.nan
        relative_iqr = (
            mean_depth_iqr / max(abs(median_depth), 1e-12)
            if math.isfinite(mean_depth_iqr) and math.isfinite(median_depth)
            else math.nan
        )
        depth_stability = 1.0 / (1.0 + relative_iqr) if math.isfinite(relative_iqr) else 0.5
        intrinsics_quality = (
            frame.camera.quality
            if frame.camera.intrinsics_source.strip().lower() == "approximate"
            else 1.0
        )
        cycle_quality = (
            1.0 / (1.0 + float(reprojection_cycle_error))
            if reprojection_cycle_error is not None
            and math.isfinite(float(reprojection_cycle_error))
            and reprojection_cycle_error >= 0.0
            else math.nan
        )
        components = [
            value
            for value in (
                valid_3d_ratio,
                keypoint_ratio,
                boundary_ratio,
                mean_point_quality,
                depth_stability,
                intrinsics_quality,
                cycle_quality,
            )
            if math.isfinite(value)
        ]
        quality = float(np.clip(np.mean(components), 0.0, 1.0)) if components else 0.0
        return ResidualEvidence.observed(
            "reconstruction_quality",
            quality,
            quality=quality,
            source_ids=(object_id,),
            metadata={
                "evidence_role": EvidenceRole.QUALITY.value,
                "quality_is_probability": False,
                "valid_3d_point_ratio": valid_3d_ratio,
                "keypoint_valid_ratio": keypoint_ratio,
                "boundary_point_valid_ratio": boundary_ratio,
                "mean_point_quality": mean_point_quality,
                "depth_local_iqr": mean_depth_iqr,
                "relative_depth_iqr": relative_iqr,
                "approximate_intrinsics_quality": intrinsics_quality,
                "reprojection_cycle_error": reprojection_cycle_error,
                "scale_status": obj.scale_status.value,
                "depth_scale_status": obj.depth_scale_status.value,
                "passes_quality_gate": quality >= self.minimum_quality,
                "minimum_quality": self.minimum_quality,
            },
        )
