"""Camera-motion regime diagnostics from foreground-filtered background tracks.

The classes in this module describe geometric operating conditions.  They are
not forged-video anomaly labels and do not use semantic physical-size priors.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping


class CameraMotionRegime(str, Enum):
    """Coarse motion regime used to select an appropriate pose model."""

    STATIC_CAMERA = "static_camera"
    ROTATION_DOMINANT = "rotation_dominant"
    GENERAL_SE3 = "general_se3"
    INSUFFICIENT_BACKGROUND = "insufficient_background"
    LOW_TEXTURE = "low_texture"
    DEGENERATE_GEOMETRY = "degenerate_geometry"
    SCENE_CUT = "scene_cut"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class MotionRegimeThresholds:
    """Reproducible engineering thresholds for regime classification."""

    minimum_background_points: int = 20
    minimum_matches: int = 12
    static_median_flow_px: float = 0.35
    static_p90_flow_px: float = 0.80
    rotation_max_parallax_px: float = 1.25
    homography_inlier_ratio: float = 0.65
    essential_inlier_ratio: float = 0.45


@dataclass(frozen=True)
class MotionRegimeObservation:
    """Measured background evidence and its camera-motion interpretation."""

    source_frame_index: int
    target_frame_index: int
    regime: CameraMotionRegime | str
    background_candidate_count: int
    background_point_count: int
    background_match_count: int
    background_inlier_count: int
    background_inlier_ratio: float
    median_background_flow: float
    p90_background_flow: float
    median_parallax: float
    homography_inlier_ratio: float
    essential_matrix_inlier_ratio: float
    model_reprojection_error: float
    image_difference: float
    foreground_excluded_ratio: float
    spatial_coverage_ratio: float
    quadrant_support: tuple[int, int, int, int]
    feature_concentrated: bool
    evidence_source: str
    valid: bool
    missing_reason: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        regime = CameraMotionRegime(self.regime)
        counts = (
            self.background_candidate_count,
            self.background_point_count,
            self.background_match_count,
            self.background_inlier_count,
        )
        if any(value < 0 for value in counts):
            raise ValueError("Motion-regime support counts must be non-negative.")
        if len(self.quadrant_support) != 4 or any(
            value < 0 for value in self.quadrant_support
        ):
            raise ValueError("quadrant_support must contain four non-negative counts.")
        for name in (
            "background_inlier_ratio",
            "homography_inlier_ratio",
            "essential_matrix_inlier_ratio",
            "image_difference",
            "foreground_excluded_ratio",
            "spatial_coverage_ratio",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1].")
            object.__setattr__(self, name, value)
        for name in (
            "median_background_flow",
            "p90_background_flow",
            "median_parallax",
            "model_reprojection_error",
        ):
            value = float(getattr(self, name))
            if not (math.isnan(value) or (math.isfinite(value) and value >= 0.0)):
                raise ValueError(f"{name} must be non-negative or NaN.")
            object.__setattr__(self, name, value)
        if self.valid and self.missing_reason:
            raise ValueError("A valid motion regime cannot have missing_reason.")
        if not self.valid and not self.missing_reason:
            raise ValueError("An invalid motion regime requires missing_reason.")
        object.__setattr__(self, "regime", regime)
        object.__setattr__(self, "quadrant_support", tuple(self.quadrant_support))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def supports_identity_pose(self) -> bool:
        """Return whether measured evidence validates a static identity edge."""

        return bool(
            self.valid
            and self.regime == CameraMotionRegime.STATIC_CAMERA
            and self.background_match_count > 0
            and self.evidence_source
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a serialization-friendly diagnostic record."""

        payload = asdict(self)
        payload["regime"] = self.regime.value
        payload["quadrant_support"] = list(self.quadrant_support)
        return payload


def classify_motion_regime(
    *,
    source_frame_index: int,
    target_frame_index: int,
    background_candidate_count: int,
    background_point_count: int,
    background_match_count: int,
    background_inlier_count: int,
    median_background_flow: float,
    p90_background_flow: float,
    median_parallax: float,
    homography_inlier_ratio: float,
    essential_matrix_inlier_ratio: float,
    model_reprojection_error: float,
    image_difference: float,
    foreground_excluded_ratio: float,
    spatial_coverage_ratio: float,
    quadrant_support: tuple[int, int, int, int],
    feature_concentrated: bool,
    scene_cut: bool = False,
    thresholds: MotionRegimeThresholds | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> MotionRegimeObservation:
    """Classify camera motion without treating low parallax as automatic failure."""

    limits = thresholds or MotionRegimeThresholds()
    match_ratio = (
        background_inlier_count / background_match_count
        if background_match_count
        else 0.0
    )
    common = dict(
        source_frame_index=source_frame_index,
        target_frame_index=target_frame_index,
        background_candidate_count=background_candidate_count,
        background_point_count=background_point_count,
        background_match_count=background_match_count,
        background_inlier_count=background_inlier_count,
        background_inlier_ratio=match_ratio,
        median_background_flow=median_background_flow,
        p90_background_flow=p90_background_flow,
        median_parallax=median_parallax,
        homography_inlier_ratio=homography_inlier_ratio,
        essential_matrix_inlier_ratio=essential_matrix_inlier_ratio,
        model_reprojection_error=model_reprojection_error,
        image_difference=image_difference,
        foreground_excluded_ratio=foreground_excluded_ratio,
        spatial_coverage_ratio=spatial_coverage_ratio,
        quadrant_support=quadrant_support,
        feature_concentrated=feature_concentrated,
        metadata=dict(metadata or {}),
    )
    if scene_cut:
        return MotionRegimeObservation(
            regime=CameraMotionRegime.SCENE_CUT,
            evidence_source="scene_cut_detector",
            valid=False,
            missing_reason="scene_cut_boundary",
            **common,
        )
    if background_candidate_count < limits.minimum_background_points:
        return MotionRegimeObservation(
            regime=CameraMotionRegime.LOW_TEXTURE,
            evidence_source="background_feature_detection",
            valid=False,
            missing_reason="low_texture_before_foreground_filter",
            **common,
        )
    if background_point_count < limits.minimum_background_points:
        return MotionRegimeObservation(
            regime=CameraMotionRegime.INSUFFICIENT_BACKGROUND,
            evidence_source="foreground_filtered_background",
            valid=False,
            missing_reason="insufficient_background_after_foreground_filter",
            **common,
        )
    if background_match_count < limits.minimum_matches:
        return MotionRegimeObservation(
            regime=CameraMotionRegime.INSUFFICIENT_BACKGROUND,
            evidence_source="forward_backward_lk_matching",
            valid=False,
            missing_reason="insufficient_background_matches",
            **common,
        )
    if (
        math.isfinite(median_background_flow)
        and math.isfinite(p90_background_flow)
        and median_background_flow <= limits.static_median_flow_px
        and p90_background_flow <= limits.static_p90_flow_px
    ):
        return MotionRegimeObservation(
            regime=CameraMotionRegime.STATIC_CAMERA,
            evidence_source="foreground_filtered_lk_near_zero_flow",
            valid=True,
            **common,
        )
    if (
        homography_inlier_ratio >= limits.homography_inlier_ratio
        and (
            not math.isfinite(median_parallax)
            or median_parallax <= limits.rotation_max_parallax_px
            or essential_matrix_inlier_ratio < limits.essential_inlier_ratio
        )
    ):
        return MotionRegimeObservation(
            regime=CameraMotionRegime.ROTATION_DOMINANT,
            evidence_source="background_homography_dominant",
            valid=True,
            **common,
        )
    if essential_matrix_inlier_ratio >= limits.essential_inlier_ratio:
        return MotionRegimeObservation(
            regime=CameraMotionRegime.GENERAL_SE3,
            evidence_source="background_essential_geometry",
            valid=True,
            **common,
        )
    return MotionRegimeObservation(
        regime=CameraMotionRegime.DEGENERATE_GEOMETRY,
        evidence_source="competing_background_geometry_models",
        valid=False,
        missing_reason=(
            "spatially_concentrated_background_features"
            if feature_concentrated
            else "insufficient_geometric_inliers"
        ),
        **common,
    )

