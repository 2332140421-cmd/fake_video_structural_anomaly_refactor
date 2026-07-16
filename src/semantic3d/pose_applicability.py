"""Observation applicability gates for physical characteristic dimensions.

These gates decide whether an observation can represent a configured physical
dimension such as upright person height. They do not classify actions or
predict forgery. Scores are transparent heuristic quality scores, not
probabilities.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

import numpy as np

from .keypoint_provider import Keypoint2D, KeypointPrediction


@dataclass(frozen=True)
class ApplicabilityGateResult:
    """Result of checking whether a physical prior applies to one observation."""

    applicable: bool
    applicability_score: float
    applicability_status: str
    passed_checks: tuple[str, ...]
    failed_checks: tuple[str, ...]
    diagnostic_details: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class PersonHeightApplicabilityConfig:
    """Thresholds for applying an upright full-body person-height prior."""

    minimum_detection_confidence: float = 0.30
    minimum_keypoint_confidence: float = 0.35
    minimum_mean_keypoint_confidence: float = 0.45
    boundary_margin_ratio: float = 0.01
    maximum_torso_tilt_degrees: float = 35.0
    minimum_hip_to_ankle_span_ratio: float = 0.32
    maximum_leg_horizontal_offset_ratio: float = 0.30
    minimum_keypoint_bbox_span_ratio: float = 0.62
    maximum_keypoint_bbox_span_ratio: float = 1.20


@dataclass(frozen=True)
class CupHeightApplicabilityConfig:
    """Thresholds for a lightweight upright cup-height observation check."""

    minimum_detection_confidence: float = 0.30
    minimum_bbox_width: float = 12.0
    minimum_bbox_height: float = 12.0
    boundary_margin_ratio: float = 0.005
    minimum_aspect_ratio: float = 0.45
    maximum_aspect_ratio: float = 1.60
    maximum_projection_cv: float = 0.15
    maximum_relative_depth_iqr: float = 0.35
    minimum_valid_depth_ratio: float = 0.50


def _finite_bbox(bbox: Sequence[float]) -> Optional[tuple[float, float, float, float]]:
    if len(bbox) != 4:
        return None
    values = tuple(float(value) for value in bbox)
    if not all(math.isfinite(value) for value in values):
        return None
    x1, y1, x2, y2 = values
    return values if x2 > x1 and y2 > y1 else None


def _boundary_contact(
    bbox: tuple[float, float, float, float],
    frame_width: int,
    frame_height: int,
    margin_ratio: float,
) -> bool:
    x1, y1, x2, y2 = bbox
    margin_x = max(2.0, frame_width * margin_ratio)
    margin_y = max(2.0, frame_height * margin_ratio)
    return bool(
        x1 <= margin_x
        or y1 <= margin_y
        or x2 >= frame_width - margin_x
        or y2 >= frame_height - margin_y
    )


def _midpoint(points: Mapping[str, Keypoint2D], names: Sequence[str]) -> Optional[np.ndarray]:
    selected = [points[name] for name in names if name in points]
    if not selected:
        return None
    return np.mean([[point.x, point.y] for point in selected], axis=0)


def _torso_tilt_degrees(shoulders: np.ndarray, hips: np.ndarray) -> float:
    vector = hips - shoulders
    if float(np.linalg.norm(vector)) <= 1e-8:
        return math.nan
    return float(math.degrees(math.atan2(abs(float(vector[0])), abs(float(vector[1])))))


class PersonHeightApplicabilityGate:
    """Check whether keypoints support an upright complete-height measurement."""

    def __init__(self, config: Optional[PersonHeightApplicabilityConfig] = None) -> None:
        self.config = config or PersonHeightApplicabilityConfig()

    def evaluate(
        self,
        bbox: Sequence[float],
        frame_size: tuple[int, int],
        prediction: KeypointPrediction,
        detection_confidence: float,
        projection_history: Optional[Sequence[float]] = None,
    ) -> ApplicabilityGateResult:
        """Return whether a person bbox represents upright complete body height."""

        del projection_history  # Reserved for a future short-track stability check.
        width, height = (int(value) for value in frame_size)
        parsed_bbox = _finite_bbox(bbox)
        checks: dict[str, bool] = {
            "valid_bbox": parsed_bbox is not None and width > 0 and height > 0,
            "detection_confidence": math.isfinite(float(detection_confidence))
            and float(detection_confidence) >= self.config.minimum_detection_confidence,
        }
        if parsed_bbox is None or width <= 0 or height <= 0:
            return self._result("unresolved_pose", checks, {"reason": "invalid_bbox"})

        boundary = _boundary_contact(
            parsed_bbox, width, height, self.config.boundary_margin_ratio
        )
        checks["not_boundary_truncated"] = not boundary
        if boundary:
            return self._result(
                "boundary_truncated", checks, {"boundary_contact": True}
            )
        if prediction.status != "ok" or not prediction.keypoints:
            checks["required_keypoints_visible"] = False
            return self._result(
                "insufficient_keypoints",
                checks,
                {"provider_status": prediction.status},
            )

        all_points = {point.keypoint_name: point for point in prediction.keypoints}
        visible = {
            name: point
            for name, point in all_points.items()
            if point.valid
            and math.isfinite(point.confidence)
            and point.confidence >= self.config.minimum_keypoint_confidence
        }
        required_names = {
            "left_shoulder", "right_shoulder", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle",
        }
        available_required_confidences = [
            point.confidence for name, point in all_points.items() if name in required_names
        ]
        mean_required_confidence = (
            float(np.mean(available_required_confidences))
            if available_required_confidences
            else 0.0
        )
        checks["overall_keypoint_confidence"] = (
            mean_required_confidence >= self.config.minimum_mean_keypoint_confidence
        )
        if not checks["overall_keypoint_confidence"]:
            return self._result(
                "low_keypoint_confidence",
                checks,
                {"mean_required_keypoint_confidence": mean_required_confidence},
            )

        shoulder_visible = any(name in visible for name in ("left_shoulder", "right_shoulder"))
        hip_visible = any(name in visible for name in ("left_hip", "right_hip"))
        complete_sides = [
            side
            for side in ("left", "right")
            if all(f"{side}_{joint}" in visible for joint in ("hip", "knee", "ankle"))
        ]
        checks.update(
            {
                "upper_body_visible": shoulder_visible,
                "hip_visible": hip_visible,
                "knee_and_ankle_visible": bool(complete_sides),
            }
        )
        if not shoulder_visible or not hip_visible or not complete_sides:
            checks["required_keypoints_visible"] = False
            return self._result(
                "insufficient_keypoints",
                checks,
                {
                    "visible_keypoints": sorted(visible),
                    "complete_leg_sides": complete_sides,
                },
            )
        checks["required_keypoints_visible"] = True

        shoulders = _midpoint(visible, ("left_shoulder", "right_shoulder"))
        hips = _midpoint(visible, ("left_hip", "right_hip"))
        assert shoulders is not None and hips is not None
        torso_tilt = _torso_tilt_degrees(shoulders, hips)
        checks["torso_axis_near_vertical"] = (
            math.isfinite(torso_tilt)
            and torso_tilt <= self.config.maximum_torso_tilt_degrees
        )

        x1, y1, x2, y2 = parsed_bbox
        bbox_height = y2 - y1
        standing_sides: list[str] = []
        side_details: dict[str, object] = {}
        for side in complete_sides:
            hip = visible[f"{side}_hip"]
            knee = visible[f"{side}_knee"]
            ankle = visible[f"{side}_ankle"]
            ordered = hip.y < knee.y < ankle.y
            vertical_span = (ankle.y - hip.y) / bbox_height
            horizontal_offset = abs(ankle.x - hip.x) / bbox_height
            standing = bool(
                ordered
                and vertical_span >= self.config.minimum_hip_to_ankle_span_ratio
                and horizontal_offset <= self.config.maximum_leg_horizontal_offset_ratio
            )
            if standing:
                standing_sides.append(side)
            side_details[side] = {
                "ordered_hip_knee_ankle": ordered,
                "hip_to_ankle_span_ratio": vertical_span,
                "hip_to_ankle_horizontal_offset_ratio": horizontal_offset,
                "standing_geometry": standing,
            }
        checks["standing_leg_geometry"] = bool(standing_sides)
        if not checks["torso_axis_near_vertical"] or not standing_sides:
            return self._result(
                "sitting_or_bending",
                checks,
                {
                    "torso_tilt_degrees": torso_tilt,
                    "leg_geometry": side_details,
                },
            )

        valid_y = [point.y for point in visible.values()]
        keypoint_span_ratio = (max(valid_y) - min(valid_y)) / bbox_height
        checks["keypoint_span_matches_bbox"] = bool(
            self.config.minimum_keypoint_bbox_span_ratio
            <= keypoint_span_ratio
            <= self.config.maximum_keypoint_bbox_span_ratio
        )
        if not checks["keypoint_span_matches_bbox"]:
            return self._result(
                "incomplete_body",
                checks,
                {
                    "keypoint_bbox_span_ratio": keypoint_span_ratio,
                    "torso_tilt_degrees": torso_tilt,
                    "leg_geometry": side_details,
                },
            )

        return self._result(
            "applicable_upright_full_body",
            checks,
            {
                "torso_tilt_degrees": torso_tilt,
                "keypoint_bbox_span_ratio": keypoint_span_ratio,
                "mean_required_keypoint_confidence": mean_required_confidence,
                "standing_sides": standing_sides,
                "leg_geometry": side_details,
                "score_semantics": "heuristic_quality_not_probability",
            },
        )

    @staticmethod
    def _result(
        status: str,
        checks: Mapping[str, bool],
        details: Mapping[str, object],
    ) -> ApplicabilityGateResult:
        passed = tuple(name for name, value in checks.items() if value)
        failed = tuple(name for name, value in checks.items() if not value)
        score = len(passed) / len(checks) if checks else 0.0
        return ApplicabilityGateResult(
            status == "applicable_upright_full_body",
            float(score),
            status,
            passed,
            failed,
            dict(details),
        )


class CupHeightApplicabilityGate:
    """Check lightweight measurement quality for an upright cup-height prior."""

    def __init__(self, config: Optional[CupHeightApplicabilityConfig] = None) -> None:
        self.config = config or CupHeightApplicabilityConfig()

    def evaluate(
        self,
        bbox: Sequence[float],
        frame_size: tuple[int, int],
        detection_confidence: float,
        projection_history: Optional[Sequence[float]] = None,
        depth_iqr: float = math.nan,
        representative_depth: float = math.nan,
        valid_depth_ratio: float = math.nan,
    ) -> ApplicabilityGateResult:
        """Return whether a cup bbox can represent upright physical height."""

        width, height = (int(value) for value in frame_size)
        parsed_bbox = _finite_bbox(bbox)
        if parsed_bbox is None or width <= 0 or height <= 0:
            return self._result("pose_unresolved", {"valid_bbox": False}, {})
        x1, y1, x2, y2 = parsed_bbox
        bbox_width, bbox_height = x2 - x1, y2 - y1
        aspect_ratio = bbox_width / bbox_height
        checks = {
            "valid_bbox": True,
            "detection_confidence": math.isfinite(float(detection_confidence))
            and float(detection_confidence) >= self.config.minimum_detection_confidence,
            "minimum_bbox_size": bbox_width >= self.config.minimum_bbox_width
            and bbox_height >= self.config.minimum_bbox_height,
            "not_boundary_truncated": not _boundary_contact(
                parsed_bbox, width, height, self.config.boundary_margin_ratio
            ),
            "aspect_ratio": self.config.minimum_aspect_ratio
            <= aspect_ratio
            <= self.config.maximum_aspect_ratio,
        }
        history = np.asarray(
            [
                float(value)
                for value in (projection_history or ())
                if math.isfinite(float(value)) and float(value) > 0
            ],
            dtype=float,
        )
        projection_cv = (
            float(np.std(history) / np.mean(history)) if history.size >= 2 else math.nan
        )
        checks["stable_projection"] = bool(
            not math.isfinite(projection_cv)
            or projection_cv <= self.config.maximum_projection_cv
        )
        relative_depth_iqr = (
            float(depth_iqr) / float(representative_depth)
            if math.isfinite(float(depth_iqr))
            and math.isfinite(float(representative_depth))
            and float(representative_depth) > 0
            else math.nan
        )
        checks["stable_depth"] = bool(
            math.isfinite(relative_depth_iqr)
            and relative_depth_iqr <= self.config.maximum_relative_depth_iqr
            and math.isfinite(float(valid_depth_ratio))
            and float(valid_depth_ratio) >= self.config.minimum_valid_depth_ratio
        )
        details = {
            "bbox_width": bbox_width,
            "bbox_height": bbox_height,
            "aspect_ratio": aspect_ratio,
            "projection_cv": projection_cv,
            "depth_iqr": depth_iqr,
            "representative_depth": representative_depth,
            "relative_depth_iqr": relative_depth_iqr,
            "valid_depth_ratio": valid_depth_ratio,
            "score_semantics": "heuristic_quality_not_probability",
        }
        if not checks["minimum_bbox_size"]:
            return self._result("too_small", checks, details)
        if not checks["not_boundary_truncated"]:
            return self._result("boundary_truncated", checks, details)
        if not checks["stable_projection"]:
            return self._result("unstable_projection", checks, details)
        if not checks["stable_depth"]:
            return self._result("unstable_depth", checks, details)
        if not checks["detection_confidence"] or not checks["aspect_ratio"]:
            return self._result("pose_unresolved", checks, details)
        return self._result("applicable", checks, details)

    @staticmethod
    def _result(
        status: str,
        checks: Mapping[str, bool],
        details: Mapping[str, object],
    ) -> ApplicabilityGateResult:
        passed = tuple(name for name, value in checks.items() if value)
        failed = tuple(name for name, value in checks.items() if not value)
        score = len(passed) / len(checks) if checks else 0.0
        return ApplicabilityGateResult(
            status == "applicable",
            float(score),
            status,
            passed,
            failed,
            dict(details),
        )


def applicability_skip_reason(label: str, result: ApplicabilityGateResult) -> str:
    """Map one failed gate result to a stable R_sd skip reason."""

    normalized = label.strip().lower().replace(" ", "_")
    if result.applicable:
        return ""
    if normalized == "person":
        if result.applicability_status == "boundary_truncated":
            return "person_boundary_truncated"
        if result.applicability_status == "incomplete_body":
            return "person_incomplete_body"
        if result.applicability_status in {
            "insufficient_keypoints", "low_keypoint_confidence"
        }:
            return "person_insufficient_keypoints"
        return "person_pose_not_applicable"
    if normalized == "cup":
        if result.applicability_status == "too_small":
            return "cup_too_small"
        if result.applicability_status == "unstable_depth":
            return "cup_unstable_depth"
        return "cup_measurement_not_applicable"
    return "physical_prior_not_applicable"
