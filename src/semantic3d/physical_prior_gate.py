"""Generic heuristic observation-quality gate for conditional physical priors."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import numpy as np

from .observations import ObjectObservationJSON
from .projected_measurement import (
    ProjectedMeasurementResult,
    ProjectedMeasurementRules,
    compute_projected_measurement,
)


@dataclass(frozen=True)
class PhysicalPriorGateResult:
    """Heuristic gate result; gate_score is not a calibrated probability."""

    gate_passed: bool
    gate_score: float
    gate_reasons: tuple[str, ...]
    failed_gate_reasons: tuple[str, ...]


def _finite_positive(value: object) -> bool:
    """Return whether a scalar is finite and positive."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number > 0.0


def evaluate_physical_prior_gate(
    obj: ObjectObservationJSON,
    frame_width: int,
    frame_height: int,
    measurement: ProjectedMeasurementResult,
    gate_config: Mapping[str, object],
    measurement_rules: ProjectedMeasurementRules,
    history: Optional[Sequence[ObjectObservationJSON]] = None,
) -> PhysicalPriorGateResult:
    """Reject visibly unsuitable observations using model-independent geometry.

    Thresholds come from the frozen v2 configuration and are not calibrated on
    the pilot videos. The returned score is only the fraction of passed
    heuristic checks, never a probability or learned confidence.
    """

    passed: list[str] = []
    failed: list[str] = []

    def check(condition: bool, name: str) -> None:
        (passed if condition else failed).append(name)

    check(measurement.valid, "projected_measurement_valid")
    threshold = float(gate_config.get("confidence_threshold", 0.3))
    check(math.isfinite(float(obj.confidence)) and float(obj.confidence) >= threshold, "detection_confidence")
    check(_finite_positive(obj.depth), "valid_depth")

    bbox_valid = obj.bbox is not None and len(obj.bbox) == 4
    if bbox_valid:
        x1, y1, x2, y2 = (float(value) for value in obj.bbox or [])
        bbox_valid = all(math.isfinite(value) for value in (x1, y1, x2, y2)) and x2 > x1 and y2 > y1
    else:
        x1 = y1 = x2 = y2 = float("nan")
    check(bbox_valid, "bbox_valid")

    if bbox_valid:
        width, height = x2 - x1, y2 - y1
        minimum = float(gate_config.get("min_bbox_dimension_px", 8.0))
        check(width >= minimum and height >= minimum, "minimum_bbox_dimensions")
        aspect_ratio = width / height
        aspect_range = gate_config.get("aspect_ratio_range", [0.1, 10.0])
        low, high = (float(value) for value in aspect_range)  # type: ignore[arg-type]
        check(low <= aspect_ratio <= high, "aspect_ratio_range")
        margin_ratio = float(gate_config.get("border_margin_ratio", 0.005))
        margin_px = float(gate_config.get("border_margin_px", 2.0))
        margin_x = max(margin_px, margin_ratio * frame_width)
        margin_y = max(margin_px, margin_ratio * frame_height)
        check(
            x1 > margin_x and y1 > margin_y and x2 < frame_width - margin_x and y2 < frame_height - margin_y,
            "not_truncated_at_frame_boundary",
        )
    else:
        failed.extend(
            ["minimum_bbox_dimensions", "aspect_ratio_range", "not_truncated_at_frame_boundary"]
        )

    frame_area = float(frame_width * frame_height)
    area_ratio = float(obj.mask_area) / frame_area if frame_area > 0 else float("nan")
    min_area = float(gate_config.get("min_area_ratio", measurement_rules.min_area_ratio))
    max_area = float(gate_config.get("max_area_ratio", measurement_rules.max_area_ratio))
    check(math.isfinite(area_ratio) and min_area <= area_ratio <= max_area, "projected_area_ratio")

    stability_enabled = bool(gate_config.get("require_track_stability", False))
    if stability_enabled and history and len(history) >= 2:
        recent = list(history)[-int(gate_config.get("stability_window", 3)) :]
        aspects: list[float] = []
        projections: list[float] = []
        for item in recent:
            if item.bbox and len(item.bbox) == 4:
                bx1, by1, bx2, by2 = (float(value) for value in item.bbox)
                if bx2 > bx1 and by2 > by1:
                    aspects.append((bx2 - bx1) / (by2 - by1))
            projected = compute_projected_measurement(
                item,
                frame_width,
                frame_height,
                measurement.measurement_type,
                measurement_rules,
            )
            if projected.valid:
                projections.append(projected.value)
        max_relative_range = float(gate_config.get("max_track_relative_range", 0.35))

        def stable(values: Sequence[float]) -> bool:
            if len(values) < 2:
                return False
            median = float(np.median(values))
            return median > 0 and (max(values) - min(values)) / median <= max_relative_range

        check(stable(aspects), "stable_track_aspect_ratio")
        check(stable(projections), "stable_track_projected_measurement")

    total = len(passed) + len(failed)
    score = float(len(passed) / total) if total else 0.0
    return PhysicalPriorGateResult(
        gate_passed=not failed,
        gate_score=score,
        gate_reasons=tuple(passed),
        failed_gate_reasons=tuple(failed),
    )

