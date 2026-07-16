"""Dimension-aligned 2D projected measurements for strict physical R_sd."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Union

import yaml

from .observations import ObjectObservationJSON


PathLike = Union[str, Path]


@dataclass(frozen=True)
class ProjectedMeasurementRule:
    """Configuration for one normalized projected linear measurement."""

    measurement_type: str
    compatibility_group: str
    formula: str
    required_geometry: str


@dataclass(frozen=True)
class ProjectedMeasurementResult:
    """One projected measurement with explicit validity and quality metadata."""

    value: float
    measurement_type: str
    compatibility_group: str
    measurement_quality: str
    invalid_reason: str = ""

    @property
    def valid(self) -> bool:
        """Return whether the projected measurement is finite and positive."""

        return not self.invalid_reason and math.isfinite(self.value) and self.value > 0.0


@dataclass(frozen=True)
class ProjectedMeasurementRules:
    """Validated projected-measurement rules loaded from YAML."""

    rules: Mapping[str, ProjectedMeasurementRule]
    min_bbox_width_px: float
    min_bbox_height_px: float
    min_area_ratio: float
    max_area_ratio: float
    allow_same_group_only: bool = True
    allow_cross_axis_conversion: bool = False

    def rule(self, measurement_type: str) -> ProjectedMeasurementRule:
        """Return one configured rule or raise a clear error."""

        if measurement_type not in self.rules:
            raise ValueError(f"Unknown projected measurement type: {measurement_type!r}.")
        return self.rules[measurement_type]


def load_projected_measurement_rules(path: PathLike) -> ProjectedMeasurementRules:
    """Load projected measurement and compatibility rules from YAML."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Projected measurement config must be a mapping: {config_path}")
    raw_types = data.get("measurement_types")
    validation = data.get("global_validation")
    compatibility = data.get("compatibility_policy", {})
    if not isinstance(raw_types, dict) or not isinstance(validation, dict):
        raise ValueError("Projected measurement config requires measurement_types and global_validation.")
    rules = {
        str(name): ProjectedMeasurementRule(
            measurement_type=str(name),
            compatibility_group=str(raw["compatibility_group"]),
            formula=str(raw["formula"]),
            required_geometry=str(raw["required_geometry"]),
        )
        for name, raw in raw_types.items()
    }
    return ProjectedMeasurementRules(
        rules=rules,
        min_bbox_width_px=float(validation.get("min_bbox_width_px", 8.0)),
        min_bbox_height_px=float(validation.get("min_bbox_height_px", 8.0)),
        min_area_ratio=float(validation.get("min_area_ratio", 0.0001)),
        max_area_ratio=float(validation.get("max_area_ratio", 0.98)),
        allow_same_group_only=bool(compatibility.get("allow_same_group_only", True)),
        allow_cross_axis_conversion=bool(compatibility.get("allow_cross_axis_conversion", False)),
    )


def _invalid(measurement_type: str, group: str, reason: str) -> ProjectedMeasurementResult:
    """Build an invalid result without substituting a zero measurement."""

    return ProjectedMeasurementResult(
        value=float("nan"),
        measurement_type=measurement_type,
        compatibility_group=group,
        measurement_quality="invalid",
        invalid_reason=reason,
    )


def _validated_bbox(
    obj: ObjectObservationJSON,
    frame_width: int,
    frame_height: int,
) -> tuple[Optional[tuple[float, float, float, float]], str]:
    """Validate an unclipped bbox so truncation remains visible to the gate."""

    if frame_width <= 0 or frame_height <= 0:
        return None, "invalid_frame_size"
    if obj.bbox is None or len(obj.bbox) != 4:
        return None, "missing_bbox"
    try:
        x1, y1, x2, y2 = (float(value) for value in obj.bbox)
    except (TypeError, ValueError):
        return None, "non_numeric_bbox"
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
        return None, "non_finite_bbox"
    if x1 < 0 or y1 < 0 or x2 > frame_width or y2 > frame_height:
        return None, "bbox_out_of_bounds"
    if x2 <= x1 or y2 <= y1:
        return None, "non_positive_bbox_extent"
    return (x1, y1, x2, y2), ""


def compute_projected_measurement(
    obj: ObjectObservationJSON,
    frame_width: int,
    frame_height: int,
    measurement_type: str,
    rules: ProjectedMeasurementRules,
) -> ProjectedMeasurementResult:
    """Compute one dimension-aligned normalized projected measurement.

    No bbox clipping is performed. Out-of-frame or very small detections are
    invalid evidence rather than silently repaired observations.
    """

    rule = rules.rule(measurement_type)
    bbox, reason = _validated_bbox(obj, frame_width, frame_height)
    if bbox is None:
        return _invalid(measurement_type, rule.compatibility_group, reason)
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    if width < rules.min_bbox_width_px or height < rules.min_bbox_height_px:
        return _invalid(measurement_type, rule.compatibility_group, "bbox_too_small")

    frame_area = float(frame_width * frame_height)
    if not math.isfinite(float(obj.mask_area)) or float(obj.mask_area) <= 0.0:
        return _invalid(measurement_type, rule.compatibility_group, "non_positive_projected_area")
    area_ratio = float(obj.mask_area) / frame_area
    if area_ratio < rules.min_area_ratio or area_ratio > rules.max_area_ratio:
        return _invalid(measurement_type, rule.compatibility_group, "projected_area_ratio_out_of_range")

    if measurement_type == "bbox_height_norm":
        value = height / float(frame_height)
        quality = "bbox_extent"
    elif measurement_type == "bbox_width_norm":
        value = width / float(frame_width)
        quality = "bbox_extent"
    elif measurement_type == "bbox_diagonal_norm":
        value = math.hypot(width / float(frame_width), height / float(frame_height))
        quality = "bbox_diagonal"
    elif measurement_type == "equivalent_diameter_norm":
        value = math.sqrt(4.0 * float(obj.mask_area) / math.pi) / math.sqrt(frame_area)
        quality = "mask_area" if obj.mask_path else "bbox_area_approximation"
    else:  # guarded by rules.rule; retained for defensive clarity
        return _invalid(measurement_type, rule.compatibility_group, "unsupported_measurement_type")
    if not math.isfinite(value) or value <= 0.0:
        return _invalid(measurement_type, rule.compatibility_group, "invalid_measurement_value")
    return ProjectedMeasurementResult(
        value=value,
        measurement_type=measurement_type,
        compatibility_group=rule.compatibility_group,
        measurement_quality=quality,
    )

