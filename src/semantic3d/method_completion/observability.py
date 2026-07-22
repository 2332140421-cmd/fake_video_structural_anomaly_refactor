"""Clip-level motion observability classification, not anomaly inference."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional


class ClipObservability(str, Enum):
    """Mutually exclusive operating conditions for one clip."""

    STATIC = "static"
    LOW_MOTION = "low_motion"
    OBJECT_MOTION = "object_motion"
    CAMERA_MOTION = "camera_motion"
    MIXED_MOTION = "mixed_motion"
    MOTION_UNRELIABLE = "motion_unreliable"


@dataclass(frozen=True)
class ObservabilityThresholds:
    """Engineering thresholds for classifying measured image motion."""

    static_motion_px: float = 0.35
    low_motion_px: float = 1.0
    object_motion_px: float = 2.0
    camera_motion_px: float = 2.0
    minimum_quality: float = 0.2

    def __post_init__(self) -> None:
        values = (
            self.static_motion_px,
            self.low_motion_px,
            self.object_motion_px,
            self.camera_motion_px,
            self.minimum_quality,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("Observability thresholds must be finite and non-negative.")


@dataclass(frozen=True)
class ClipMotionMeasurements:
    """Motion measurements kept separate from their interpretation."""

    clip_id: str
    background_motion_px: Optional[float]
    object_motion_px: Optional[float]
    camera_pose_available: bool
    object_tracks_available: bool
    quality: float
    provider_failed: bool = False
    scene_cut: bool = False
    missing_reason: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        quality = float(self.quality)
        if not math.isfinite(quality) or not 0.0 <= quality <= 1.0:
            raise ValueError("quality must be finite and in [0, 1].")
        for name in ("background_motion_px", "object_motion_px"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(float(value)) or float(value) < 0.0):
                raise ValueError(f"{name} must be non-negative and finite when present.")
        if self.provider_failed and not self.missing_reason:
            raise ValueError("provider_failed measurements require missing_reason.")
        object.__setattr__(self, "quality", quality)
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class ClipObservabilityResult:
    """Classification result with explicit reliability and diagnostics."""

    clip_id: str
    observability: ClipObservability | str
    valid: bool
    quality: float
    provider_failed: bool
    missing_reason: str = ""
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        status = ClipObservability(self.observability)
        if self.valid and status == ClipObservability.MOTION_UNRELIABLE:
            raise ValueError("motion_unreliable cannot be a valid classification.")
        if self.valid and (self.provider_failed or self.missing_reason):
            raise ValueError("Valid observability cannot be a provider failure.")
        if not self.valid and not self.missing_reason:
            raise ValueError("Invalid observability requires missing_reason.")
        object.__setattr__(self, "observability", status)
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))


def classify_clip_observability(
    measurements: ClipMotionMeasurements,
    thresholds: ObservabilityThresholds | None = None,
) -> ClipObservabilityResult:
    """Classify motion without treating static or low motion as failure."""

    limits = thresholds or ObservabilityThresholds()
    diagnostics = {
        "background_motion_px": measurements.background_motion_px,
        "object_motion_px": measurements.object_motion_px,
        "camera_pose_available": measurements.camera_pose_available,
        "object_tracks_available": measurements.object_tracks_available,
        "scene_cut": measurements.scene_cut,
    }
    if measurements.provider_failed:
        return ClipObservabilityResult(
            measurements.clip_id,
            ClipObservability.MOTION_UNRELIABLE,
            False,
            0.0,
            True,
            measurements.missing_reason,
            diagnostics,
        )
    if measurements.scene_cut:
        return ClipObservabilityResult(
            measurements.clip_id,
            ClipObservability.MOTION_UNRELIABLE,
            False,
            measurements.quality,
            False,
            "scene_cut_breaks_motion_observation",
            diagnostics,
        )
    if measurements.quality < limits.minimum_quality:
        return ClipObservabilityResult(
            measurements.clip_id,
            ClipObservability.MOTION_UNRELIABLE,
            False,
            measurements.quality,
            False,
            "insufficient_motion_observation_quality",
            diagnostics,
        )
    background = measurements.background_motion_px
    objects = measurements.object_motion_px
    if background is None and objects is None:
        return ClipObservabilityResult(
            measurements.clip_id,
            ClipObservability.MOTION_UNRELIABLE,
            False,
            measurements.quality,
            False,
            "motion_measurements_unavailable",
            diagnostics,
        )
    background_value = 0.0 if background is None else float(background)
    object_value = 0.0 if objects is None else float(objects)
    if (
        background is not None
        and objects is not None
        and background_value <= limits.static_motion_px
        and object_value <= limits.static_motion_px
    ):
        status = ClipObservability.STATIC
    elif (
        background_value <= limits.low_motion_px
        and object_value <= limits.low_motion_px
    ):
        status = ClipObservability.LOW_MOTION
    elif (
        background_value <= limits.low_motion_px
        and object_value >= limits.object_motion_px
    ):
        status = ClipObservability.OBJECT_MOTION
    elif (
        background_value >= limits.camera_motion_px
        and object_value < limits.object_motion_px
    ):
        status = ClipObservability.CAMERA_MOTION
    elif (
        background_value >= limits.camera_motion_px
        and object_value >= limits.object_motion_px
    ):
        status = ClipObservability.MIXED_MOTION
    else:
        status = ClipObservability.LOW_MOTION
    return ClipObservabilityResult(
        measurements.clip_id,
        status,
        True,
        measurements.quality,
        False,
        "",
        diagnostics,
    )
