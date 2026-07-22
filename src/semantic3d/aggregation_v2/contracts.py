"""Traceable point-to-clip aggregate contracts for partial P4 experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class _EvidenceAggregate:
    value: float
    valid: bool
    quality: float
    coverage: float
    missing_reason: str
    contributing_source_ids: tuple[str, ...]
    contributing_branch_names: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    applicable_count: int = 0
    valid_count: int = 0
    observation_missing_count: int = 0
    not_applicable_count: int = 0
    invalid_geometry_count: int = 0
    unsupported_mode_count: int = 0
    top_contributors: tuple[Mapping[str, Any], ...] = ()
    coverage_dimensions: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        value, quality, coverage = float(self.value), float(self.quality), float(self.coverage)
        if not 0.0 <= quality <= 1.0 or not 0.0 <= coverage <= 1.0:
            raise ValueError("Aggregate quality and coverage must be in [0, 1].")
        if self.valid:
            if not math.isfinite(value) or self.missing_reason:
                raise ValueError("Valid aggregate requires a finite value and no missing reason.")
        elif not math.isnan(value) or not self.missing_reason:
            raise ValueError("Invalid aggregate requires NaN and a missing reason.")
        counts = (
            self.applicable_count, self.valid_count,
            self.observation_missing_count, self.not_applicable_count,
            self.invalid_geometry_count, self.unsupported_mode_count,
        )
        if any(int(count) < 0 for count in counts):
            raise ValueError("Aggregate evidence counts must be non-negative.")
        dimensions = {str(key): float(item) for key, item in self.coverage_dimensions.items()}
        if any(not 0.0 <= item <= 1.0 for item in dimensions.values()):
            raise ValueError("Coverage dimensions must be in [0, 1].")
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "quality", quality)
        object.__setattr__(self, "coverage", coverage)
        object.__setattr__(self, "contributing_source_ids", tuple(dict.fromkeys(self.contributing_source_ids)))
        object.__setattr__(self, "contributing_branch_names", tuple(dict.fromkeys(self.contributing_branch_names)))
        object.__setattr__(self, "top_contributors", tuple(dict(item) for item in self.top_contributors))
        object.__setattr__(self, "coverage_dimensions", dimensions)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def anomaly_value(self) -> float:
        return self.value

    @property
    def evidence_quality(self) -> float:
        return self.quality

    @property
    def evidence_coverage(self) -> float:
        return self.coverage


@dataclass(frozen=True)
class PointEvidenceAggregate(_EvidenceAggregate):
    """Aggregate retaining one point and its object/frame identity."""

    point_id: str = ""
    object_track_id: str = ""
    video_id: str = ""
    frame_index: int = -1


@dataclass(frozen=True)
class EdgeEvidenceAggregate(_EvidenceAggregate):
    """Aggregate retaining one fixed structure-edge identity."""

    edge_id: str = ""
    object_track_id: str = ""
    video_id: str = ""
    frame_index: int = -1


@dataclass(frozen=True)
class ObjectEvidenceAggregate(_EvidenceAggregate):
    """Object-frame score with mask-aware spatial localization."""

    object_track_id: str = ""
    semantic_label: str = ""
    video_id: str = ""
    frame_index: int = -1
    branch_scores: Mapping[str, float] = field(default_factory=dict)
    valid_point_ratio: float = float("nan")
    valid_edge_ratio: float = float("nan")
    top_anomalous_point_ids: tuple[str, ...] = ()
    top_anomalous_edge_ids: tuple[str, ...] = ()
    localization_bbox: tuple[float, float, float, float] | None = None
    localization_mask_reference: str = ""

    @property
    def object_score(self) -> float:
        return self.value


@dataclass(frozen=True)
class FrameEvidenceAggregate(_EvidenceAggregate):
    """Frame score retaining object, point, edge and branch localization."""

    video_id: str = ""
    frame_index: int = -1
    object_scores: Mapping[str, float] = field(default_factory=dict)
    active_branches: tuple[str, ...] = ()
    branch_coverage: Mapping[str, float] = field(default_factory=dict)
    top_object_ids: tuple[str, ...] = ()
    top_point_ids: tuple[str, ...] = ()
    top_edge_ids: tuple[str, ...] = ()

    @property
    def frame_score(self) -> float:
        return self.value


@dataclass(frozen=True)
class ClipEvidenceAggregate(_EvidenceAggregate):
    """Clip localization aggregate; it never encodes a real/fake decision."""

    video_id: str = ""
    clip_id: str = ""
    frame_score_sequence: tuple[float, ...] = ()
    frame_indices: tuple[int, ...] = ()
    peak_score: float = float("nan")
    top_k_frame_mean: float = float("nan")
    persistent_interval_score: float = float("nan")
    candidate_intervals: tuple[Mapping[str, Any], ...] = ()
    valid_frame_ratio: float = 0.0
    branch_coverage: Mapping[str, float] = field(default_factory=dict)
    top_objects: tuple[str, ...] = ()
    top_spatial_regions: tuple[str, ...] = ()
