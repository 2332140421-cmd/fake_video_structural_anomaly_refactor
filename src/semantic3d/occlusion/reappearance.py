"""Conservative object reappearance identity and consistency evidence."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from ..validity import ResidualEvidence


@dataclass(frozen=True)
class ReappearanceObservation:
    """Candidate link from a pre-occlusion track to a post-occlusion object."""

    previous_object_track_id: str
    candidate_object_track_id: str
    frame_index: int
    predicted_reappearance_region: Optional[tuple[float, float, float, float]]
    semantic_label_match: bool
    appearance_similarity: float
    structure_similarity: float
    relative_depth_consistency: float
    motion_direction_consistency: float
    reid_source: str
    quality: float
    valid: bool
    missing_reason: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        quality = float(self.quality)
        values = (self.appearance_similarity, self.structure_similarity, self.relative_depth_consistency, self.motion_direction_consistency)
        if not math.isfinite(quality) or not 0.0 <= quality <= 1.0:
            raise ValueError("Reappearance quality must be in [0, 1].")
        if any(not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0 for value in values):
            raise ValueError("Reappearance similarities must be in [0, 1].")
        if self.valid and (not self.semantic_label_match or self.missing_reason):
            raise ValueError("Valid reappearance requires semantic support and no reason.")
        if not self.valid and not self.missing_reason:
            raise ValueError("Invalid reappearance requires missing_reason.")
        object.__setattr__(self, "quality", quality)
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class ReappearanceConsistencyResidual:
    """Consistency residual emitted only for a sufficiently supported re-id."""

    observation: ReappearanceObservation
    evidence: ResidualEvidence
    valid: bool
    missing_reason: str = ""


def evaluate_reappearance(
    *,
    previous_object_track_id: str,
    candidate_object_track_id: str,
    frame_index: int,
    predicted_reappearance_region: Optional[Sequence[float]],
    semantic_label_match: bool,
    appearance_similarity: float,
    structure_similarity: float,
    relative_depth_consistency: float,
    motion_direction_consistency: float,
    reid_source: str,
    scene_cut: bool = False,
    minimum_quality: float = 0.60,
) -> ReappearanceConsistencyResidual:
    """Reject uncertain/wrong identities instead of creating false continuity."""

    similarities = np.asarray([appearance_similarity, structure_similarity, relative_depth_consistency, motion_direction_consistency], dtype=float)
    quality = float(np.mean(similarities))
    region = None if predicted_reappearance_region is None else tuple(float(value) for value in predicted_reappearance_region)
    reason = ""
    if scene_cut:
        reason = "scene_cut_forbids_reappearance_link"
    elif region is None or len(region) != 4:
        reason = "missing_predicted_reappearance_region"
    elif not semantic_label_match:
        reason = "semantic_identity_mismatch"
    elif previous_object_track_id != candidate_object_track_id and quality < 0.85:
        reason = "uncertain_cross_id_reappearance"
    elif quality < minimum_quality:
        reason = "insufficient_reid_support"
    observation = ReappearanceObservation(
        previous_object_track_id, candidate_object_track_id, frame_index, region,
        semantic_label_match, float(appearance_similarity), float(structure_similarity),
        float(relative_depth_consistency), float(motion_direction_consistency),
        reid_source, quality if math.isfinite(quality) else 0.0, not reason,
        reason, {"bbox_iou_only": False, "scene_cut": scene_cut},
    )
    source_ids = (previous_object_track_id, candidate_object_track_id, str(frame_index))
    if reason:
        evidence = ResidualEvidence.missing("r_reappearance_consistency", reason, source_ids=source_ids)
        return ReappearanceConsistencyResidual(observation, evidence, False, reason)
    residual = 1.0 - quality
    evidence = ResidualEvidence.observed("r_reappearance_consistency", residual, quality=quality, source_ids=source_ids, metadata={"reid_source": reid_source, "identity_link_accepted": True})
    return ReappearanceConsistencyResidual(observation, evidence, True)
