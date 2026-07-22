"""Sparse evidence-backed foreground/background occlusion relations."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import cv2
import numpy as np

from .mask_observation import InstanceMaskObservation, PredictedObjectSupport


@dataclass(frozen=True)
class OcclusionRelation:
    """One candidate ordering relation; bbox overlap alone remains invalid."""

    foreground_object_id: str
    background_object_id: str
    frame_index: int
    predicted_overlap_area: float
    visible_overlap_area: float
    foreground_depth: float
    background_depth: float
    depth_margin: float
    boundary_contact_length: float
    occlusion_confidence: float
    valid: bool
    missing_reason: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        confidence = float(self.occlusion_confidence)
        if self.foreground_object_id == self.background_object_id:
            raise ValueError("Occlusion relation requires two object IDs.")
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("occlusion_confidence must be in [0, 1].")
        if self.valid:
            if self.predicted_overlap_area <= 0.0 or self.foreground_depth >= self.background_depth or self.missing_reason:
                raise ValueError("Valid occlusion requires overlap and foreground_depth < background_depth.")
        elif not self.missing_reason:
            raise ValueError("Invalid relation requires missing_reason.")
        object.__setattr__(self, "occlusion_confidence", confidence)
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class OcclusionGraph:
    """Sparse per-frame relation set; no complete n-by-n matrix is fabricated."""

    video_id: str
    frame_index: int
    relations: tuple[OcclusionRelation, ...]
    valid: bool
    quality: float
    missing_reason: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _contact_length(first: np.ndarray, second: np.ndarray) -> float:
    first_boundary = first ^ cv2.erode(first.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    near_second = cv2.dilate(second.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    return float(np.count_nonzero(first_boundary & near_second))


def build_occlusion_graph(
    *,
    video_id: str,
    frame_index: int,
    predicted_supports: Mapping[str, PredictedObjectSupport],
    observed_masks: Mapping[str, InstanceMaskObservation],
    object_depths: Mapping[str, float],
    depth_uncertainty: float = 0.05,
) -> OcclusionGraph:
    """Build only support-overlap candidates and order them by smaller Z-depth."""

    relations = []
    invalid_support_count = sum(
        not item.valid or item.support_mask is None
        for item in predicted_supports.values()
    )
    ids = sorted(predicted_supports)
    for index, first_id in enumerate(ids):
        for second_id in ids[index + 1 :]:
            first_support, second_support = predicted_supports[first_id], predicted_supports[second_id]
            if not first_support.valid or not second_support.valid or first_support.support_mask is None or second_support.support_mask is None:
                continue
            predicted_overlap = first_support.support_mask & second_support.support_mask
            overlap_area = float(np.count_nonzero(predicted_overlap))
            if overlap_area <= 0.0:
                continue
            first_mask, second_mask = observed_masks.get(first_id), observed_masks.get(second_id)
            visible_overlap = 0.0
            contact = 0.0
            legacy = True
            if first_mask and second_mask and first_mask.valid and second_mask.valid and first_mask.visible_mask is not None and second_mask.visible_mask is not None:
                visible_overlap = float(np.count_nonzero(first_mask.visible_mask & second_mask.visible_mask))
                contact = _contact_length(first_mask.visible_mask, second_mask.visible_mask)
                legacy = first_mask.is_legacy_bbox_fallback or second_mask.is_legacy_bbox_fallback
            depth_a, depth_b = float(object_depths.get(first_id, float("nan"))), float(object_depths.get(second_id, float("nan")))
            if not math.isfinite(depth_a) or not math.isfinite(depth_b):
                foreground, background, reason = first_id, second_id, "missing_relation_depth"
                fg_depth, bg_depth, margin = depth_a, depth_b, float("nan")
                valid = False
            elif abs(depth_a - depth_b) <= depth_uncertainty:
                foreground, background, reason = first_id, second_id, "depth_order_uncertain"
                fg_depth, bg_depth, margin = depth_a, depth_b, abs(depth_a - depth_b)
                valid = False
            else:
                foreground, background = (first_id, second_id) if depth_a < depth_b else (second_id, first_id)
                fg_depth, bg_depth, margin = min(depth_a, depth_b), max(depth_a, depth_b), abs(depth_a - depth_b)
                reason = "bbox_overlap_only_not_occlusion" if legacy else ""
                valid = not legacy
            quality = min(first_support.quality, second_support.quality)
            if legacy:
                quality = min(quality, 0.25)
            relations.append(OcclusionRelation(
                foreground, background, frame_index, overlap_area, visible_overlap,
                fg_depth, bg_depth, margin, contact, quality, valid, reason,
                {"predicted_support_overlap": True, "bbox_overlap_only": legacy, "depth_source": "object_center_depth", "depth_quality": "low_center_only"},
            ))
    valid_relations = [relation for relation in relations if relation.valid]
    if valid_relations:
        graph_reason = ""
    elif relations:
        graph_reason = "occlusion_observation_missing"
    elif invalid_support_count:
        graph_reason = "occlusion_observation_missing"
    else:
        graph_reason = "no_occlusion_event"
    return OcclusionGraph(
        video_id, frame_index, tuple(relations), bool(valid_relations),
        float(np.mean([item.occlusion_confidence for item in valid_relations])) if valid_relations else 0.0,
        graph_reason,
        {
            "candidate_relation_count": len(relations),
            "invalid_support_count": invalid_support_count,
            "complete_pair_matrix_generated": False,
            "no_event_distinct_from_missing": True,
        },
    )
