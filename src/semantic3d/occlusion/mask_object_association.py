"""One-to-one association of real segmentation candidates to object tracks."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from ..observations import FrameObservationJSON, ObjectObservationJSON
from .mask_observation import InstanceMaskObservation, mask_bbox
from .mask_provider import InstanceMaskCandidate


def _label(value: str) -> str:
    normalized = "_".join(value.strip().lower().split())
    return "soccer_ball" if normalized in {"sports_ball", "ball"} else normalized


def _bbox_iou(first: Sequence[float], second: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = (float(value) for value in first)
    bx1, by1, bx2, by2 = (float(value) for value in second)
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return float(intersection / union) if union > 0.0 else 0.0


@dataclass(frozen=True)
class MaskAssociationDiagnostic:
    """Association decision for one object with all candidate explanations."""

    object_id: str
    object_track_id: str
    candidate_id: str | None
    association_quality: float
    association_source: str
    valid: bool
    missing_reason: str = ""
    candidate_details: tuple[Mapping[str, Any], ...] = ()


@dataclass(frozen=True)
class MaskObjectAssociationResult:
    """Aligned masks plus one-to-one assignment diagnostics."""

    masks: tuple[InstanceMaskObservation, ...]
    diagnostics: tuple[MaskAssociationDiagnostic, ...]
    assigned_candidate_ids: tuple[str, ...]
    rejected_candidate_ids: tuple[str, ...]


def _candidate_score(
    obj: ObjectObservationJSON,
    candidate: InstanceMaskCandidate,
    frame: FrameObservationJSON,
) -> tuple[float, str, Mapping[str, Any]]:
    object_label, candidate_label = _label(obj.canonical_label or obj.label), _label(candidate.class_name)
    if object_label != candidate_label:
        return 0.0, "semantic_category_conflict", {
            "candidate_id": candidate.candidate_id,
            "accepted": False,
            "rejection_reason": "semantic_category_conflict",
            "object_label": object_label,
            "candidate_label": candidate_label,
        }
    candidate_bbox = mask_bbox(candidate.visible_mask)
    if obj.bbox is None or candidate_bbox is None:
        return 0.0, "missing_bbox_for_association", {
            "candidate_id": candidate.candidate_id,
            "accepted": False,
            "rejection_reason": "missing_bbox_for_association",
        }
    iou = _bbox_iou(obj.bbox, candidate_bbox)
    ox = 0.5 * (obj.bbox[0] + obj.bbox[2])
    oy = 0.5 * (obj.bbox[1] + obj.bbox[3])
    cx = 0.5 * (candidate_bbox[0] + candidate_bbox[2])
    cy = 0.5 * (candidate_bbox[1] + candidate_bbox[3])
    diagonal = max(math.hypot(frame.width, frame.height), 1e-8)
    center_score = max(0.0, 1.0 - math.hypot(ox - cx, oy - cy) / diagonal)
    x1, y1, x2, y2 = (int(round(value)) for value in obj.bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.width, x2), min(frame.height, y2)
    inside = float(np.count_nonzero(candidate.visible_mask[y1:y2, x1:x2])) if x2 > x1 and y2 > y1 else 0.0
    containment = inside / max(float(np.count_nonzero(candidate.visible_mask)), 1.0)
    same_source = str(obj.object_id) == candidate.source_detection_id
    score = min(1.0, 0.50 * iou + 0.15 * center_score + 0.15 * containment + 0.10 * candidate.confidence + (0.10 if same_source else 0.0))
    source = "source_detection_id" if same_source else "semantic_mask_bbox_geometry"
    return score, source, {
        "candidate_id": candidate.candidate_id,
        "accepted": False,
        "rejection_reason": "below_minimum_association_quality",
        "bbox_iou": iou,
        "center_score": center_score,
        "containment": containment,
        "segmentation_confidence": candidate.confidence,
        "same_source_detection_id": same_source,
        "association_score": score,
    }


def associate_instance_masks(
    *,
    video_id: str,
    frame: FrameObservationJSON,
    candidates: Sequence[InstanceMaskCandidate],
    minimum_quality: float = 0.35,
) -> MaskObjectAssociationResult:
    """Greedily assign highest-quality compatible pairs with one-to-one use."""

    pair_rows = []
    details_by_object: dict[str, list[dict[str, Any]]] = {obj.object_id: [] for obj in frame.objects}
    for object_index, obj in enumerate(frame.objects):
        for candidate_index, candidate in enumerate(candidates):
            score, source, details = _candidate_score(obj, candidate, frame)
            details_by_object[obj.object_id].append(dict(details))
            if score >= minimum_quality and source != "semantic_category_conflict":
                pair_rows.append((score, object_index, candidate_index, source))
    pair_rows.sort(key=lambda row: (row[3] != "source_detection_id", -row[0], row[1], row[2]))
    assigned_objects: dict[int, tuple[int, float, str]] = {}
    assigned_candidates: set[int] = set()
    for score, object_index, candidate_index, source in pair_rows:
        if object_index in assigned_objects or candidate_index in assigned_candidates:
            continue
        assigned_objects[object_index] = (candidate_index, score, source)
        assigned_candidates.add(candidate_index)

    masks, diagnostics = [], []
    shape = (frame.height, frame.width)
    for object_index, obj in enumerate(frame.objects):
        track_id = str(obj.track_id or obj.person_track_id or obj.object_id)
        assignment = assigned_objects.get(object_index)
        if assignment is None:
            reason = "no_compatible_instance_mask_candidate" if candidates else "no_instance_mask_candidates"
            masks.append(InstanceMaskObservation.missing(
                video_id=video_id, frame_index=frame.frame_index,
                object_track_id=track_id, semantic_label=obj.label,
                image_shape=shape, reason=reason,
                source_provider="real_instance_mask_provider",
            ))
            diagnostics.append(MaskAssociationDiagnostic(
                obj.object_id, track_id, None, 0.0, "unassigned", False,
                reason, tuple(details_by_object[obj.object_id]),
            ))
            continue
        candidate_index, quality, source = assignment
        candidate = candidates[candidate_index]
        for details in details_by_object[obj.object_id]:
            if details["candidate_id"] == candidate.candidate_id:
                details["accepted"] = True
                details["rejection_reason"] = ""
        masks.append(InstanceMaskObservation.from_visible_mask(
            video_id=video_id, frame_index=frame.frame_index,
            object_track_id=track_id, semantic_label=obj.label,
            mask=candidate.visible_mask,
            confidence=min(candidate.confidence, float(obj.confidence), quality),
            source_provider=candidate.source_provider,
            metadata={
                "model_name": candidate.model_name,
                "model_version": candidate.model_version,
                "segmentation_confidence": candidate.confidence,
                "class_id": candidate.class_id,
                "class_name": candidate.class_name,
                "source_detection_id": candidate.source_detection_id,
                "inference_device": candidate.inference_device,
                "weight_sha256": candidate.weight_sha256,
                "preprocessing_metadata": dict(candidate.preprocessing_metadata),
                "whether_bbox_prompted": False,
                "whether_temporally_propagated": False,
                "truth_label_used": False,
                "formal_mask_evidence": True,
                "association_quality": quality,
                "association_source": source,
                "candidate_id": candidate.candidate_id,
            },
        ))
        diagnostics.append(MaskAssociationDiagnostic(
            obj.object_id, track_id, candidate.candidate_id, quality, source,
            True, candidate_details=tuple(details_by_object[obj.object_id]),
        ))
    assigned_ids = tuple(candidates[index].candidate_id for index in sorted(assigned_candidates))
    rejected_ids = tuple(candidate.candidate_id for index, candidate in enumerate(candidates) if index not in assigned_candidates)
    return MaskObjectAssociationResult(tuple(masks), tuple(diagnostics), assigned_ids, rejected_ids)
