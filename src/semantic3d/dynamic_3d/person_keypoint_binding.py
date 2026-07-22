"""Bind semantic human keypoints to object tracks and shared 3D geometry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..keypoint_provider import BaseKeypointProvider, COCO_PERSON_KEYPOINT_NAMES
from ..observations import FrameObservationJSON
from ..sequence_geometry import Shared3DClipObservation
from ..shared_3d_observation import VisibilityStatus
from .object_track_binding import (
    ObjectPointBinding,
    ObjectPointTrack3D,
    PointRole,
    assemble_object_point_tracks_3d,
)
from .readiness import Dynamic3DReadiness
from .track_observation import (
    PointTrack2DObservation,
    PointTrack3DObservation,
    reconstruct_point_tracks_3d,
)


@dataclass(frozen=True)
class PersonKeypointCoverage:
    """Per-frame/person keypoint availability without anomaly semantics."""

    video_id: str
    frame_index: int
    object_track_id: str
    total_keypoints: int
    valid_keypoints: int
    valid_ratio: float
    provider_name: str
    status: str
    valid: bool
    missing_reason: str = ""


@dataclass(frozen=True)
class PersonKeypointBindingResult:
    """Stable semantic point IDs, shared-depth 3D samples, and object tracks."""

    points_2d: tuple[PointTrack2DObservation, ...]
    points_3d: tuple[PointTrack3DObservation, ...]
    bindings: tuple[ObjectPointBinding, ...]
    object_tracks: tuple[ObjectPointTrack3D, ...]
    coverage: tuple[PersonKeypointCoverage, ...]


def _track_id(obj: object) -> str:
    return str(
        getattr(obj, "track_id", None)
        or getattr(obj, "person_track_id", None)
        or getattr(obj, "object_id")
    )


def bind_person_keypoints_to_shared_3d(
    *,
    video_id: str,
    clip_id: str,
    frames: Sequence[FrameObservationJSON],
    provider: BaseKeypointProvider,
    shared_clip: Shared3DClipObservation,
    readiness: Dynamic3DReadiness,
) -> PersonKeypointBindingResult:
    """Infer per-person keypoints and reconstruct each valid point independently."""

    points_2d: list[PointTrack2DObservation] = []
    coverage: list[PersonKeypointCoverage] = []
    confidence_by_point: dict[str, list[float]] = {}
    valid_frames_by_point: dict[str, list[int]] = {}
    track_by_point: dict[str, str] = {}
    name_by_point: dict[str, str] = {}

    for frame in sorted(frames, key=lambda item: item.frame_index):
        for obj in frame.objects:
            if obj.label.strip().lower().replace(" ", "_") != "person":
                continue
            track_id = _track_id(obj)
            if obj.bbox is None or not frame.image_path:
                status = "missing_bbox_or_image"
                provider_name = type(provider).__name__
                keypoint_by_name = {}
            else:
                prediction = provider.predict(frame.image_path, obj.bbox, obj.label)
                status = prediction.status
                provider_name = prediction.provider_name
                keypoint_by_name = {point.keypoint_name: point for point in prediction.keypoints}
            valid_count = 0
            for name in COCO_PERSON_KEYPOINT_NAMES:
                point_id = f"{track_id}:keypoint:{name}"
                track_by_point[point_id] = track_id
                name_by_point[point_id] = name
                point = keypoint_by_name.get(name)
                if point is None or not point.valid:
                    points_2d.append(PointTrack2DObservation.missing(
                        point_id=point_id,
                        object_track_id=track_id,
                        frame_index=frame.frame_index,
                        reason=("keypoint_not_returned" if point is None else "low_keypoint_confidence"),
                        source_tracker=provider_name,
                    ))
                    continue
                valid_count += 1
                confidence_by_point.setdefault(point_id, []).append(float(point.confidence))
                valid_frames_by_point.setdefault(point_id, []).append(frame.frame_index)
                points_2d.append(PointTrack2DObservation(
                    point_id=point_id,
                    object_track_id=track_id,
                    frame_index=frame.frame_index,
                    pixel_uv=(float(point.x), float(point.y)),
                    visibility=VisibilityStatus.VISIBLE,
                    occlusion_status="visible",
                    tracking_confidence=float(point.confidence),
                    source_tracker=provider_name,
                    valid=True,
                    metadata={
                        "independent_observation": True,
                        "generated_from_projection": False,
                        "semantic_keypoint_name": name,
                        "left_right_identity_fixed_by_provider": True,
                        "source_object_id": obj.object_id,
                        "frame_index_shared_with_object": True,
                    },
                ))
            total = len(COCO_PERSON_KEYPOINT_NAMES)
            ratio = valid_count / total
            coverage.append(PersonKeypointCoverage(
                video_id, frame.frame_index, track_id, total, valid_count,
                ratio, provider_name, status, bool(valid_count),
                "" if valid_count else status or "no_valid_keypoints",
            ))

    bindings = []
    for point_id in sorted(track_by_point):
        frame_indices = tuple(sorted(set(valid_frames_by_point.get(point_id, ()))))
        valid = bool(frame_indices)
        quality = float(np.mean(confidence_by_point.get(point_id, (0.0,)))) if valid else 0.0
        bindings.append(ObjectPointBinding(
            video_id=video_id,
            clip_id=clip_id,
            object_track_id=track_by_point[point_id],
            point_id=point_id,
            point_role=PointRole.SEMANTIC_KEYPOINT,
            semantic_keypoint_name=name_by_point[point_id],
            frame_indices=frame_indices,
            assignment_source="real_human_pose_provider",
            assignment_quality=quality,
            mask_support_ratio=0.0,
            bbox_support_ratio=1.0,
            track_consistency_ratio=1.0 if valid else 0.0,
            valid=valid,
            missing_reason="" if valid else "semantic_keypoint_never_valid",
            metadata={
                "semantic_label": "person",
                "semantic_identity_fixed": True,
                "truth_label_used": False,
            },
        ))
    reconstructed = reconstruct_point_tracks_3d(points_2d, shared_clip, readiness)
    tracks = assemble_object_point_tracks_3d(bindings, points_2d, reconstructed)
    return PersonKeypointBindingResult(
        tuple(points_2d), tuple(reconstructed), tuple(bindings), tracks,
        tuple(coverage),
    )
