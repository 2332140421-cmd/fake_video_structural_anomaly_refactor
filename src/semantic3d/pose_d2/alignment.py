"""Short-clip camera alignment without claiming a global world frame."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from ..geometry.camera import validate_rigid_transform
from .contracts import PairwisePoseObservation


@dataclass(frozen=True)
class ClipLocalAlignment:
    """Transform from one camera frame into the short-clip reference gauge."""

    clip_id: str
    frame_index: int
    reference_frame_index: int
    T_clip_from_camera: Optional[np.ndarray]
    coordinate_frame: str
    pose_source: str
    confidence: float
    valid: bool
    failure_reason: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        transform = self.T_clip_from_camera
        if transform is not None:
            transform = validate_rigid_transform(transform, "T_clip_from_camera")
        if self.coordinate_frame != "clip_local_aligned":
            raise ValueError("M4 alignment must be clip_local_aligned, not world_frame.")
        confidence = float(self.confidence)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("Alignment confidence must be in [0, 1].")
        if self.valid:
            if transform is None or self.failure_reason:
                raise ValueError("Valid alignment requires transform and no failure reason.")
        elif transform is not None or not self.failure_reason:
            raise ValueError("Invalid alignment requires no transform and a reason.")
        object.__setattr__(self, "T_clip_from_camera", transform)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "metadata", dict(self.metadata))


def build_clip_local_alignment(
    clip_id: str,
    frame_indices: Sequence[int],
    pairwise_poses: Sequence[PairwisePoseObservation],
) -> tuple[ClipLocalAlignment, ...]:
    """Compose accepted adjacent poses into a short-clip reference coordinate.

    The first frame defines a gauge identity. That identity is a coordinate
    definition, not an assertion that the camera is static.
    """

    frames = tuple(int(value) for value in frame_indices)
    if not frames:
        raise ValueError("frame_indices cannot be empty.")
    if len(set(frames)) != len(frames) or list(frames) != sorted(frames):
        raise ValueError("frame_indices must be unique and sorted.")
    reference = frames[0]
    by_edge = {(pose.frame_t, pose.frame_t1): pose for pose in pairwise_poses}
    result: list[ClipLocalAlignment] = [
        ClipLocalAlignment(
            clip_id=clip_id,
            frame_index=reference,
            reference_frame_index=reference,
            T_clip_from_camera=np.eye(4),
            coordinate_frame="clip_local_aligned",
            pose_source="reference_gauge",
            confidence=1.0,
            valid=True,
            metadata={
                "reference_gauge_only": True,
                "verified_static_identity": False,
                "world_frame_claimed": False,
            },
        )
    ]
    T_current_from_reference = np.eye(4)
    path_valid = True
    path_confidence = 1.0
    for previous, current in zip(frames, frames[1:]):
        pose = by_edge.get((previous, current))
        if (
            not path_valid
            or pose is None
            or not pose.valid
            or pose.T_target_from_source is None
        ):
            path_valid = False
            reason = (
                "upstream_alignment_path_invalid"
                if pose is not None
                else "pairwise_pose_missing"
            )
            if pose is not None and pose.failure_reason:
                reason = pose.failure_reason
            result.append(
                ClipLocalAlignment(
                    clip_id=clip_id,
                    frame_index=current,
                    reference_frame_index=reference,
                    T_clip_from_camera=None,
                    coordinate_frame="clip_local_aligned",
                    pose_source=pose.provider_name if pose is not None else "unavailable",
                    confidence=0.0,
                    valid=False,
                    failure_reason=reason,
                    metadata={"world_frame_claimed": False},
                )
            )
            continue
        T_current_from_reference = (
            pose.T_target_from_source @ T_current_from_reference
        )
        T_reference_from_current = np.linalg.inv(T_current_from_reference)
        path_confidence = min(path_confidence, pose.confidence)
        result.append(
            ClipLocalAlignment(
                clip_id=clip_id,
                frame_index=current,
                reference_frame_index=reference,
                T_clip_from_camera=T_reference_from_current,
                coordinate_frame="clip_local_aligned",
                pose_source=pose.provider_name,
                confidence=path_confidence,
                valid=True,
                metadata={
                    "pair_provider_status": pose.provider_status.value,
                    "translation_scale_status": pose.translation_scale_status,
                    "world_frame_claimed": False,
                },
            )
        )
    return tuple(result)
