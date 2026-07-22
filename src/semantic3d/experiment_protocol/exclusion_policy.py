"""Explicit P4-C1 technical exclusion policy without label-performance logic."""

from __future__ import annotations

from collections.abc import Iterable

from .manifest_schema import ALLOWED_SPLITS, SampleAvailability


def evaluate_exclusion_reasons(
    availability: SampleAvailability,
    *,
    clip_valid: bool,
    clip_missing_reason: str,
    authenticity_label: int | None,
    split: str,
    required_modalities: Iterable[str],
    require_complete_decoded_frames: bool = True,
) -> tuple[str, ...]:
    """Return stable reasons for excluding a clip from technical use.

    Missing optional dynamic or 3D branches remain availability facts and do
    not make the whole sample unusable. No residual, score, or class outcome is
    consulted here.
    """

    reasons: list[str] = []
    required = frozenset(str(value).strip() for value in required_modalities)
    if not clip_valid:
        reasons.append(f"invalid_clip:{clip_missing_reason or 'unspecified'}")
    if authenticity_label not in {0, 1}:
        reasons.append("authenticity_label_unavailable")
    if split not in ALLOWED_SPLITS or split == "official_conflict":
        reasons.append(f"invalid_or_conflicting_split:{split or 'missing'}")
    if "video" in required:
        if not availability.video_exists:
            reasons.append("source_video_missing")
        elif not availability.video_readable:
            reasons.append("source_video_unreadable")
        elif not availability.video_hash_matches:
            reasons.append("source_video_sha256_mismatch")
    if "frames" in required:
        if availability.indexed_frame_count <= 0:
            reasons.append("frame_index_missing")
        if availability.decoded_frame_count <= 0:
            reasons.append("decoded_frames_missing")
        elif (
            require_complete_decoded_frames
            and availability.decoded_frame_count < availability.expected_frame_count
        ):
            reasons.append("decoded_frames_incomplete")
    count_by_modality = {
        "objects": availability.valid_object_count,
        "depth": availability.valid_depth_count,
        "camera": availability.camera_observation_count,
        "pose": availability.valid_pose_count,
        "tracks": availability.valid_track_point_count,
        "semantic3d": availability.valid_semantic3d_count,
        "camera_identity": int(availability.camera_identity_available),
    }
    for modality in sorted(required - {"video", "frames"}):
        if modality not in count_by_modality:
            raise ValueError(f"Unknown required modality: {modality}")
        if count_by_modality[modality] <= 0:
            reasons.append(f"required_{modality}_unavailable")
    return tuple(reasons)


def exclusion_reason_text(reasons: Iterable[str]) -> str:
    """Serialize exclusion reasons without losing multiple causes."""

    return "|".join(dict.fromkeys(str(reason) for reason in reasons if str(reason)))

