"""Derive observation eligibility tiers without treating missingness as anomaly."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from semantic3d.aggregation_v2.evidence_registry import get_evidence_registry

from .schema import BranchEligibilityRecord


def _rows(path: Path) -> list[dict[str, Any]]:
    return pq.read_table(path).to_pylist()


def _record(
    entity_type: str,
    entity_id: str,
    video_id: str,
    tier: str,
    branch: str,
    status: str,
    reason: str,
    *,
    expected: bool,
    geometry: str,
    mask: str = "not_required",
    keypoints: str = "not_required",
    event: str = "not_required",
    metadata: dict[str, Any] | None = None,
) -> BranchEligibilityRecord:
    return BranchEligibilityRecord(
        entity_type=entity_type,
        entity_id=entity_id,
        video_id=video_id,
        tier=tier,
        branch_name=branch,
        applicable=status == "applicable",
        eligibility_status=status,
        expected_observation_available=expected,
        geometry_requirement=geometry,
        mask_requirement=mask,
        keypoint_requirement=keypoints,
        event_requirement=event,
        exclusion_reason=reason,
        metadata=metadata or {},
    )


def build_branch_eligibility(structural_dataset_root: str | Path) -> list[BranchEligibilityRecord]:
    """Build Tier S/D1/D2/D3/O records for every video and clip."""

    root = Path(structural_dataset_root)
    dataset_manifest = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))
    depth_convention = str(dataset_manifest.get("depth_convention", "unknown"))
    metric_scale_available = "metric" in depth_convention and "not_metric" not in depth_convention
    videos = _rows(root / "manifests/videos.parquet")
    clips = _rows(root / "manifests/clips.parquet")
    frames = _rows(root / "manifests/frames.parquet")
    shared = _rows(root / "observations/shared_3d_frames.parquet")
    readiness = {row["clip_id"]: row for row in _rows(root / "observations/dynamic_readiness.parquet")}
    masks = _rows(root / "observations/masks.parquet")
    keypoints = _rows(root / "observations/keypoints.parquet")
    point_tracks = _rows(root / "observations/point_tracks_2d.parquet")
    occlusion = _rows(root / "reports/occlusion_depth_order_sync.parquet")

    valid_shared = {(row["video_id"], int(row["frame_index"])) for row in shared if row["valid"]}
    frame_indices_by_clip: dict[str, set[int]] = defaultdict(set)
    for row in frames:
        frame_indices_by_clip[str(row["clip_id"])].add(int(row["frame_index"]))
    valid_masks_by_video: dict[str, int] = defaultdict(int)
    for row in masks:
        valid_masks_by_video[str(row["video_id"])] += int(bool(row["valid"] and not row["bbox_fallback"]))
    valid_keypoints_by_video: dict[str, int] = defaultdict(int)
    for row in keypoints:
        valid_keypoints_by_video[str(row["video_id"])] += int(bool(row["valid"]))
    valid_points_by_clip: dict[str, int] = defaultdict(int)
    for row in point_tracks:
        valid_points_by_clip[str(row["clip_id"])] += int(bool(row["valid"]))
    occ_by_clip: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in occlusion:
        occ_by_clip[str(row["clip_id"])].append(row)

    output: list[BranchEligibilityRecord] = []
    clips_by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for clip in clips:
        clips_by_video[str(clip["video_id"])].append(clip)

    def append_entity(entity_type: str, entity_id: str, video_id: str, entity_clips: list[dict[str, Any]]) -> None:
        indices = set().union(*(frame_indices_by_clip[str(clip["clip_id"])] for clip in entity_clips)) if entity_clips else set()
        shared_count = sum((video_id, index) in valid_shared for index in indices)
        s_status = "applicable" if shared_count > 0 else "observation_missing"
        output.append(_record(entity_type, entity_id, video_id, "S", "frame_static_3d", s_status, "" if shared_count else "no_valid_frame_shared_3d", expected=bool(indices), geometry="frame_camera_relative", metadata={"valid_frame_count": shared_count}))

        for tier, mode, branch in (
            ("D1", "static_camera_3d", "dynamic_static_camera"),
            ("D2", "rotation_compensated", "dynamic_rotation_compensated"),
            ("D3", "full_se3_3d", "dynamic_full_se3"),
        ):
            matching = [readiness[str(clip["clip_id"])] for clip in entity_clips if readiness[str(clip["clip_id"])]["geometry_mode"] == mode and readiness[str(clip["clip_id"])]["dynamic_3d_ready"]]
            status = "applicable" if matching else "observation_missing"
            output.append(_record(entity_type, entity_id, video_id, tier, branch, status, "" if matching else f"no_{mode}_ready_clip", expected=bool(entity_clips), geometry=mode, metadata={"ready_clip_count": len(matching)}))

        occ_rows = [row for clip in entity_clips for row in occ_by_clip[str(clip["clip_id"])]]
        formal_events = sum(bool(row["formal_occlusion_event"]) for row in occ_rows)
        observed_candidates = sum(bool(row["new_overlap_candidate"] and row["depth_order_valid"]) for row in occ_rows)
        if formal_events:
            o_status, reason = "applicable", ""
        elif observed_candidates:
            o_status, reason = "not_applicable", "no_validated_occlusion_event"
        elif occ_rows:
            o_status, reason = "observation_missing", "occlusion_observation_incomplete"
        else:
            o_status, reason = "not_applicable", "no_observable_occlusion_event"
        output.append(_record(entity_type, entity_id, video_id, "O", "occlusion", o_status, reason, expected=bool(occ_rows), geometry="depth_order", mask="formal_instance_mask", event="partial_or_full_occlusion", metadata={"formal_event_count": formal_events, "observed_candidate_count": observed_candidates}))

        ready_modes = {
            str(readiness[str(clip["clip_id"])]["geometry_mode"])
            for clip in entity_clips
            if readiness[str(clip["clip_id"])]["dynamic_3d_ready"]
        }
        valid_point_count = sum(valid_points_by_clip[str(clip["clip_id"])] for clip in entity_clips)
        registry = get_evidence_registry(formal_only=True)
        for branch_name, spec in registry.items():
            supported = set(spec.supported_geometry_modes)
            geometry_ok = bool(ready_modes & supported)
            mask_required = spec.localization_target in {
                "object_mask", "object_boundary", "mask_boundary", "occlusion_pair"
            } or branch_name in {"structure_temporal", "visibility_explanation", "reappearance_consistency"}
            point_required = spec.evidence_level.value in {"point", "edge"} and branch_name not in {
                "boundary_depth_3d", "boundary_occlusion"
            }
            mask_ok = valid_masks_by_video[video_id] > 0
            points_ok = valid_point_count > 0
            event_ok = formal_events > 0
            if branch_name == "semantic_size_3d" and not metric_scale_available:
                status, branch_reason = (
                    "observation_missing",
                    "metric_or_calibrated_scale_anchor_unavailable",
                )
            elif spec.event_conditioned and not event_ok:
                if occ_rows and (observed_candidates or any(row["depth_order_valid"] for row in occ_rows)):
                    status, branch_reason = "not_applicable", "required_event_not_observed"
                elif occ_rows:
                    status, branch_reason = "observation_missing", "event_observation_incomplete"
                else:
                    status, branch_reason = "not_applicable", "required_event_not_present"
            elif not geometry_ok:
                status, branch_reason = "invalid_geometry", "supported_geometry_mode_unavailable"
            elif mask_required and not mask_ok:
                status, branch_reason = "observation_missing", "formal_mask_unavailable"
            elif point_required and not points_ok:
                status, branch_reason = "observation_missing", "independent_structure_points_unavailable"
            else:
                status, branch_reason = "applicable", ""
            if "static_camera_3d" in supported:
                formal_tier = "D1"
            elif "rotation_compensated" in supported:
                formal_tier = "D2"
            else:
                formal_tier = "D3"
            output.append(
                _record(
                    entity_type,
                    entity_id,
                    video_id,
                    formal_tier,
                    branch_name,
                    status,
                    branch_reason,
                    expected=bool(entity_clips),
                    geometry="|".join(spec.supported_geometry_modes),
                    mask="formal_instance_mask" if mask_required else "not_required",
                    keypoints=(
                        "semantic_keypoints_or_formal_mask_internal_points"
                        if branch_name == "structure_temporal"
                        else "not_required"
                    ),
                    event="required" if spec.event_conditioned else "not_required",
                    metadata={
                        "evidence_level": spec.evidence_level.value,
                        "formal_branch": True,
                        "ready_geometry_modes": sorted(ready_modes),
                        "valid_independent_point_observations": valid_point_count,
                    },
                )
            )

    for video in videos:
        video_id = str(video["video_id"])
        append_entity("video", video_id, video_id, clips_by_video[video_id])
    for clip in clips:
        append_entity("clip", str(clip["clip_id"]), str(clip["video_id"]), [clip])
    return output


def branch_rows(records: list[BranchEligibilityRecord]) -> list[dict[str, Any]]:
    """Convert eligibility dataclasses to table rows."""

    return [asdict(record) for record in records]
