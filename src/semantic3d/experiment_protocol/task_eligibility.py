"""Build task-specific P4-C2 eligibility without collapsing missing states."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping

from .formal_schema import TaskEligibilityRecord

TASK_NAMES = (
    "video_classification",
    "temporal_localization",
    "spatial_localization",
    "object_localization",
    "static_geometry",
    "depth_residual",
    "reprojection_D1",
    "reprojection_D2",
    "reprojection_D3",
    "occlusion_and_reappearance",
)


def _record(
    video: Mapping[str, Any],
    task: str,
    status: str,
    reason: str,
    *,
    available: int = 0,
    expected: int = 0,
    provider: str = "metadata",
) -> TaskEligibilityRecord:
    smoke_eligible = status == "eligible"
    return TaskEligibilityRecord(
        dataset_name=str(video["dataset_name"]),
        derived_video_id=str(video["video_id"]),
        task_name=task,
        eligibility=status,
        eligible_for_declared_role=smoke_eligible,
        eligible_for_formal_experiment=False,
        evaluation_scope="geometry_validation_smoke",
        ineligibility_reason=reason,
        provider_name=provider,
        available_observation_count=max(0, int(available)),
        expected_observation_count=max(0, int(expected)),
        metadata={
            "dataset_role": "geometry_validation_smoke",
            "provider_failure_is_anomaly_evidence": False,
            "formal_eligibility_blocked_by_dataset_role": True,
        },
    )


def build_task_eligibility_matrix(
    video_rows: Iterable[Mapping[str, Any]],
    p4c1_records: Iterable[Mapping[str, Any]],
    branch_rows: Iterable[Mapping[str, Any]],
) -> tuple[TaskEligibilityRecord, ...]:
    """Return one explicit eligibility state for every video and required task."""

    sample_rows: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in p4c1_records:
        sample_rows[str(row["source_video_id"])].append(row)
    branch_by_video: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in branch_rows:
        if str(row.get("entity_type")) != "video":
            continue
        key = (str(row["video_id"]), str(row["branch_name"]))
        branch_by_video.setdefault(key, row)

    output: list[TaskEligibilityRecord] = []
    for video in sorted(video_rows, key=lambda row: str(row["video_id"])):
        video_id = str(video["video_id"])
        clips = sample_rows[video_id]
        clip_count = len(clips)
        total_depth = sum(int(row.get("valid_depth_count", 0)) for row in clips)
        total_objects = sum(int(row.get("valid_object_count", 0)) for row in clips)

        for task, reason in (
            ("video_classification", "geometry_validation_smoke_not_formal_classification_data"),
            ("temporal_localization", "temporal_annotations_unavailable"),
            ("spatial_localization", "spatial_annotations_unavailable"),
            ("object_localization", "object_localization_annotations_unavailable"),
        ):
            output.append(_record(video, task, "ineligible", reason, expected=clip_count))

        static = branch_by_video.get((video_id, "frame_static_3d"))
        if static and str(static.get("eligibility_status")) == "applicable":
            static_count = int(_metadata_value(static, "valid_frame_count", 0))
            output.append(
                _record(
                    video,
                    "static_geometry",
                    "eligible",
                    "",
                    available=static_count,
                    expected=int(video.get("frame_count", 0)),
                    provider="shared_3d_frame_observation",
                )
            )
        else:
            output.append(
                _record(
                    video,
                    "static_geometry",
                    "provider_failed",
                    str(static.get("exclusion_reason", "static_geometry_metadata_missing"))
                    if static
                    else "static_geometry_metadata_missing",
                    expected=int(video.get("frame_count", 0)),
                    provider="shared_3d_frame_observation",
                )
            )

        depth_status = "eligible" if total_depth > 0 else "provider_failed"
        output.append(
            _record(
                video,
                "depth_residual",
                depth_status,
                "" if total_depth else "depth_observation_unavailable",
                available=total_depth,
                expected=sum(int(row.get("num_frames", 0)) for row in clips),
                provider="depth_observation",
            )
        )

        for task, branch_name in (
            ("reprojection_D1", "dynamic_static_camera"),
            ("reprojection_D2", "dynamic_rotation_compensated"),
            ("reprojection_D3", "dynamic_full_se3"),
        ):
            branch = branch_by_video.get((video_id, branch_name))
            if branch and str(branch.get("eligibility_status")) == "applicable":
                ready = int(_metadata_value(branch, "ready_clip_count", 0))
                output.append(
                    _record(
                        video,
                        task,
                        "eligible",
                        "",
                        available=ready,
                        expected=clip_count,
                        provider="camera_pose_and_track_geometry",
                    )
                )
            else:
                reason = (
                    str(branch.get("exclusion_reason", "reprojection_metadata_missing"))
                    if branch
                    else "reprojection_metadata_missing"
                )
                output.append(
                    _record(
                        video,
                        task,
                        "provider_failed",
                        reason,
                        expected=clip_count,
                        provider="camera_pose_and_track_geometry",
                    )
                )

        occlusion = branch_by_video.get((video_id, "occlusion"))
        if not occlusion:
            occ_status, reason, count = "unknown", "occlusion_metadata_missing", 0
        else:
            source_status = str(occlusion.get("eligibility_status", "unknown"))
            count = int(_metadata_value(occlusion, "formal_event_count", 0))
            if source_status == "applicable":
                occ_status, reason = "eligible", ""
            elif source_status == "not_applicable":
                occ_status = "not_applicable"
                reason = str(occlusion.get("exclusion_reason", "no_observed_event"))
            else:
                occ_status = "provider_failed"
                reason = str(occlusion.get("exclusion_reason", "occlusion_provider_failed"))
        output.append(
            _record(
                video,
                "occlusion_and_reappearance",
                occ_status,
                reason,
                available=count,
                expected=clip_count,
                provider="formal_mask_depth_order_visibility",
            )
        )

        produced = {row.task_name for row in output if row.derived_video_id == video_id}
        if produced != set(TASK_NAMES):
            raise RuntimeError(f"Incomplete task matrix for {video_id}: {sorted(produced)}")

    return tuple(sorted(output, key=lambda row: (row.dataset_name, row.derived_video_id, row.task_name)))


def _metadata_value(row: Mapping[str, Any], name: str, default: Any) -> Any:
    """Read a branch metadata value stored as a mapping or canonical JSON text."""

    import json

    metadata = row.get("metadata", {})
    if isinstance(metadata, str):
        metadata = json.loads(metadata) if metadata else {}
    return metadata.get(name, default)

