"""Audit split leakage and modeling-route isolation."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping

from .manifest_schema import ExperimentSampleRecord, LeakageFinding
from .schema import SourceGroupRecord, SplitAssignment, VideoInventoryRecord


def audit_leakage(
    inventory: Iterable[VideoInventoryRecord],
    assignments: Iterable[SplitAssignment],
    source_groups: Iterable[SourceGroupRecord],
    duplicate_rows: Iterable[Mapping[str, Any]],
    *,
    near_duplicate_threshold: float = 0.92,
    normalization_fit_splits: tuple[str, ...] = ("train",),
    threshold_tuning_splits: tuple[str, ...] = ("validation",),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return explicit conflicts; never silently repair official or source leakage."""

    rows = list(inventory)
    splits = {row.video_id: row.split for row in assignments}
    conflicts: list[dict[str, Any]] = []
    group_splits: dict[str, set[str]] = defaultdict(set)
    group_videos: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        group_splits[row.source_group_id].add(splits[row.video_id])
        group_videos[row.source_group_id].append(row.video_id)
    for group_id, values in group_splits.items():
        if len(values) > 1:
            conflicts.append(
                {
                    "conflict_type": "source_group_cross_split",
                    "severity": "error",
                    "video_id_a": group_videos[group_id][0],
                    "video_id_b": ";".join(group_videos[group_id][1:]),
                    "source_group_id": group_id,
                    "split_a": ";".join(sorted(values)),
                    "split_b": "",
                    "details": "one source group spans multiple splits",
                    "review_required": True,
                }
            )
    for row in duplicate_rows:
        if not bool(row["split_conflict"]):
            continue
        if bool(row["exact_duplicate"]):
            kind = "exact_duplicate_cross_split"
            severity = "error"
        elif float(row["near_duplicate_score"]) >= near_duplicate_threshold:
            kind = "near_duplicate_cross_split"
            severity = "error"
        else:
            continue
        conflicts.append(
            {
                "conflict_type": kind,
                "severity": severity,
                "video_id_a": row["video_id_a"],
                "video_id_b": row["video_id_b"],
                "source_group_id": "",
                "split_a": splits[str(row["video_id_a"])],
                "split_b": splits[str(row["video_id_b"])],
                "details": f"near_duplicate_score={float(row['near_duplicate_score']):.6f}",
                "review_required": True,
            }
        )
    for assignment in assignments:
        if assignment.split == "official_conflict":
            conflicts.append(
                {
                    "conflict_type": "official_split_source_group_conflict",
                    "severity": "error",
                    "video_id_a": assignment.video_id,
                    "video_id_b": "",
                    "source_group_id": assignment.source_group_id,
                    "split_a": assignment.split,
                    "split_b": "",
                    "details": "official split conflict was preserved for review",
                    "review_required": True,
                }
            )
    test_in_fit = "test" in normalization_fit_splits
    test_in_threshold = "test" in threshold_tuning_splits
    if test_in_fit:
        conflicts.append(
            {
                "conflict_type": "test_used_for_normalization_fit",
                "severity": "error",
                "video_id_a": "",
                "video_id_b": "",
                "source_group_id": "",
                "split_a": "test",
                "split_b": "",
                "details": "test cannot fit normalization statistics",
                "review_required": True,
            }
        )
    if test_in_threshold:
        conflicts.append(
            {
                "conflict_type": "test_used_for_threshold_tuning",
                "severity": "error",
                "video_id_a": "",
                "video_id_b": "",
                "source_group_id": "",
                "split_a": "test",
                "split_b": "",
                "details": "test cannot tune thresholds",
                "review_required": True,
            }
        )
    summary = {
        "conflict_count": len(conflicts),
        "error_count": sum(row["severity"] == "error" for row in conflicts),
        "source_group_review_required_count": sum(row.source_group_review_required for row in rows),
        "normalization_fit_splits": list(normalization_fit_splits),
        "threshold_tuning_splits": list(threshold_tuning_splits),
        "test_used_for_normalization_fit": test_in_fit,
        "test_used_for_threshold_tuning": test_in_threshold,
        "residual_values_used_for_split": False,
        "labels_read_by_structural_builder": False,
    }
    return conflicts, summary


def audit_manifest_leakage(
    records: Iterable[ExperimentSampleRecord],
    source_group_rows: Mapping[str, Mapping[str, Any]],
    duplicate_rows: Iterable[Mapping[str, Any]],
    *,
    near_duplicate_threshold: float = 0.92,
) -> tuple[list[LeakageFinding], dict[str, Any]]:
    """Audit P4-C1 source isolation after deduplicating overlapping clips.

    The function never reads residuals, scores, or model outputs. Unresolved
    original/derivative identity is a warning that requires provenance review,
    not proof of cross-split leakage.
    """

    samples = sorted(records, key=lambda row: row.sample_id)
    findings: list[LeakageFinding] = []
    sample_ids = [row.sample_id for row in samples]
    if len(sample_ids) != len(set(sample_ids)):
        findings.append(
            LeakageFinding(
                "duplicate_sample_id",
                "error",
                "",
                "",
                "",
                "",
                "",
                "sample_id must be globally unique",
                True,
            )
        )

    by_video: dict[str, ExperimentSampleRecord] = {}
    video_splits: dict[str, set[str]] = defaultdict(set)
    for row in samples:
        video_splits[row.source_video_id].add(row.split)
        by_video.setdefault(row.source_video_id, row)
    for video_id, splits in sorted(video_splits.items()):
        if len(splits) > 1:
            findings.append(
                LeakageFinding(
                    "source_video_cross_split",
                    "error",
                    video_id,
                    "",
                    by_video[video_id].source_group_id,
                    ";".join(sorted(splits)),
                    "",
                    "clips from one source video span multiple splits",
                    True,
                )
            )

    group_videos: dict[str, set[str]] = defaultdict(set)
    group_splits: dict[str, set[str]] = defaultdict(set)
    hash_videos: dict[str, set[str]] = defaultdict(set)
    hash_splits: dict[str, set[str]] = defaultdict(set)
    identity_videos: dict[str, set[str]] = defaultdict(set)
    identity_splits: dict[str, set[str]] = defaultdict(set)
    unresolved_videos: list[str] = []
    for video_id, row in sorted(by_video.items()):
        group_videos[row.source_group_id].add(video_id)
        group_splits[row.source_group_id].add(row.split)
        hash_videos[row.source_sha256].add(video_id)
        hash_splits[row.source_sha256].add(row.split)
        source_group = source_group_rows.get(video_id, {})
        identity = str(source_group.get("original_source_identity", "")).strip()
        review_required = bool(source_group.get("source_group_review_required", False))
        if identity:
            identity_videos[identity].add(video_id)
            identity_splits[identity].add(row.split)
        elif review_required:
            unresolved_videos.append(video_id)

    for group_id, splits in sorted(group_splits.items()):
        if len(splits) > 1:
            videos = sorted(group_videos[group_id])
            findings.append(
                LeakageFinding(
                    "source_group_cross_split",
                    "error",
                    videos[0],
                    ";".join(videos[1:]),
                    group_id,
                    ";".join(sorted(splits)),
                    "",
                    "original and known derivatives must share one split",
                    True,
                )
            )
    for identity, splits in sorted(identity_splits.items()):
        if len(splits) > 1:
            videos = sorted(identity_videos[identity])
            findings.append(
                LeakageFinding(
                    "original_source_identity_cross_split",
                    "error",
                    videos[0],
                    ";".join(videos[1:]),
                    "",
                    ";".join(sorted(splits)),
                    "",
                    f"original_source_identity={identity}",
                    True,
                )
            )
    for digest, splits in sorted(hash_splits.items()):
        if len(splits) > 1:
            videos = sorted(hash_videos[digest])
            findings.append(
                LeakageFinding(
                    "exact_video_hash_cross_split",
                    "error",
                    videos[0],
                    ";".join(videos[1:]),
                    "",
                    ";".join(sorted(splits)),
                    "",
                    f"source_sha256={digest}",
                    True,
                )
            )

    split_by_video = {video_id: row.split for video_id, row in by_video.items()}
    for duplicate in sorted(
        duplicate_rows,
        key=lambda row: (str(row.get("video_id_a", "")), str(row.get("video_id_b", ""))),
    ):
        first = str(duplicate.get("video_id_a", ""))
        second = str(duplicate.get("video_id_b", ""))
        if first not in split_by_video or second not in split_by_video:
            continue
        first_split = split_by_video[first]
        second_split = split_by_video[second]
        if first_split == second_split:
            continue
        exact = bool(duplicate.get("exact_duplicate", False))
        score = float(duplicate.get("near_duplicate_score", 0.0))
        if not exact and score < near_duplicate_threshold:
            continue
        findings.append(
            LeakageFinding(
                "exact_duplicate_cross_split" if exact else "near_duplicate_cross_split",
                "error",
                first,
                second,
                "",
                first_split,
                second_split,
                f"near_duplicate_score={score:.6f}",
                True,
            )
        )

    for video_id in sorted(unresolved_videos):
        row = by_video[video_id]
        findings.append(
            LeakageFinding(
                "unresolved_original_derivative_identity",
                "warning",
                video_id,
                "",
                row.source_group_id,
                row.split,
                "",
                "source group uses a file-hash fallback; original/derivative relationship is unknown",
                True,
            )
        )

    findings.sort(
        key=lambda row: (
            row.severity,
            row.finding_type,
            row.source_video_id_a,
            row.source_video_id_b,
        )
    )
    summary = {
        "sample_count": len(samples),
        "unique_source_video_count": len(by_video),
        "unique_source_group_count": len(group_splits),
        "error_count": sum(row.severity == "error" for row in findings),
        "warning_count": sum(row.severity == "warning" for row in findings),
        "cross_split_leakage_detected": any(row.severity == "error" for row in findings),
        "unresolved_original_derivative_identity_count": len(unresolved_videos),
        "overlapping_clips_inherit_video_split": True,
        "residual_values_used": False,
        "test_used_for_tuning": False,
    }
    return findings, summary
