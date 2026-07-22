"""Source lineage construction, formal split planning, and leakage audit."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any, Iterable, Mapping

from .formal_schema import FormalSplitRecord, SourceLineageRecord

FORMAL_SPLIT_ALGORITHM = "verified_source_group_hash_v1"


def build_smoke_source_lineage(
    video_rows: Iterable[Mapping[str, Any]],
    source_groups: Mapping[str, Mapping[str, Any]],
) -> tuple[SourceLineageRecord, ...]:
    """Convert current P4-C0 videos to explicitly unverified smoke lineage."""

    output = []
    for video in sorted(video_rows, key=lambda row: str(row["video_id"])):
        video_id = str(video["video_id"])
        group = source_groups[video_id]
        original = str(group.get("original_source_identity", ""))
        verified = bool(original) and not bool(group.get("source_group_review_required", True))
        label = video.get("binary_label")
        if verified:
            derivation_status = "original_or_derivation_verified"
            reason = ""
            evidence = "verified_source_metadata"
            verification = "verified"
        else:
            derivation_status = (
                "derived_source_unresolved" if label == 1 else "claimed_original_unverified"
            )
            reason = "original_source_identity_unverified"
            evidence = "file_sha256_only_insufficient_for_lineage"
            verification = "unverified"
        output.append(
            SourceLineageRecord(
                dataset_name=str(video["dataset_name"]),
                original_source_id=original,
                source_group_id=str(group["source_group_id"]),
                derived_video_id=video_id,
                source_video_name=str(video["source_name"]),
                video_path=str(video["source_path"]),
                source_sha256=str(video["source_sha256"]),
                authenticity_label=int(label) if label is not None else None,
                manipulation_type=str(video.get("manipulation_type", "unknown")),
                derivation_status=derivation_status,
                identity_evidence=evidence,
                verification_status=verification,
                formal_split_eligible=verified,
                ineligibility_reason=reason,
                dataset_role="geometry_validation_smoke",
                metadata={
                    "grouping_basis": str(group.get("grouping_basis", "")),
                    "source_group_review_required": bool(
                        group.get("source_group_review_required", True)
                    ),
                    "review_reason": str(group.get("review_reason", "")),
                },
            )
        )
    return tuple(output)


def _hash_split(group_id: str, ratios: Mapping[str, float], seed: int) -> str:
    total = sum(float(value) for value in ratios.values())
    if abs(total - 1.0) > 1e-9 or any(float(value) < 0 for value in ratios.values()):
        raise ValueError("Formal split ratios must be non-negative and sum to one")
    digest = hashlib.sha256(f"{seed}:{group_id}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    cumulative = 0.0
    for split in ("train", "validation", "test"):
        cumulative += float(ratios.get(split, 0.0))
        if value < cumulative:
            return split
    return "test"


def plan_verified_formal_splits(
    lineage: Iterable[SourceLineageRecord],
    *,
    official_split_by_video: Mapping[str, str] | None = None,
    ratios: Mapping[str, float] | None = None,
    random_seed: int = 20260722,
) -> tuple[FormalSplitRecord, ...]:
    """Prefer official splits, otherwise assign complete verified source groups."""

    rows = tuple(lineage)
    official = dict(official_split_by_video or {})
    ratios = ratios or {"train": 0.7, "validation": 0.15, "test": 0.15}
    groups: dict[tuple[str, str], list[SourceLineageRecord]] = defaultdict(list)
    for row in rows:
        group_key = row.original_source_id or row.source_group_id
        groups[(row.dataset_name, group_key)].append(row)
    decisions: dict[tuple[str, str], tuple[str, str, bool, str]] = {}
    for key, members in sorted(groups.items()):
        if not all(row.formal_split_eligible for row in members):
            decisions[key] = ("", "blocked_unverified_lineage", False, "unverified_source_lineage")
            continue
        official_values = set()
        for row in members:
            if row.derived_video_id not in official:
                continue
            value = str(official[row.derived_video_id]).strip().lower()
            value = {"val": "validation", "valid": "validation", "dev": "validation"}.get(
                value, value
            )
            if value:
                official_values.add(value)
        if len(official_values) > 1:
            decisions[key] = ("", "official_split_conflict", False, "source_group_official_split_conflict")
        elif official_values:
            split = next(iter(official_values))
            if split not in {"train", "validation", "test"}:
                decisions[key] = ("", "unsupported_official_split", False, "unsupported_official_split")
            else:
                decisions[key] = (split, "official", True, "")
        else:
            decisions[key] = (
                _hash_split(f"{key[0]}:{key[1]}", ratios, random_seed),
                "group_aware_deterministic",
                False,
                "",
            )
    output = []
    for row in sorted(rows, key=lambda item: (item.dataset_name, item.derived_video_id)):
        key = (row.dataset_name, row.original_source_id or row.source_group_id)
        split, source, preserved, reason = decisions[key]
        output.append(
            FormalSplitRecord(
                dataset_name=row.dataset_name,
                derived_video_id=row.derived_video_id,
                original_source_id=row.original_source_id,
                source_group_id=row.source_group_id,
                formal_split=split,
                split_source=source,
                eligible=bool(split),
                blocked_reason=reason,
                official_split_preserved=preserved,
                algorithm_version=FORMAL_SPLIT_ALGORITHM,
                random_seed=random_seed,
            )
        )
    return tuple(output)


def audit_formal_split_leakage(
    lineage: Iterable[SourceLineageRecord],
    splits: Iterable[FormalSplitRecord],
) -> dict[str, Any]:
    """Audit verified origin, group, derived-video, and exact-hash split isolation."""

    lineage_rows = tuple(lineage)
    split_rows = tuple(splits)
    findings: list[dict[str, Any]] = []
    by_video: dict[str, list[FormalSplitRecord]] = defaultdict(list)
    for row in split_rows:
        by_video[row.derived_video_id].append(row)
    for video_id, members in sorted(by_video.items()):
        values = {row.formal_split for row in members if row.formal_split}
        if len(values) > 1:
            findings.append({"type": "derived_video_cross_split", "video_id": video_id})
    split_by_video = {row.derived_video_id: row.formal_split for row in split_rows}
    for field_name, finding_type in (
        ("original_source_id", "original_source_cross_split"),
        ("source_group_id", "source_group_cross_split"),
        ("source_sha256", "exact_hash_cross_split"),
    ):
        grouped: dict[str, set[str]] = defaultdict(set)
        for row in lineage_rows:
            key = str(getattr(row, field_name))
            split = split_by_video.get(row.derived_video_id, "")
            if key and split:
                grouped[key].add(split)
        for identity, values in sorted(grouped.items()):
            if len(values) > 1:
                findings.append(
                    {"type": finding_type, "identity": identity, "splits": sorted(values)}
                )
    findings.sort(key=lambda item: (str(item["type"]), str(item.get("identity", ""))))
    return {
        "formal_split_count": sum(bool(row.formal_split) for row in split_rows),
        "blocked_split_count": sum(not bool(row.formal_split) for row in split_rows),
        "verified_lineage_count": sum(row.verification_status == "verified" for row in lineage_rows),
        "unverified_lineage_count": sum(row.verification_status == "unverified" for row in lineage_rows),
        "cross_split_leakage_detected": bool(findings),
        "finding_count": len(findings),
        "findings": findings,
        "residual_values_used": False,
        "authenticity_performance_read": False,
    }
