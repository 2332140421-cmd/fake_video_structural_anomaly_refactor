"""Coverage and missingness audits that never become authenticity scores."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

from .contracts import (
    EvidenceBranchGroup,
    UnifiedEvidence,
    provider_status_is_failure,
)


def missingness_only_features(
    evidences: Sequence[UnifiedEvidence],
) -> Mapping[str, float | int]:
    """Return availability-only features for future shortcut audits, not prediction."""

    total = len(evidences)
    valid = sum(item.valid for item in evidences)
    applicable = sum(item.applicable for item in evidences)
    failures = sum(provider_status_is_failure(item.provider_status) for item in evidences)
    return {
        "evidence_count": total,
        "applicable_count": applicable,
        "valid_count": valid,
        "valid_ratio": valid / applicable if applicable else 0.0,
        "provider_failure_count": failures,
        "provider_failure_ratio": failures / total if total else 0.0,
        "feature_only": 1,
        "trained": 0,
        "used_in_risk": 0,
    }


def branch_availability_rows(
    evidences: Sequence[UnifiedEvidence],
) -> tuple[Mapping[str, Any], ...]:
    """Summarize availability by video, clip, and branch."""

    grouped: dict[tuple[str, str, EvidenceBranchGroup], list[UnifiedEvidence]] = {}
    for item in evidences:
        grouped.setdefault((item.video_id, item.clip_id, item.branch_group), []).append(
            item
        )
    output = []
    for (video_id, clip_id, group), rows in sorted(
        grouped.items(), key=lambda item: (item[0][0], item[0][1], item[0][2].value)
    ):
        reasons = Counter(row.failure_reason for row in rows if row.failure_reason)
        providers = Counter(row.provider_status for row in rows)
        applicable = sum(row.applicable for row in rows)
        valid = sum(row.valid for row in rows)
        output.append(
            {
                "video_id": video_id,
                "clip_id": clip_id,
                "branch_group": group.value,
                "total_evidence": len(rows),
                "applicable_evidence": applicable,
                "valid_evidence": valid,
                "availability_ratio": valid / applicable if applicable else 0.0,
                "provider_failure_count": sum(
                    provider_status_is_failure(row.provider_status) for row in rows
                ),
                "provider_status_counts": dict(sorted(providers.items())),
                "missing_reason_counts": dict(sorted(reasons.items())),
            }
        )
    return tuple(output)


def coverage_by_group(
    evidences: Sequence[UnifiedEvidence],
    group_by_video: Mapping[str, str],
) -> tuple[Mapping[str, Any], ...]:
    """Audit real/fake or other supplied groups using availability only."""

    grouped: dict[tuple[str, EvidenceBranchGroup], list[UnifiedEvidence]] = {}
    for item in evidences:
        group_name = str(group_by_video.get(item.video_id, "unassigned"))
        grouped.setdefault((group_name, item.branch_group), []).append(item)
    output = []
    for (name, branch), rows in sorted(
        grouped.items(), key=lambda item: (item[0][0], item[0][1].value)
    ):
        applicable = sum(row.applicable for row in rows)
        valid = sum(row.valid for row in rows)
        output.append(
            {
                "coverage_group": name,
                "branch_group": branch.value,
                "evidence_count": len(rows),
                "applicable_count": applicable,
                "valid_count": valid,
                "valid_ratio": valid / applicable if applicable else 0.0,
                "risk_values_read": False,
                "performance_metric_computed": False,
            }
        )
    return tuple(output)


def provider_failure_balance(
    evidences: Sequence[UnifiedEvidence],
    group_by_video: Mapping[str, str],
) -> tuple[Mapping[str, Any], ...]:
    """Count provider failures by supplied group without turning them into risk."""

    grouped: dict[tuple[str, str], list[UnifiedEvidence]] = {}
    for item in evidences:
        key = (
            str(group_by_video.get(item.video_id, "unassigned")),
            item.provider_status,
        )
        grouped.setdefault(key, []).append(item)
    return tuple(
        {
            "coverage_group": group,
            "provider_status": provider,
            "evidence_count": len(rows),
            "provider_failure_count": sum(
                provider_status_is_failure(row.provider_status) for row in rows
            ),
            "provider_failure_used_as_risk": False,
        }
        for (group, provider), rows in sorted(grouped.items())
    )

