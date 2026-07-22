"""Cumulative eligibility funnels for residual branches."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class EligibilityRecord:
    """One candidate unit's progress through a residual computation funnel."""

    branch_name: str
    unit_id: str
    applicable: bool
    input_ready: bool
    attempted: bool
    valid: bool
    provider_failed: bool = False
    blocked: bool = False
    not_applicable: bool = False
    reason: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.branch_name or not self.unit_id:
            raise ValueError("EligibilityRecord requires branch_name and unit_id.")
        if self.valid and not (self.applicable and self.input_ready and self.attempted):
            raise ValueError("Valid evidence must pass every cumulative funnel stage.")
        if self.attempted and not (self.applicable and self.input_ready):
            raise ValueError("attempted requires applicable and input_ready.")
        if self.input_ready and not self.applicable:
            raise ValueError("input_ready requires applicable.")
        terminal = sum(
            bool(value)
            for value in (self.valid, self.provider_failed, self.blocked, self.not_applicable)
        )
        if terminal != 1:
            raise ValueError("Exactly one terminal outcome must be true.")
        if self.not_applicable and self.applicable:
            raise ValueError("not_applicable cannot also be applicable.")
        if (self.provider_failed or self.blocked or self.not_applicable) and not self.reason:
            raise ValueError("Unavailable funnel outcomes require a reason.")
        object.__setattr__(self, "metadata", dict(self.metadata))


def summarize_eligibility(
    records: Sequence[EligibilityRecord],
) -> dict[str, dict[str, object]]:
    """Return deterministic cumulative and terminal counts per branch."""

    output: dict[str, dict[str, object]] = {}
    for branch_name in sorted({record.branch_name for record in records}):
        branch = [record for record in records if record.branch_name == branch_name]
        reasons: dict[str, int] = {}
        for record in branch:
            if record.reason:
                reasons[record.reason] = reasons.get(record.reason, 0) + 1
        output[branch_name] = {
            "unit": str(branch[0].metadata.get("unit", "candidate")),
            "total": len(branch),
            "applicable": sum(record.applicable for record in branch),
            "input_ready": sum(record.input_ready for record in branch),
            "attempted": sum(record.attempted for record in branch),
            "valid": sum(record.valid for record in branch),
            "provider_failed": sum(record.provider_failed for record in branch),
            "blocked": sum(record.blocked for record in branch),
            "not_applicable": sum(record.not_applicable for record in branch),
            "reason_counts": dict(sorted(reasons.items())),
        }
    return output
