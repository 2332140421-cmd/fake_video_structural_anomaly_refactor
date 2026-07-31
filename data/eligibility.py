"""Collect-all residual eligibility inspection, independent from leakage."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

MODEL_ELIGIBLE_STATUSES = ("eligible", "partial_missing")
ELIGIBILITY_STATUSES = (
    "eligible",
    "partial_missing",
    "no_valid_residual",
    "malformed",
    "missing_file",
    "schema_error",
)
NO_VALID_RESIDUAL_POLICIES = ("error", "exclude", "keep_empty")


@dataclass(frozen=True)
class ResidualEligibilityReport:
    sample_id: str
    split: str
    label: int
    residual_path: str
    clip_count: int
    residual_record_count: int
    valid_count: int
    observed_count: int
    blocked_by_input_count: int
    not_applicable_count: int
    eligibility_status: str
    exclusion_reason: str
    model_eligible: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResidualEligibilityInspection:
    report: ResidualEligibilityReport
    payload: Mapping[str, Any] | None
    clip_ids: tuple[str, ...]


def normalize_no_valid_residual_policy(value: str | None) -> str:
    policy = str(value or "error").strip().lower()
    if policy not in NO_VALID_RESIDUAL_POLICIES:
        raise ValueError(f"Unsupported no_valid_residual policy: {policy!r}.")
    return policy


def _report(
    *,
    sample_id: str,
    split: str,
    label: int,
    residual_path: Path,
    status: str,
    reason: str,
    clip_count: int = 0,
    residual_record_count: int = 0,
    valid_count: int = 0,
    observed_count: int = 0,
    blocked_by_input_count: int = 0,
    not_applicable_count: int = 0,
) -> ResidualEligibilityReport:
    return ResidualEligibilityReport(
        sample_id=sample_id,
        split=split,
        label=label,
        residual_path=str(residual_path),
        clip_count=clip_count,
        residual_record_count=residual_record_count,
        valid_count=valid_count,
        observed_count=observed_count,
        blocked_by_input_count=blocked_by_input_count,
        not_applicable_count=not_applicable_count,
        eligibility_status=status,
        exclusion_reason=reason,
        model_eligible=status in MODEL_ELIGIBLE_STATUSES,
    )


def inspect_residual_eligibility(
    *,
    sample_id: str,
    split: str,
    label: int,
    residual_path: str | Path,
    channel_names: Sequence[str],
) -> ResidualEligibilityInspection:
    """Inspect one result without raising, so a manifest audit can collect all."""

    path = Path(residual_path).expanduser().resolve()
    if not path.is_file():
        return ResidualEligibilityInspection(
            _report(
                sample_id=sample_id,
                split=split,
                label=label,
                residual_path=path,
                status="missing_file",
                reason="residual_file_missing",
            ),
            None,
            (),
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        return ResidualEligibilityInspection(
            _report(
                sample_id=sample_id,
                split=split,
                label=label,
                residual_path=path,
                status="malformed",
                reason=f"{type(error).__name__}: {error}",
            ),
            None,
            (),
        )
    if not isinstance(payload, Mapping):
        return ResidualEligibilityInspection(
            _report(
                sample_id=sample_id,
                split=split,
                label=label,
                residual_path=path,
                status="malformed",
                reason="result_root_must_be_an_object",
            ),
            None,
            (),
        )
    clips = payload.get("clips")
    if not isinstance(clips, list) or not clips:
        return ResidualEligibilityInspection(
            _report(
                sample_id=sample_id,
                split=split,
                label=label,
                residual_path=path,
                status="schema_error",
                reason="clips_must_be_a_non_empty_list",
            ),
            payload,
            (),
        )

    allowed_names = set(channel_names)
    clip_ids: list[str] = []
    residual_record_count = 0
    valid_count = 0
    observed_count = 0
    blocked_count = 0
    not_applicable_count = 0
    schema_errors: list[str] = []
    for clip_index, clip in enumerate(clips):
        if not isinstance(clip, Mapping):
            schema_errors.append(f"clip[{clip_index}]_must_be_an_object")
            continue
        clip_id = str(clip.get("clip_id", "") or "").strip()
        if not clip_id:
            schema_errors.append(f"clip[{clip_index}]_missing_clip_id")
        else:
            clip_ids.append(clip_id)
        try:
            int(clip["start_frame"])
        except (KeyError, TypeError, ValueError):
            schema_errors.append(f"clip[{clip_index}]_invalid_start_frame")
        residuals = clip.get("residuals")
        if not isinstance(residuals, list):
            schema_errors.append(f"clip[{clip_index}]_residuals_must_be_a_list")
            continue
        for row_index, row in enumerate(residuals):
            residual_record_count += 1
            if not isinstance(row, Mapping):
                schema_errors.append(
                    f"clip[{clip_index}].residuals[{row_index}]_must_be_an_object"
                )
                continue
            name = str(row.get("name", "") or "")
            if name not in allowed_names:
                schema_errors.append(
                    f"unknown residual channel {name!r} at "
                    f"clip[{clip_index}].residuals[{row_index}]"
                )
            availability = str(row.get("availability", "") or "")
            observed_count += int(availability == "observed")
            blocked_count += int(availability == "blocked_by_input")
            not_applicable_count += int(availability == "not_applicable")
            valid = bool(row.get("valid_mask", False))
            if not valid:
                continue
            valid_count += 1
            if availability != "observed":
                schema_errors.append(
                    f"clip[{clip_index}].residuals[{row_index}]_valid_not_observed"
                )
            try:
                value = float(row["normalized_value"])
                confidence = float(row["confidence"])
            except (KeyError, TypeError, ValueError):
                schema_errors.append(
                    f"clip[{clip_index}].residuals[{row_index}]_invalid_value"
                )
                continue
            if not math.isfinite(value) or not math.isfinite(confidence):
                schema_errors.append(
                    f"clip[{clip_index}].residuals[{row_index}]_nonfinite_value"
                )
            if not 0.0 <= confidence <= 1.0:
                schema_errors.append(
                    f"clip[{clip_index}].residuals[{row_index}]_confidence_out_of_range"
                )

    common = {
        "sample_id": sample_id,
        "split": split,
        "label": label,
        "residual_path": path,
        "clip_count": len(clips),
        "residual_record_count": residual_record_count,
        "valid_count": valid_count,
        "observed_count": observed_count,
        "blocked_by_input_count": blocked_count,
        "not_applicable_count": not_applicable_count,
    }
    if schema_errors:
        report = _report(
            **common,
            status="schema_error",
            reason="; ".join(schema_errors[:10]),
        )
    elif valid_count == 0:
        report = _report(
            **common,
            status="no_valid_residual",
            reason="all_residual_records_unavailable",
        )
    elif valid_count < residual_record_count:
        report = _report(
            **common,
            status="partial_missing",
            reason="",
        )
    else:
        report = _report(
            **common,
            status="eligible",
            reason="",
        )
    return ResidualEligibilityInspection(report, payload, tuple(clip_ids))


def build_eligible_split(
    reports: Sequence[ResidualEligibilityReport],
    split: str,
) -> tuple[str, ...]:
    return tuple(
        report.sample_id
        for report in reports
        if report.split == split and report.model_eligible
    )


def summarize_eligibility(
    reports: Sequence[ResidualEligibilityReport],
) -> dict[str, Any]:
    output: dict[str, Any] = {"total": len(reports), "splits": {}}
    for split in ("train", "validation", "test"):
        selected = [report for report in reports if report.split == split]
        eligible_count = sum(report.model_eligible for report in selected)
        status_counts = {
            status: sum(
                report.eligibility_status == status for report in selected
            )
            for status in ELIGIBILITY_STATUSES
        }
        output["splits"][split] = {
            "original_count": len(selected),
            "eligible_count": eligible_count,
            "excluded_count": len(selected) - eligible_count,
            "coverage": eligible_count / max(len(selected), 1),
            "status_counts": status_counts,
        }
    output["status_counts"] = {
        status: sum(report.eligibility_status == status for report in reports)
        for status in ELIGIBILITY_STATUSES
    }
    output["eligible_count"] = sum(report.model_eligible for report in reports)
    output["excluded_count"] = len(reports) - output["eligible_count"]
    return output


__all__ = [
    "ELIGIBILITY_STATUSES",
    "MODEL_ELIGIBLE_STATUSES",
    "NO_VALID_RESIDUAL_POLICIES",
    "ResidualEligibilityInspection",
    "ResidualEligibilityReport",
    "build_eligible_split",
    "inspect_residual_eligibility",
    "normalize_no_valid_residual_policy",
    "summarize_eligibility",
]
