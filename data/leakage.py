"""Explicit clip identity and configurable cross-split leakage auditing."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

SPLITS = ("train", "validation", "test")
LEAKAGE_MODES = ("warn", "strict")
LEAKAGE_SCOPES = ("clip", "source")


@dataclass(frozen=True)
class LeakagePolicy:
    enabled: bool = True
    mode: str = "strict"
    scope: str = "source"


def normalize_leakage_policy(
    value: Mapping[str, Any] | LeakagePolicy | None,
) -> LeakagePolicy:
    if value is None:
        return LeakagePolicy()
    if isinstance(value, LeakagePolicy):
        policy = value
    else:
        policy = LeakagePolicy(
            enabled=bool(value.get("enabled", True)),
            mode=str(value.get("mode", "strict")).strip().lower(),
            scope=str(value.get("scope", "source")).strip().lower(),
        )
    if policy.mode not in LEAKAGE_MODES:
        raise ValueError(f"Unsupported leakage mode: {policy.mode!r}.")
    if policy.scope not in LEAKAGE_SCOPES:
        raise ValueError(f"Unsupported leakage scope: {policy.scope!r}.")
    return policy


def build_clip_uid(sample_id: str, clip_id: str) -> str:
    """Build a globally comparable clip identity from its owning sample."""

    sample = str(sample_id).strip()
    clip = str(clip_id).strip()
    if not sample or not clip:
        raise ValueError("sample_id and clip_id must be non-empty.")
    return f"{sample}::{clip}"


def resolve_leakage_key(
    record: Mapping[str, Any],
    key: str,
) -> tuple[str, ...]:
    """Resolve one audit key without assigning global meaning to local clip_id."""

    def one(name: str) -> tuple[str, ...]:
        value = str(record.get(name, "") or "").strip()
        return (value,) if value else ()

    if key == "clip_uid":
        sample_id = str(record.get("sample_id", "") or "").strip()
        return tuple(
            build_clip_uid(sample_id, clip_id)
            for clip_id in record.get("clip_ids", ())
        )
    if key == "source_video_id":
        dataset = str(record.get("dataset_name", "") or "").strip()
        source = str(record.get("source_video_id", "") or "").strip()
        return (f"{dataset}::{source}",) if dataset and source else ()
    if key == "group_id":
        dataset = str(record.get("dataset_name", "") or "").strip()
        group = str(record.get("group_id", "") or "").strip()
        return (f"{dataset}::{group}",) if dataset and group else ()
    return one(key)


def audit_split_leakage(
    records: Sequence[Mapping[str, Any]],
    policy: Mapping[str, Any] | LeakagePolicy | None = None,
) -> dict[str, Any]:
    """Audit global clip and source identities across formal splits."""

    normalized = normalize_leakage_policy(policy)
    keys = ["sample_id", "clip_uid", "residual_sequence_path"]
    if normalized.scope == "source":
        keys.extend(
            (
                "source_video_sha256",
                "source_video_id",
                "group_id",
                "source_video_path",
            )
        )
    values: dict[str, dict[str, set[str]]] = {
        split: {key: set() for key in keys} for split in SPLITS
    }
    for record in records:
        split = str(record.get("split", "")).strip()
        if split not in values:
            raise ValueError(f"Unsupported split {split!r}.")
        for key in keys:
            values[split][key].update(resolve_leakage_key(record, key))

    findings: list[dict[str, Any]] = []
    if normalized.enabled:
        for left_index, left in enumerate(SPLITS):
            for right in SPLITS[left_index + 1 :]:
                for key in keys:
                    overlaps = sorted(values[left][key] & values[right][key])
                    if overlaps:
                        findings.append(
                            {
                                "key": key,
                                "left_split": left,
                                "right_split": right,
                                "overlap_count": len(overlaps),
                                "examples": overlaps[:5],
                            }
                        )

    result = {
        "enabled": normalized.enabled,
        "mode": normalized.mode,
        "scope": normalized.scope,
        "status": "FAIL" if findings else "PASS",
        "finding_count": len(findings),
        "findings": findings,
        "audited_keys": keys,
        "clip_identity_contract": "clip_uid=sample_id::clip_id",
    }
    if findings:
        first = findings[0]
        message = (
            f"Cross-split leakage detected for {first['key']}: "
            f"{first['left_split']}/{first['right_split']} "
            f"({first['overlap_count']} overlap(s))."
        )
        if normalized.mode == "strict":
            raise ValueError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)
    return result


__all__ = [
    "LEAKAGE_MODES",
    "LEAKAGE_SCOPES",
    "LeakagePolicy",
    "audit_split_leakage",
    "build_clip_uid",
    "normalize_leakage_policy",
    "resolve_leakage_key",
]
