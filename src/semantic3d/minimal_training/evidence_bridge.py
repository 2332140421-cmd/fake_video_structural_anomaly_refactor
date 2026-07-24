"""Deterministic M6 audit-table to A2 evidence-manifest conversion.

The bridge reads only branch-level M6 contribution and availability audits.
The deterministic fusion score is deliberately outside this interface.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from semantic3d.dataset_builder.formal_schema import FormalVideoSample
from semantic3d.evidence_fusion.contracts import provider_status_is_failure

from .contracts import (
    EVIDENCE_MANIFEST_SCHEMA_VERSION,
    FeatureContract,
    load_feature_contract,
    sha256_file,
)
from .data import _coerce_formal_row, _read_records, _unique_by_sample_id

EVIDENCE_BRIDGE_SCHEMA_VERSION = "semantic3d.p4c3c.a3.evidence_bridge.v1"
EVIDENCE_BRIDGE_CONFIG_SCHEMA_VERSION = (
    "semantic3d.p4c3c.a3.evidence_bridge_config.v1"
)


@dataclass(frozen=True)
class EvidenceBridgeResult:
    """Summary of one completed deterministic conversion."""

    output_path: Path
    sample_count: int
    output_sha256: str
    input_sha256: Mapping[str, str]
    branch_order: tuple[str, ...]


def _nonempty(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _strict_bool(value: Any, *, field: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = _nonempty(value).lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"{field} must be an explicit boolean, got {value!r}.")


def _optional_int(value: Any, *, field: str) -> int | None:
    if not _nonempty(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer, got {value!r}.") from exc


def _bounded_float(value: Any, *, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} cannot be boolean.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric, got {value!r}.") from exc
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{field} must be finite and in [0, 1].")
    return number


def _json_mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if value is None or value == "":
        return {}
    decoded = json.loads(value) if isinstance(value, str) else value
    if not isinstance(decoded, Mapping):
        raise ValueError(f"{field} must be a JSON object.")
    return decoded


def _load_formal_samples(path: Path) -> dict[str, Mapping[str, Any]]:
    samples = [
        FormalVideoSample(**_coerce_formal_row(row)).to_dict()
        for row in _read_records(path)
    ]
    return dict(
        _unique_by_sample_id(samples, manifest_name=str(path))
    )


def _load_bridge_config(path: Path | None) -> tuple[Mapping[str, Any], str]:
    if path is None:
        return {}, ""
    import yaml

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Evidence bridge config must be a YAML mapping.")
    if payload.get("schema_version") != EVIDENCE_BRIDGE_CONFIG_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported evidence bridge config schema_version: "
            f"{payload.get('schema_version')!r}."
        )
    if payload.get("fit_statistics") is not False:
        raise ValueError("Evidence bridge must not fit statistics.")
    if payload.get("select_threshold") is not False:
        raise ValueError("Evidence bridge must not select thresholds.")
    if payload.get("read_deterministic_fusion_score") is not False:
        raise ValueError("Evidence bridge must not read deterministic fusion scores.")
    return payload, sha256_file(path)


@dataclass(frozen=True)
class _JoinResolver:
    by_source_key: Mapping[tuple[str, str], str]
    formal_ids: frozenset[str]
    split_by_sample_id: Mapping[str, str]
    mapping_sha256: str

    @classmethod
    def build(
        cls,
        formal_by_id: Mapping[str, Mapping[str, Any]],
        mapping_path: Path | None,
    ) -> "_JoinResolver":
        formal_ids = frozenset(formal_by_id)
        split_by_id = {
            sample_id: _nonempty(row.get("split"))
            for sample_id, row in formal_by_id.items()
        }
        if mapping_path is None:
            return cls({}, formal_ids, split_by_id, "")

        by_key: dict[tuple[str, str], str] = {}
        seen_sample_ids: set[str] = set()
        for row in _read_records(mapping_path):
            sample_id = _nonempty(row.get("sample_id"))
            video_id = _nonempty(row.get("video_id"))
            clip_id = _nonempty(row.get("clip_id"))
            split = _nonempty(row.get("split"))
            if not sample_id or not video_id or not clip_id or not split:
                raise ValueError(
                    "Every sample mapping row requires sample_id, video_id, "
                    "clip_id, and split."
                )
            if sample_id in seen_sample_ids:
                raise ValueError(f"Duplicate sample_id {sample_id!r} in sample mapping.")
            if sample_id not in formal_by_id:
                raise ValueError(
                    f"Unknown formal sample_id {sample_id!r} in sample mapping."
                )
            if split != split_by_id[sample_id]:
                raise ValueError(
                    f"Split mismatch for {sample_id}: mapping={split!r}, "
                    f"formal={split_by_id[sample_id]!r}."
                )
            key = (video_id, clip_id)
            if key in by_key:
                raise ValueError(f"Duplicate M6 source key in sample mapping: {key!r}.")
            by_key[key] = sample_id
            seen_sample_ids.add(sample_id)
        if seen_sample_ids != formal_ids:
            missing = sorted(formal_ids - seen_sample_ids)
            raise ValueError(
                f"Sample mapping does not cover every formal sample: missing={missing}."
            )
        return cls(by_key, formal_ids, split_by_id, sha256_file(mapping_path))

    def resolve(self, row: Mapping[str, Any], *, source: str) -> str:
        direct_id = _nonempty(row.get("sample_id"))
        if direct_id:
            if self.by_source_key:
                key = (
                    _nonempty(row.get("video_id")),
                    _nonempty(row.get("clip_id")),
                )
                mapped_id = self.by_source_key.get(key)
                if mapped_id is None:
                    raise ValueError(
                        f"{source} row has unmapped M6 source key {key!r}."
                    )
                if direct_id != mapped_id:
                    raise ValueError(
                        f"{source} sample_id {direct_id!r} disagrees with "
                        f"mapping sample_id {mapped_id!r}."
                    )
            sample_id = direct_id
        else:
            if not self.by_source_key:
                raise ValueError(
                    f"{source} rows without sample_id require an explicit "
                    "sample mapping manifest."
                )
            key = (
                _nonempty(row.get("video_id")),
                _nonempty(row.get("clip_id")),
            )
            sample_id = self.by_source_key.get(key, "")
            if not sample_id:
                raise ValueError(
                    f"{source} row has unmapped M6 source key {key!r}."
                )
        if sample_id not in self.formal_ids:
            raise ValueError(f"{source} contains unknown sample_id {sample_id!r}.")
        evidence_split = _nonempty(row.get("split"))
        if evidence_split and evidence_split != self.split_by_sample_id[sample_id]:
            raise ValueError(
                f"Split mismatch for {sample_id}: {source}={evidence_split!r}, "
                f"formal={self.split_by_sample_id[sample_id]!r}."
            )
        return sample_id


def _validate_branch(
    row: Mapping[str, Any],
    *,
    allowed: frozenset[str],
    source: str,
) -> str:
    branch = _nonempty(row.get("branch_group"))
    if branch not in allowed:
        raise ValueError(f"{source} contains unknown branch_group {branch!r}.")
    return branch


def _read_contributions(
    path: Path,
    *,
    resolver: _JoinResolver,
    branch_order: Sequence[str],
) -> tuple[dict[tuple[str, str], Mapping[str, Any]], set[str]]:
    allowed = frozenset(branch_order)
    output: dict[tuple[str, str], Mapping[str, Any]] = {}
    covered: set[str] = set()
    for row in _read_records(path):
        sample_id = resolver.resolve(row, source="branch contribution manifest")
        branch = _validate_branch(
            row, allowed=allowed, source="branch contribution manifest"
        )
        key = (sample_id, branch)
        if key in output:
            raise ValueError(
                f"Duplicate branch contribution for {sample_id}/{branch}."
            )
        output[key] = {
            "bounded_risk": _bounded_float(
                row.get("bounded_risk"),
                field=f"{sample_id}/{branch} bounded_risk",
            ),
            "confidence": _bounded_float(
                row.get("confidence"),
                field=f"{sample_id}/{branch} confidence",
            ),
        }
        covered.add(sample_id)
    return output, covered


def _read_availability(
    path: Path,
    *,
    resolver: _JoinResolver,
    branch_order: Sequence[str],
) -> tuple[
    dict[tuple[str, str], Mapping[str, Any]],
    dict[tuple[str, str], Mapping[str, Any]],
    set[str],
]:
    allowed = frozenset(branch_order)
    summary: dict[tuple[str, str], Mapping[str, Any]] = {}
    routes: dict[tuple[str, str], Mapping[str, Any]] = {}
    covered: set[str] = set()
    for row in _read_records(path):
        sample_id = resolver.resolve(row, source="branch availability manifest")
        branch = _validate_branch(
            row, allowed=allowed, source="branch availability manifest"
        )
        key = (sample_id, branch)
        is_summary = bool(_nonempty(row.get("total_evidence")))
        is_route = bool(_nonempty(row.get("route_status")))
        if is_summary == is_route:
            raise ValueError(
                f"Availability row for {sample_id}/{branch} must be exactly "
                "one M6 summary row or one route row."
            )
        target = summary if is_summary else routes
        if key in target:
            kind = "summary" if is_summary else "route"
            raise ValueError(
                f"Duplicate availability {kind} for {sample_id}/{branch}."
            )
        target[key] = row
        covered.add(sample_id)
    return summary, routes, covered


def _reason_from_counts(value: Any) -> str:
    counts = _json_mapping(value, field="missing_reason_counts")
    present = [
        (str(reason), int(count))
        for reason, count in counts.items()
        if str(reason) and int(count) > 0
    ]
    if not present:
        return ""
    return ";".join(f"{reason}:{count}" for reason, count in sorted(present))


def _provider_failure_only(row: Mapping[str, Any]) -> bool:
    failures = _optional_int(
        row.get("provider_failure_count"), field="provider_failure_count"
    )
    if failures is not None and failures > 0:
        return True
    counts = _json_mapping(
        row.get("provider_status_counts"), field="provider_status_counts"
    )
    return any(
        int(count) > 0 and provider_status_is_failure(str(status))
        for status, count in counts.items()
    )


def _missing_branch(
    *,
    sample_id: str,
    branch: str,
    summary: Mapping[str, Any] | None,
    route: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    route_status = "" if route is None else _nonempty(route.get("route_status"))
    route_reason = "" if route is None else _nonempty(route.get("route_reason"))
    route_applicable = (
        None
        if route is None
        else _strict_bool(
            route.get("route_applicable"),
            field=f"{sample_id}/{branch} route_applicable",
        )
    )
    if route_status and route_status not in {
        "enabled",
        "not_applicable",
        "blocked_by_input",
    }:
        raise ValueError(
            f"Unsupported M6 route_status {route_status!r} for "
            f"{sample_id}/{branch}."
        )
    if route_status == "not_applicable":
        status, observable = "not_applicable", False
        reason = route_reason or "m6_route_not_applicable"
    elif route_status == "blocked_by_input":
        status, observable = "blocked_by_input", bool(route_applicable)
        reason = route_reason or "m6_route_blocked_by_input"
    elif summary is not None:
        applicable = _optional_int(
            summary.get("applicable_evidence"), field="applicable_evidence"
        )
        valid = _optional_int(
            summary.get("valid_evidence"), field="valid_evidence"
        )
        summary_reason = _reason_from_counts(
            summary.get("missing_reason_counts")
        )
        if valid is not None and valid > 0:
            status = "missing"
            observable = (
                bool(route_applicable)
                if route_applicable is not None
                else bool(applicable)
            )
            reason = summary_reason or "m6_branch_contribution_unavailable"
        elif _provider_failure_only(summary):
            status, observable = "provider_failed", False
            reason = summary_reason or "m6_provider_failed"
        elif applicable == 0:
            status, observable = "not_applicable", False
            reason = summary_reason or "m6_no_applicable_evidence"
        else:
            status = "missing"
            observable = (
                bool(route_applicable)
                if route_applicable is not None
                else bool(applicable)
            )
            reason = summary_reason or "m6_branch_evidence_missing"
    else:
        status = "missing"
        observable = bool(route_applicable)
        reason = route_reason or "branch_not_emitted_by_m6"
    return {
        "bounded_risk": None,
        "feature_available": False,
        "observability": observable,
        "confidence": None,
        "missing_reason": reason,
        "status": status,
        "observable": observable,
    }


def _observed_branch(
    *,
    sample_id: str,
    branch: str,
    contribution: Mapping[str, Any],
    summary: Mapping[str, Any] | None,
    route: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    if route is not None:
        route_status = _nonempty(route.get("route_status"))
        route_applicable = _strict_bool(
            route.get("route_applicable"),
            field=f"{sample_id}/{branch} route_applicable",
        )
        if route_status != "enabled" or not route_applicable:
            raise ValueError(
                f"Observed contribution conflicts with M6 route for "
                f"{sample_id}/{branch}: status={route_status!r}, "
                f"applicable={route_applicable!r}."
            )
    if summary is not None:
        valid = _optional_int(
            summary.get("valid_evidence"), field="valid_evidence"
        )
        if valid is not None and valid <= 0:
            raise ValueError(
                f"Observed contribution conflicts with zero valid M6 evidence "
                f"for {sample_id}/{branch}."
            )
        if _provider_failure_only(summary) and (valid is None or valid <= 0):
            raise ValueError(
                f"Observed contribution conflicts with provider failure for "
                f"{sample_id}/{branch}."
            )
    return {
        "bounded_risk": contribution["bounded_risk"],
        "feature_available": True,
        "observability": True,
        "confidence": contribution["confidence"],
        "missing_reason": None,
        "status": "observed",
        "observable": True,
    }


def build_a2_evidence_manifest(
    *,
    formal_manifest: str | Path,
    branch_contribution_manifest: str | Path,
    branch_availability_manifest: str | Path,
    feature_contract: str | Path | FeatureContract,
    output_path: str | Path,
    sample_mapping_manifest: str | Path | None = None,
    bridge_config: str | Path | None = None,
) -> EvidenceBridgeResult:
    """Convert M6 branch audits into an A2-readable deterministic JSONL file."""

    formal_path = Path(formal_manifest)
    contribution_path = Path(branch_contribution_manifest)
    availability_path = Path(branch_availability_manifest)
    mapping_path = (
        None if sample_mapping_manifest is None else Path(sample_mapping_manifest)
    )
    config_path = None if bridge_config is None else Path(bridge_config)
    output = Path(output_path)
    if output.suffix.lower() != ".jsonl":
        raise ValueError("A3 evidence bridge output must use the .jsonl format.")

    contract = (
        feature_contract
        if isinstance(feature_contract, FeatureContract)
        else load_feature_contract(feature_contract)
    )
    _, config_sha = _load_bridge_config(config_path)
    formal_by_id = _load_formal_samples(formal_path)
    resolver = _JoinResolver.build(formal_by_id, mapping_path)
    contributions, contribution_coverage = _read_contributions(
        contribution_path,
        resolver=resolver,
        branch_order=contract.branch_order,
    )
    summaries, routes, availability_coverage = _read_availability(
        availability_path,
        resolver=resolver,
        branch_order=contract.branch_order,
    )
    covered = contribution_coverage | availability_coverage
    if covered != resolver.formal_ids:
        missing = sorted(resolver.formal_ids - covered)
        raise ValueError(
            f"Formal/M6 sample_id join is incomplete: missing_evidence={missing}."
        )

    input_sha = {
        "formal_manifest": sha256_file(formal_path),
        "branch_contribution_manifest": sha256_file(contribution_path),
        "branch_availability_manifest": sha256_file(availability_path),
        "feature_contract": contract.sha256,
    }
    if resolver.mapping_sha256:
        input_sha["sample_mapping_manifest"] = resolver.mapping_sha256
    if config_sha:
        input_sha["bridge_config"] = config_sha

    rows: list[Mapping[str, Any]] = []
    for sample_id in sorted(formal_by_id):
        formal = formal_by_id[sample_id]
        split = _nonempty(formal.get("split"))
        if not split:
            raise ValueError(f"Formal sample {sample_id!r} has no explicit split.")
        branches: dict[str, Mapping[str, Any]] = {}
        for branch in contract.branch_order:
            key = (sample_id, branch)
            contribution = contributions.get(key)
            if contribution is None:
                branches[branch] = _missing_branch(
                    sample_id=sample_id,
                    branch=branch,
                    summary=summaries.get(key),
                    route=routes.get(key),
                )
            else:
                branches[branch] = _observed_branch(
                    sample_id=sample_id,
                    branch=branch,
                    contribution=contribution,
                    summary=summaries.get(key),
                    route=routes.get(key),
                )
        rows.append(
            {
                "schema_version": EVIDENCE_MANIFEST_SCHEMA_VERSION,
                "feature_contract_version": contract.version,
                "sample_id": sample_id,
                "split": split,
                "branches": branches,
                "label": formal.get("label"),
                "source_dataset": formal.get("source_dataset"),
                "generator": formal.get("generator"),
                "source_lineage": formal.get("source_lineage"),
                "metadata": {
                    "bridge_schema_version": EVIDENCE_BRIDGE_SCHEMA_VERSION,
                    "source_schema_version": contract.source_schema_version,
                    "input_sha256": input_sha,
                    "branch_order": list(contract.branch_order),
                    "deterministic_fusion_score_read": False,
                    "provider_failure_encoded_as_feature": False,
                    "normalization_fitted": False,
                    "threshold_selected": False,
                    "label_source": "formal_manifest_only",
                },
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    return EvidenceBridgeResult(
        output_path=output,
        sample_count=len(rows),
        output_sha256=sha256_file(output),
        input_sha256=input_sha,
        branch_order=contract.branch_order,
    )


__all__ = [
    "EVIDENCE_BRIDGE_CONFIG_SCHEMA_VERSION",
    "EVIDENCE_BRIDGE_SCHEMA_VERSION",
    "EvidenceBridgeResult",
    "build_a2_evidence_manifest",
]
