"""P4-C3C-A3 M6-to-A2 evidence bridge tests without provider inference."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import pytest
import torch

from semantic3d.dataset_builder.formal_schema import (
    OPTIONAL_METADATA_FIELDS,
    FormalVideoSample,
)
from semantic3d.minimal_training.contracts import load_feature_contract
from semantic3d.minimal_training.data import (
    EvidenceTrainingDataset,
    collate_evidence_samples,
)
from semantic3d.minimal_training.evidence_bridge import (
    EVIDENCE_BRIDGE_SCHEMA_VERSION,
    build_a2_evidence_manifest,
)
from semantic3d.minimal_training.loss import MaskedBinaryLoss
from semantic3d.minimal_training.model import MinimalMissingAwareEvidenceHead

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = PROJECT_ROOT / "configs/p4c3c_a2_m6_feature_contract_v1.yaml"
BRIDGE_CONFIG_PATH = PROJECT_ROOT / "configs/p4c3c_a3_evidence_bridge_v1.yaml"
BRANCHES = load_feature_contract(CONTRACT_PATH).branch_order


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    columns = sorted({name for row in rows for name in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    name: (
                        json.dumps(value, sort_keys=True)
                        if isinstance(value, Mapping)
                        else value
                    )
                    for name, value in row.items()
                }
            )


def _formal_row(
    sample_id: str,
    *,
    split: str = "train",
    label: int | None = 0,
) -> Mapping[str, Any]:
    statuses = {name: "missing" for name in OPTIONAL_METADATA_FIELDS}
    statuses.update(
        {
            "label": "provided" if label is not None else "missing",
            "split": "provided",
            "source_id": "provided",
            "source_lineage": "provided",
            "generator": "provided" if label == 1 else "missing",
            "is_real": "derived" if label is not None else "missing",
        }
    )
    return FormalVideoSample(
        sample_id=sample_id,
        video_path=f"videos/{sample_id}.mp4",
        label=label,
        split=split,
        source_dataset="p4c3c_a3_fixture",
        source_id=f"source:{sample_id}",
        source_lineage={"archive": "fixture.tar", "member": f"{sample_id}.mp4"},
        generator="fixture_generator" if label == 1 else None,
        is_real=None if label is None else label == 0,
        duration=None,
        fps=None,
        frame_count=None,
        width=None,
        height=None,
        file_size=1,
        sha256="0" * 64,
        temporal_annotation=None,
        spatial_annotation=None,
        metadata_status=statuses,
        source_path=f"videos/{sample_id}.mp4",
        path_mode="data_root_relative",
        official_split=True,
    ).to_dict()


def _contribution(
    sample_id: str,
    branch: str,
    *,
    bounded_risk: float = 0.25,
    confidence: float = 0.75,
) -> Mapping[str, Any]:
    return {
        "sample_id": sample_id,
        "split": "train",
        "video_id": f"video:{sample_id}",
        "clip_id": f"clip:{sample_id}",
        "branch_group": branch,
        "bounded_risk": bounded_risk,
        "confidence": confidence,
        "risk_score": 0.999,
    }


def _route(
    sample_id: str,
    branch: str,
    *,
    split: str = "train",
    status: str = "enabled",
    applicable: bool = True,
    reason: str = "",
) -> Mapping[str, Any]:
    return {
        "sample_id": sample_id,
        "split": split,
        "video_id": f"video:{sample_id}",
        "clip_id": f"clip:{sample_id}",
        "branch_group": branch,
        "route_status": status,
        "route_applicable": applicable,
        "route_reason": reason,
    }


def _summary(
    sample_id: str,
    branch: str,
    *,
    applicable: int,
    valid: int,
    provider_failures: int = 0,
    reasons: Mapping[str, int] | None = None,
    provider_statuses: Mapping[str, int] | None = None,
) -> Mapping[str, Any]:
    return {
        "sample_id": sample_id,
        "split": "train",
        "video_id": f"video:{sample_id}",
        "clip_id": f"clip:{sample_id}",
        "branch_group": branch,
        "total_evidence": max(1, applicable),
        "applicable_evidence": applicable,
        "valid_evidence": valid,
        "provider_failure_count": provider_failures,
        "missing_reason_counts": reasons or {},
        "provider_status_counts": provider_statuses or {},
    }


def _build(
    tmp_path: Path,
    *,
    formal_rows: list[Mapping[str, Any]] | None = None,
    contribution_rows: list[Mapping[str, Any]] | None = None,
    availability_rows: list[Mapping[str, Any]] | None = None,
    mapping_rows: list[Mapping[str, Any]] | None = None,
) -> tuple[Path, Path, object]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    formal_rows = formal_rows or [_formal_row("sample-a")]
    ids = [str(row["sample_id"]) for row in formal_rows]
    contribution_rows = (
        contribution_rows
        if contribution_rows is not None
        else [
            _contribution(sample_id, branch)
            for sample_id in ids
            for branch in BRANCHES
        ]
    )
    availability_rows = (
        availability_rows
        if availability_rows is not None
        else [
            _route(
                sample_id,
                branch,
                split=str(formal_rows[ids.index(sample_id)]["split"]),
            )
            for sample_id in ids
            for branch in BRANCHES
        ]
    )
    formal_path = tmp_path / "formal.jsonl"
    contribution_path = tmp_path / "branch_contribution.jsonl"
    availability_path = tmp_path / "branch_availability.jsonl"
    output_path = tmp_path / "evidence.jsonl"
    _write_jsonl(formal_path, formal_rows)
    _write_jsonl(contribution_path, contribution_rows)
    _write_jsonl(availability_path, availability_rows)
    mapping_path = None
    if mapping_rows is not None:
        mapping_path = tmp_path / "mapping.jsonl"
        _write_jsonl(mapping_path, mapping_rows)
    result = build_a2_evidence_manifest(
        formal_manifest=formal_path,
        branch_contribution_manifest=contribution_path,
        branch_availability_manifest=availability_path,
        sample_mapping_manifest=mapping_path,
        feature_contract=CONTRACT_PATH,
        bridge_config=BRIDGE_CONFIG_PATH,
        output_path=output_path,
    )
    return formal_path, output_path, result


def _read_one(path: Path) -> Mapping[str, Any]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert len(rows) == 1
    return rows[0]


def test_complete_nine_branch_sample_is_a2_readable(tmp_path: Path) -> None:
    formal, evidence, result = _build(tmp_path)
    row = _read_one(evidence)
    assert result.sample_count == 1
    assert list(row["branches"]) == list(BRANCHES)
    assert all(item["feature_available"] for item in row["branches"].values())
    dataset = EvidenceTrainingDataset(
        formal_manifest=formal,
        evidence_manifest=evidence,
        feature_contract=CONTRACT_PATH,
        expected_split="train",
    )
    assert dataset[0]["feature_mask"].all()


def test_partial_missing_branch_is_explicitly_masked(tmp_path: Path) -> None:
    missing = BRANCHES[2]
    contributions = [
        _contribution("sample-a", branch)
        for branch in BRANCHES
        if branch != missing
    ]
    availability = [_route("sample-a", branch) for branch in BRANCHES]
    availability.append(
        _summary(
            "sample-a",
            missing,
            applicable=1,
            valid=0,
            reasons={"observation_missing": 1},
        )
    )
    _, evidence, _ = _build(
        tmp_path,
        contribution_rows=contributions,
        availability_rows=availability,
    )
    branch = _read_one(evidence)["branches"][missing]
    assert branch == {
        "bounded_risk": None,
        "feature_available": False,
        "observability": True,
        "confidence": None,
        "missing_reason": "observation_missing:1",
        "status": "missing",
        "observable": True,
    }


def test_all_branches_missing_are_retained_with_false_a2_mask(
    tmp_path: Path,
) -> None:
    availability = [_route("sample-a", branch) for branch in BRANCHES]
    formal, evidence, _ = _build(
        tmp_path,
        contribution_rows=[],
        availability_rows=availability,
    )
    dataset = EvidenceTrainingDataset(
        formal_manifest=formal,
        evidence_manifest=evidence,
        feature_contract=CONTRACT_PATH,
        expected_split="train",
    )
    assert not dataset[0]["feature_mask"].any()
    assert dataset[0]["missing_mask"].all()


def test_missing_label_is_preserved_from_formal_manifest(tmp_path: Path) -> None:
    _, evidence, _ = _build(
        tmp_path, formal_rows=[_formal_row("sample-a", label=None)]
    )
    assert _read_one(evidence)["label"] is None


def test_real_and_fake_labels_and_lineage_come_only_from_formal(
    tmp_path: Path,
) -> None:
    rows = [
        _formal_row("real", label=0),
        _formal_row("fake", label=1),
    ]
    _, evidence, result = _build(tmp_path, formal_rows=rows)
    output = [
        json.loads(line)
        for line in evidence.read_text(encoding="utf-8").splitlines()
    ]
    assert result.sample_count == 2
    assert {row["sample_id"]: row["label"] for row in output} == {
        "fake": 1,
        "real": 0,
    }
    assert all(row["source_lineage"]["archive"] == "fixture.tar" for row in output)


def test_train_and_validation_splits_are_preserved(tmp_path: Path) -> None:
    formal_rows = [
        _formal_row("train-a", split="train"),
        _formal_row("val-a", split="validation"),
    ]
    contributions = []
    availability = []
    for row in formal_rows:
        for branch in BRANCHES:
            contribution = dict(_contribution(str(row["sample_id"]), branch))
            contribution["split"] = row["split"]
            contributions.append(contribution)
            availability.append(
                _route(
                    str(row["sample_id"]),
                    branch,
                    split=str(row["split"]),
                )
            )
    _, evidence, _ = _build(
        tmp_path,
        formal_rows=formal_rows,
        contribution_rows=contributions,
        availability_rows=availability,
    )
    splits = {
        row["split"]
        for row in map(
            json.loads, evidence.read_text(encoding="utf-8").splitlines()
        )
    }
    assert splits == {"train", "validation"}


def test_test_split_is_rejected_by_a2_training_path(tmp_path: Path) -> None:
    contributions = [
        {**_contribution("sample-a", branch), "split": "test"}
        for branch in BRANCHES
    ]
    availability = [
        _route("sample-a", branch, split="test") for branch in BRANCHES
    ]
    formal, evidence, _ = _build(
        tmp_path,
        formal_rows=[_formal_row("sample-a", split="test")],
        contribution_rows=contributions,
        availability_rows=availability,
    )
    with pytest.raises(ValueError, match="test is forbidden"):
        EvidenceTrainingDataset(
            formal_manifest=formal,
            evidence_manifest=evidence,
            feature_contract=CONTRACT_PATH,
            expected_split="test",
        )


def test_duplicate_sample_id_in_mapping_is_rejected(tmp_path: Path) -> None:
    mapping = [
        {
            "sample_id": "sample-a",
            "video_id": "video-a",
            "clip_id": "clip-a",
            "split": "train",
        },
        {
            "sample_id": "sample-a",
            "video_id": "video-b",
            "clip_id": "clip-b",
            "split": "train",
        },
    ]
    with pytest.raises(ValueError, match="Duplicate sample_id"):
        _build(tmp_path, mapping_rows=mapping)


def test_formal_evidence_join_missing_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing_evidence"):
        _build(
            tmp_path,
            formal_rows=[
                _formal_row("sample-a"),
                _formal_row("sample-b"),
            ],
            contribution_rows=[
                _contribution("sample-a", branch) for branch in BRANCHES
            ],
            availability_rows=[
                _route("sample-a", branch) for branch in BRANCHES
            ],
        )


def test_split_mismatch_is_rejected(tmp_path: Path) -> None:
    contributions = [
        {**_contribution("sample-a", branch), "split": "validation"}
        for branch in BRANCHES
    ]
    with pytest.raises(ValueError, match="Split mismatch"):
        _build(tmp_path, contribution_rows=contributions)


def test_unknown_branch_is_rejected(tmp_path: Path) -> None:
    contributions = [_contribution("sample-a", branch) for branch in BRANCHES]
    contributions.append(_contribution("sample-a", "invented_branch"))
    with pytest.raises(ValueError, match="unknown branch_group"):
        _build(tmp_path, contribution_rows=contributions)


def test_deterministic_fusion_score_does_not_enter_features(tmp_path: Path) -> None:
    _, evidence, _ = _build(tmp_path)
    row = _read_one(evidence)
    serialized = json.dumps(row)
    assert "risk_score" not in serialized
    assert all(
        branch["bounded_risk"] == pytest.approx(0.25)
        for branch in row["branches"].values()
    )


def test_provider_failure_stays_null_and_non_observable(tmp_path: Path) -> None:
    failed = BRANCHES[0]
    contributions = [
        _contribution("sample-a", branch)
        for branch in BRANCHES
        if branch != failed
    ]
    availability = [_route("sample-a", branch) for branch in BRANCHES]
    availability.append(
        _summary(
            "sample-a",
            failed,
            applicable=1,
            valid=0,
            provider_failures=1,
            reasons={"execution_failed": 1},
            provider_statuses={"execution_failed": 1},
        )
    )
    _, evidence, _ = _build(
        tmp_path,
        contribution_rows=contributions,
        availability_rows=availability,
    )
    value = _read_one(evidence)["branches"][failed]
    assert value["status"] == "provider_failed"
    assert value["bounded_risk"] is None
    assert value["confidence"] is None
    assert value["observability"] is False
    assert "execution_failed" in value["missing_reason"]


def test_valid_m6_rows_without_contribution_remain_masked(tmp_path: Path) -> None:
    missing = BRANCHES[0]
    contributions = [
        _contribution("sample-a", branch)
        for branch in BRANCHES
        if branch != missing
    ]
    availability = [_route("sample-a", branch) for branch in BRANCHES]
    availability.append(
        _summary("sample-a", missing, applicable=1, valid=1)
    )
    _, evidence, _ = _build(
        tmp_path,
        contribution_rows=contributions,
        availability_rows=availability,
    )
    value = _read_one(evidence)["branches"][missing]
    assert value["status"] == "missing"
    assert value["feature_available"] is False
    assert value["observability"] is True
    assert value["missing_reason"] == "m6_branch_contribution_unavailable"


def test_contract_branch_order_is_stable(tmp_path: Path) -> None:
    shuffled = [
        _contribution("sample-a", branch)
        for branch in reversed(BRANCHES)
    ]
    _, evidence, result = _build(tmp_path, contribution_rows=shuffled)
    assert result.branch_order == BRANCHES
    assert list(_read_one(evidence)["branches"]) == list(BRANCHES)


def test_conversion_and_input_checksums_are_reproducible(tmp_path: Path) -> None:
    _, output, first = _build(tmp_path / "first")
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    first_copy = tmp_path / "first-copy.jsonl"
    first_copy.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    second = build_a2_evidence_manifest(
        formal_manifest=tmp_path / "first/formal.jsonl",
        branch_contribution_manifest=tmp_path / "first/branch_contribution.jsonl",
        branch_availability_manifest=tmp_path / "first/branch_availability.jsonl",
        feature_contract=CONTRACT_PATH,
        bridge_config=BRIDGE_CONFIG_PATH,
        output_path=tmp_path / "second.jsonl",
    )
    assert first.output_sha256 == second.output_sha256
    assert first.input_sha256 == second.input_sha256
    assert first_copy.read_bytes() == output.read_bytes()


def test_legacy_video_clip_key_requires_and_uses_explicit_mapping(
    tmp_path: Path,
) -> None:
    contributions = []
    availability = []
    for branch in BRANCHES:
        contribution = dict(_contribution("sample-a", branch))
        route = dict(_route("sample-a", branch))
        contribution.pop("sample_id")
        route.pop("sample_id")
        contribution["video_id"] = route["video_id"] = "legacy-video"
        contribution["clip_id"] = route["clip_id"] = "legacy-clip"
        contributions.append(contribution)
        availability.append(route)
    with pytest.raises(ValueError, match="require an explicit sample mapping"):
        _build(
            tmp_path / "blocked",
            contribution_rows=contributions,
            availability_rows=availability,
        )
    _, evidence, _ = _build(
        tmp_path / "mapped",
        contribution_rows=contributions,
        availability_rows=availability,
        mapping_rows=[
            {
                "sample_id": "sample-a",
                "video_id": "legacy-video",
                "clip_id": "legacy-clip",
                "split": "train",
            }
        ],
    )
    assert _read_one(evidence)["sample_id"] == "sample-a"


def test_output_runs_a2_model_forward(tmp_path: Path) -> None:
    formal, evidence, _ = _build(tmp_path)
    dataset = EvidenceTrainingDataset(
        formal_manifest=formal,
        evidence_manifest=evidence,
        feature_contract=CONTRACT_PATH,
        expected_split="train",
    )
    batch = collate_evidence_samples([dataset[0]])
    model = MinimalMissingAwareEvidenceHead(
        branch_count=len(BRANCHES),
        feature_dim=1,
        hidden_dim=8,
    )
    outputs = model(
        features=batch["features"],
        feature_mask=batch["feature_mask"],
        observability=batch["observability"],
        reliability=batch["reliability"],
    )
    assert outputs["logits"].shape == (1,)
    assert torch.isfinite(outputs["logits"]).all()


def test_output_runs_a2_loss_backward(tmp_path: Path) -> None:
    formal, evidence, _ = _build(tmp_path)
    dataset = EvidenceTrainingDataset(
        formal_manifest=formal,
        evidence_manifest=evidence,
        feature_contract=CONTRACT_PATH,
        expected_split="train",
    )
    batch = collate_evidence_samples([dataset[0]])
    model = MinimalMissingAwareEvidenceHead(
        branch_count=len(BRANCHES),
        feature_dim=1,
        hidden_dim=8,
    )
    outputs = model(
        features=batch["features"],
        feature_mask=batch["feature_mask"],
        observability=batch["observability"],
        reliability=batch["reliability"],
    )
    loss = MaskedBinaryLoss()(
        logits=outputs["logits"],
        labels=batch["labels"],
        label_mask=batch["label_mask"],
        valid_sample_mask=outputs["valid_sample_mask"],
    )
    assert loss.loss is not None
    loss.loss.backward()
    assert all(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def test_output_metadata_records_non_fitting_invariants(tmp_path: Path) -> None:
    _, evidence, _ = _build(tmp_path)
    metadata = _read_one(evidence)["metadata"]
    assert metadata["bridge_schema_version"] == EVIDENCE_BRIDGE_SCHEMA_VERSION
    assert metadata["deterministic_fusion_score_read"] is False
    assert metadata["provider_failure_encoded_as_feature"] is False
    assert metadata["normalization_fitted"] is False
    assert metadata["threshold_selected"] is False
    assert metadata["label_source"] == "formal_manifest_only"


def test_cli_help_is_available_without_assets() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts/build_p4c3c_a3_evidence_manifest.py"),
            "--help",
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--branch-contribution-manifest" in result.stdout
    assert "--sample-mapping-manifest" in result.stdout


def test_actual_m6_csv_shapes_convert_through_legacy_mapping(
    tmp_path: Path,
) -> None:
    formal_path = tmp_path / "formal.jsonl"
    contribution_path = tmp_path / "branch_contribution_audit.csv"
    availability_path = tmp_path / "branch_availability_audit.csv"
    mapping_path = tmp_path / "mapping.jsonl"
    output_path = tmp_path / "evidence.jsonl"
    _write_jsonl(formal_path, [_formal_row("sample-a")])
    _write_jsonl(
        mapping_path,
        [
            {
                "sample_id": "sample-a",
                "video_id": "legacy-video",
                "clip_id": "legacy-clip",
                "split": "train",
            }
        ],
    )
    contribution_rows = []
    availability_rows = []
    for branch in BRANCHES:
        contribution = dict(_contribution("sample-a", branch))
        contribution.pop("sample_id")
        contribution.pop("split")
        contribution["video_id"] = "legacy-video"
        contribution["clip_id"] = "legacy-clip"
        contribution_rows.append(contribution)
        route = dict(_route("sample-a", branch))
        route.pop("sample_id")
        route.pop("split")
        route["video_id"] = "legacy-video"
        route["clip_id"] = "legacy-clip"
        availability_rows.append(route)
        summary = dict(
            _summary("sample-a", branch, applicable=1, valid=1)
        )
        summary.pop("sample_id")
        summary.pop("split")
        summary["video_id"] = "legacy-video"
        summary["clip_id"] = "legacy-clip"
        availability_rows.append(summary)
    _write_csv(contribution_path, contribution_rows)
    _write_csv(availability_path, availability_rows)
    result = build_a2_evidence_manifest(
        formal_manifest=formal_path,
        branch_contribution_manifest=contribution_path,
        branch_availability_manifest=availability_path,
        sample_mapping_manifest=mapping_path,
        feature_contract=CONTRACT_PATH,
        bridge_config=BRIDGE_CONFIG_PATH,
        output_path=output_path,
    )
    assert result.sample_count == 1
    assert all(
        value["status"] == "observed"
        for value in _read_one(output_path)["branches"].values()
    )
