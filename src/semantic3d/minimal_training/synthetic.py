"""Deterministic, non-video synthetic fixtures for A2 engineering tests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from semantic3d.dataset_builder.formal_schema import (
    OPTIONAL_METADATA_FIELDS,
    FormalVideoSample,
)

from .contracts import EVIDENCE_MANIFEST_SCHEMA_VERSION, load_feature_contract


@dataclass(frozen=True)
class SyntheticFixturePaths:
    train_formal_manifest: Path
    train_manifest: Path
    validation_formal_manifest: Path
    validation_manifest: Path


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(
        json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows
    )
    path.write_text(text, encoding="utf-8")


def _formal_row(sample_id: str, split: str, label: int | None) -> dict[str, Any]:
    status = {name: "missing" for name in OPTIONAL_METADATA_FIELDS}
    status.update(
        {
            "label": "provided" if label is not None else "missing",
            "split": "provided",
            "source_id": "provided",
            "source_lineage": "provided",
            "generator": "provided",
            "source_domain": "provided",
            "is_real": "derived" if label is not None else "missing",
        }
    )
    digest = hashlib.sha256(f"synthetic:{sample_id}".encode("utf-8")).hexdigest()
    return FormalVideoSample(
        sample_id=sample_id,
        video_path=f"synthetic/{sample_id}.mp4",
        label=label,
        split=split,
        source_dataset="p4c3c_a2_synthetic_engineering",
        source_id=f"source-{sample_id}",
        source_lineage={"fixture_family": split, "private_asset": False},
        generator="deterministic_fixture",
        is_real=None if label is None else label == 0,
        duration=None,
        fps=None,
        frame_count=None,
        width=None,
        height=None,
        file_size=0,
        sha256=digest,
        temporal_annotation=None,
        spatial_annotation=None,
        metadata_status=status,
        source_path=f"synthetic/{sample_id}.mp4",
        path_mode="data_root_relative",
        source_domain="synthetic_non_video",
        official_split=False,
    ).to_dict()


def _branch(
    *,
    value: float | None,
    confidence: float | None,
    observable: bool,
    status: str,
) -> dict[str, Any]:
    return {
        "bounded_risk": value,
        "confidence": confidence,
        "observable": observable,
        "status": status,
    }


def _evidence_row(
    *,
    sample_id: str,
    split: str,
    contract_version: str,
    branch_order: tuple[str, ...],
    primary: float | None,
    secondary: float | None,
    missing_secondary: bool = False,
    all_missing: bool = False,
) -> dict[str, Any]:
    branches = {
        name: _branch(
            value=None,
            confidence=None,
            observable=False,
            status="not_applicable",
        )
        for name in branch_order
    }
    first, second = branch_order[:2]
    if all_missing:
        branches[first] = _branch(
            value=None,
            confidence=None,
            observable=True,
            status="missing",
        )
        branches[second] = _branch(
            value=None,
            confidence=None,
            observable=True,
            status="blocked_by_input",
        )
    else:
        branches[first] = _branch(
            value=primary,
            confidence=0.9,
            observable=True,
            status="observed",
        )
        branches[second] = (
            _branch(
                value=None,
                confidence=None,
                observable=True,
                status="missing",
            )
            if missing_secondary
            else _branch(
                value=secondary,
                confidence=0.75,
                observable=True,
                status="observed",
            )
        )
    return {
        "schema_version": EVIDENCE_MANIFEST_SCHEMA_VERSION,
        "feature_contract_version": contract_version,
        "sample_id": sample_id,
        "split": split,
        "branches": branches,
        "metadata": {
            "fixture_only": True,
            "provider_inference_executed": False,
            "video_decoded": False,
            "performance_claim_allowed": False,
        },
    }


def create_synthetic_fixture(
    root: str | Path,
    *,
    feature_contract_path: str | Path,
) -> SyntheticFixturePaths:
    """Write disjoint train/validation JSONL manifests without media files."""

    output = Path(root)
    contract = load_feature_contract(feature_contract_path)
    definitions = {
        "train": [
            ("train-real", 0, 0.08, 0.12, False, False),
            ("train-fake", 1, 0.82, 0.70, False, False),
            ("train-partial", 1, 0.67, None, True, False),
            ("train-no-evidence", 0, None, None, False, True),
            ("train-unlabeled", None, 0.41, 0.34, False, False),
        ],
        "validation": [
            ("validation-real", 0, 0.10, 0.16, False, False),
            ("validation-fake", 1, 0.78, 0.74, False, False),
            ("validation-partial", 1, 0.61, None, True, False),
            ("validation-no-evidence", 0, None, None, False, True),
            ("validation-unlabeled", None, 0.35, 0.31, False, False),
        ],
    }
    paths: dict[str, Path] = {}
    for split, rows in definitions.items():
        formal_rows = [_formal_row(sample_id, split, label) for sample_id, label, *_ in rows]
        evidence_rows = [
            _evidence_row(
                sample_id=sample_id,
                split=split,
                contract_version=contract.version,
                branch_order=contract.branch_order,
                primary=primary,
                secondary=secondary,
                missing_secondary=missing_secondary,
                all_missing=all_missing,
            )
            for (
                sample_id,
                _label,
                primary,
                secondary,
                missing_secondary,
                all_missing,
            ) in rows
        ]
        formal_path = output / f"{split}_formal.jsonl"
        evidence_path = output / f"{split}_evidence.jsonl"
        _write_jsonl(formal_path, formal_rows)
        _write_jsonl(evidence_path, evidence_rows)
        paths[f"{split}_formal"] = formal_path
        paths[f"{split}_evidence"] = evidence_path
    return SyntheticFixturePaths(
        train_formal_manifest=paths["train_formal"],
        train_manifest=paths["train_evidence"],
        validation_formal_manifest=paths["validation_formal"],
        validation_manifest=paths["validation_evidence"],
    )
