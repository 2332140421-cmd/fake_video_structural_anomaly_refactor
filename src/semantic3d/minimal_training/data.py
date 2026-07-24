"""Dataset, DataLoader, and collate for precomputed M6 branch evidence."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from semantic3d.dataset_builder.formal_schema import (
    FORMAL_SAMPLE_SCHEMA_VERSION,
    FormalVideoSample,
)

from .contracts import (
    EVIDENCE_MANIFEST_SCHEMA_VERSION,
    FeatureContract,
    combined_manifest_checksum,
    load_feature_contract,
)

_A2_SPLITS = frozenset({"train", "validation"})
_FORMAL_JSON_FIELDS = {
    "source_lineage",
    "temporal_annotation",
    "spatial_annotation",
    "metadata_status",
}
_FORMAL_INT_FIELDS = {"label", "frame_count", "width", "height", "file_size"}
_FORMAL_FLOAT_FIELDS = {"duration", "fps"}


def _read_records(path: str | Path) -> list[dict[str, Any]]:
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {file_path}")
    suffix = file_path.suffix.lower()
    if suffix == ".jsonl":
        records: list[dict[str, Any]] = []
        for line_number, raw_line in enumerate(
            file_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not raw_line.strip():
                continue
            value = json.loads(raw_line)
            if not isinstance(value, dict):
                raise ValueError(
                    f"{file_path}:{line_number} must contain a JSON object."
                )
            records.append(value)
        return records
    if suffix == ".json":
        value = json.loads(file_path.read_text(encoding="utf-8"))
        if not isinstance(value, list) or not all(
            isinstance(item, dict) for item in value
        ):
            raise ValueError(f"{file_path} must contain a JSON list of objects.")
        return list(value)
    if suffix == ".csv":
        with file_path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    if suffix in {".parquet", ".pq"}:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("Reading Parquet requires the declared pyarrow dependency.") from exc
        return [dict(item) for item in pq.read_table(file_path).to_pylist()]
    raise ValueError(
        f"Unsupported manifest format for {file_path}; use JSONL, JSON, CSV, or Parquet."
    )


def _none_if_empty(value: Any) -> Any:
    return None if value is None or (isinstance(value, str) and not value.strip()) else value


def _json_or_value(value: Any) -> Any:
    value = _none_if_empty(value)
    if value is None or not isinstance(value, str):
        return value
    return json.loads(value)


def _coerce_formal_row(row: Mapping[str, Any]) -> dict[str, Any]:
    output = dict(row)
    for name in _FORMAL_JSON_FIELDS:
        output[name] = _json_or_value(output.get(name))
    for name in _FORMAL_INT_FIELDS:
        value = _none_if_empty(output.get(name))
        output[name] = None if value is None else int(value)
    for name in _FORMAL_FLOAT_FIELDS:
        value = _none_if_empty(output.get(name))
        output[name] = None if value is None else float(value)
    official = output.get("official_split", False)
    if isinstance(official, str):
        official = official.strip().lower() in {"1", "true", "yes"}
    output["official_split"] = bool(official)
    is_real = _none_if_empty(output.get("is_real"))
    if isinstance(is_real, str):
        normalized = is_real.strip().lower()
        if normalized not in {"true", "false", "1", "0", "yes", "no"}:
            raise ValueError(f"Unsupported is_real value: {is_real!r}")
        is_real = normalized in {"true", "1", "yes"}
    output["is_real"] = is_real
    output.setdefault("source_domain", None)
    output.setdefault("schema_version", FORMAL_SAMPLE_SCHEMA_VERSION)
    return output


def _unique_by_sample_id(
    records: Iterable[Mapping[str, Any]],
    *,
    manifest_name: str,
) -> dict[str, Mapping[str, Any]]:
    output: dict[str, Mapping[str, Any]] = {}
    for row in records:
        sample_id = str(row.get("sample_id", "")).strip()
        if not sample_id:
            raise ValueError(f"{manifest_name} contains an empty sample_id.")
        if sample_id in output:
            raise ValueError(
                f"Duplicate sample_id {sample_id!r} in {manifest_name}."
            )
        output[sample_id] = row
    if not output:
        raise ValueError(f"{manifest_name} must contain at least one sample.")
    return output


class EvidenceTrainingDataset(Dataset[dict[str, Any]]):
    """Join A1 formal samples with precomputed branch-level M6 evidence."""

    def __init__(
        self,
        *,
        formal_manifest: str | Path,
        evidence_manifest: str | Path,
        feature_contract: str | Path | FeatureContract,
        expected_split: str,
    ) -> None:
        if expected_split not in _A2_SPLITS:
            raise ValueError(
                "A2 datasets only accept expected_split='train' or 'validation'; "
                "test is forbidden."
            )
        self.expected_split = expected_split
        self.formal_manifest_path = Path(formal_manifest)
        self.evidence_manifest_path = Path(evidence_manifest)
        self.feature_contract = (
            feature_contract
            if isinstance(feature_contract, FeatureContract)
            else load_feature_contract(feature_contract)
        )
        formal_records = [
            FormalVideoSample(**_coerce_formal_row(row))
            for row in _read_records(self.formal_manifest_path)
        ]
        formal_by_id = _unique_by_sample_id(
            (sample.to_dict() for sample in formal_records),
            manifest_name=str(self.formal_manifest_path),
        )
        evidence_by_id = _unique_by_sample_id(
            _read_records(self.evidence_manifest_path),
            manifest_name=str(self.evidence_manifest_path),
        )
        if set(formal_by_id) != set(evidence_by_id):
            missing_evidence = sorted(set(formal_by_id) - set(evidence_by_id))
            unknown_evidence = sorted(set(evidence_by_id) - set(formal_by_id))
            raise ValueError(
                "Formal/evidence sample_id sets differ: "
                f"missing_evidence={missing_evidence}, "
                f"unknown_evidence={unknown_evidence}."
            )
        self._samples = tuple(
            self._build_sample(formal_by_id[sample_id], evidence_by_id[sample_id])
            for sample_id in sorted(formal_by_id)
        )
        self.sample_ids = frozenset(item["sample_id"] for item in self._samples)
        self.manifest_checksum = combined_manifest_checksum(
            self.formal_manifest_path,
            self.evidence_manifest_path,
        )

    def _build_sample(
        self,
        formal: Mapping[str, Any],
        evidence: Mapping[str, Any],
    ) -> dict[str, Any]:
        sample_id = str(formal["sample_id"])
        formal_split = formal.get("split")
        evidence_split = evidence.get("split")
        if formal_split is None or evidence_split is None:
            raise ValueError(f"Explicit split is required for sample {sample_id}.")
        if formal_split == "test" or evidence_split == "test":
            raise ValueError(f"A2 refuses test split sample {sample_id}.")
        if formal_split != self.expected_split or evidence_split != self.expected_split:
            raise ValueError(
                f"Split mismatch for {sample_id}: formal={formal_split!r}, "
                f"evidence={evidence_split!r}, expected={self.expected_split!r}."
            )
        schema_version = evidence.get("schema_version")
        if schema_version != EVIDENCE_MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported evidence schema_version for {sample_id}: "
                f"{schema_version!r}."
            )
        contract_version = evidence.get("feature_contract_version")
        if contract_version != self.feature_contract.version:
            raise ValueError(
                f"Feature contract mismatch for {sample_id}: "
                f"{contract_version!r} != {self.feature_contract.version!r}."
            )
        branches = evidence.get("branches")
        if not isinstance(branches, dict):
            raise ValueError(f"Evidence branches must be a mapping for {sample_id}.")
        missing_branches = sorted(set(self.feature_contract.branch_order) - set(branches))
        unexpected_branches = sorted(set(branches) - set(self.feature_contract.branch_order))
        if missing_branches or unexpected_branches:
            raise ValueError(
                f"Branch set mismatch for {sample_id}: missing={missing_branches}, "
                f"unexpected={unexpected_branches}."
            )

        features: list[list[float]] = []
        feature_mask: list[list[bool]] = []
        observability: list[bool] = []
        reliability: list[float] = []
        missing_status: dict[str, str] = {}
        for branch_name in self.feature_contract.branch_order:
            branch = branches[branch_name]
            if not isinstance(branch, dict):
                raise ValueError(
                    f"Branch {branch_name!r} for {sample_id} must be a mapping."
                )
            status = str(branch.get("status", ""))
            if status not in self.feature_contract.allowed_branch_status:
                raise ValueError(
                    f"Unsupported branch status {status!r} for "
                    f"{sample_id}/{branch_name}."
                )
            observable = branch.get("observable")
            if not isinstance(observable, bool):
                raise ValueError(
                    f"observable must be bool for {sample_id}/{branch_name}."
                )
            raw_value = branch.get("bounded_risk")
            raw_confidence = branch.get("confidence")
            observed = status == "observed"
            if observed:
                if not observable:
                    raise ValueError(
                        f"Observed branch must be observable for {sample_id}/{branch_name}."
                    )
                if isinstance(raw_value, bool) or isinstance(raw_confidence, bool):
                    raise ValueError(
                        f"Numeric fields cannot be bool for {sample_id}/{branch_name}."
                    )
                try:
                    value = float(raw_value)
                    confidence = float(raw_confidence)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Observed branch requires numeric bounded_risk/confidence for "
                        f"{sample_id}/{branch_name}."
                    ) from exc
                if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                    raise ValueError(
                        f"bounded_risk must be finite in [0,1] for "
                        f"{sample_id}/{branch_name}."
                    )
                if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                    raise ValueError(
                        f"confidence must be finite in [0,1] for "
                        f"{sample_id}/{branch_name}."
                    )
                features.append([value])
                feature_mask.append([True])
                reliability.append(confidence)
            else:
                if raw_value is not None or raw_confidence is not None:
                    raise ValueError(
                        f"Missing branch values must be null for {sample_id}/{branch_name}."
                    )
                if status in {"not_applicable", "provider_failed"} and observable:
                    raise ValueError(
                        f"{status} branch cannot be observable for "
                        f"{sample_id}/{branch_name}."
                    )
                features.append([0.0])
                feature_mask.append([False])
                reliability.append(0.0)
            observability.append(observable)
            missing_status[branch_name] = status

        label = formal.get("label")
        if label not in {0, 1, None}:
            raise ValueError(f"label must be 0, 1, or null for {sample_id}.")
        return {
            "sample_id": sample_id,
            "features": torch.tensor(features, dtype=torch.float32),
            "feature_mask": torch.tensor(feature_mask, dtype=torch.bool),
            "missing_mask": torch.tensor(feature_mask, dtype=torch.bool).logical_not(),
            "observability": torch.tensor(observability, dtype=torch.bool),
            "reliability": torch.tensor(reliability, dtype=torch.float32),
            "label": label,
            "split": self.expected_split,
            "source_dataset": str(formal["source_dataset"]),
            "generator": formal.get("generator"),
            "metadata": {
                "source_lineage": formal.get("source_lineage"),
                "metadata_status": formal.get("metadata_status"),
                "evidence_metadata": evidence.get("metadata", {}),
                "branch_missing_status": missing_status,
                "future_targets": {
                    "temporal": formal.get("temporal_annotation"),
                    "spatial": formal.get("spatial_annotation"),
                    "object": None,
                    "branch_auxiliary": None,
                },
            },
        }

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._samples[index]


def collate_evidence_samples(
    samples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Stack fixed tensors without truncation or unknown-dimension padding."""

    if not samples:
        raise ValueError("Cannot collate an empty batch.")
    expected_feature_shape = tuple(samples[0]["features"].shape)
    expected_observability_shape = tuple(samples[0]["observability"].shape)
    for sample in samples:
        sample_id = sample.get("sample_id", "<unknown>")
        for name in ("features", "feature_mask", "missing_mask"):
            shape = tuple(sample[name].shape)
            if shape != expected_feature_shape:
                raise ValueError(
                    f"Inconsistent {name} shape for {sample_id}: "
                    f"{shape} != {expected_feature_shape}; A2 never truncates."
                )
        for name in ("observability", "reliability"):
            shape = tuple(sample[name].shape)
            if shape != expected_observability_shape:
                raise ValueError(
                    f"Inconsistent {name} shape for {sample_id}: "
                    f"{shape} != {expected_observability_shape}; A2 never truncates."
                )
        if not torch.equal(sample["missing_mask"], sample["feature_mask"].logical_not()):
            raise ValueError(f"missing_mask is not inverse feature_mask for {sample_id}.")
    labels = torch.tensor(
        [0.0 if sample["label"] is None else float(sample["label"]) for sample in samples],
        dtype=torch.float32,
    )
    label_mask = torch.tensor(
        [sample["label"] is not None for sample in samples],
        dtype=torch.bool,
    )
    return {
        "features": torch.stack([sample["features"] for sample in samples]),
        "feature_mask": torch.stack([sample["feature_mask"] for sample in samples]),
        "missing_mask": torch.stack([sample["missing_mask"] for sample in samples]),
        "observability": torch.stack([sample["observability"] for sample in samples]),
        "reliability": torch.stack([sample["reliability"] for sample in samples]),
        "labels": labels,
        "label_mask": label_mask,
        "sample_ids": [str(sample["sample_id"]) for sample in samples],
        "splits": [str(sample["split"]) for sample in samples],
        "source_datasets": [str(sample["source_dataset"]) for sample in samples],
        "generators": [sample["generator"] for sample in samples],
        "metadata": [sample["metadata"] for sample in samples],
    }


@dataclass(frozen=True)
class TrainingDataBundle:
    """Strict train/validation datasets and deterministic loaders."""

    train_dataset: EvidenceTrainingDataset
    validation_dataset: EvidenceTrainingDataset
    train_loader: DataLoader[dict[str, Any]]
    validation_loader: DataLoader[dict[str, Any]]


def build_training_dataloaders(
    *,
    train_formal_manifest: str | Path,
    train_manifest: str | Path,
    validation_formal_manifest: str | Path,
    validation_manifest: str | Path,
    feature_contract: str | Path | FeatureContract,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> TrainingDataBundle:
    """Build isolated A2 loaders and reject sample-ID leakage."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative.")
    contract = (
        feature_contract
        if isinstance(feature_contract, FeatureContract)
        else load_feature_contract(feature_contract)
    )
    train_dataset = EvidenceTrainingDataset(
        formal_manifest=train_formal_manifest,
        evidence_manifest=train_manifest,
        feature_contract=contract,
        expected_split="train",
    )
    validation_dataset = EvidenceTrainingDataset(
        formal_manifest=validation_formal_manifest,
        evidence_manifest=validation_manifest,
        feature_contract=contract,
        expected_split="validation",
    )
    overlap = sorted(train_dataset.sample_ids & validation_dataset.sample_ids)
    if overlap:
        raise ValueError(f"Train/validation sample_id leakage detected: {overlap}.")
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_evidence_samples,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_evidence_samples,
    )
    return TrainingDataBundle(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        train_loader=train_loader,
        validation_loader=validation_loader,
    )
