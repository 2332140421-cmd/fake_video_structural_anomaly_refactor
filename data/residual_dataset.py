"""Validated datasets built directly from frozen paper-core residual JSON."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from data.leakage import audit_split_leakage
from data.runtime_paths import relocate_manifest_rows

RESIDUAL_NAMES = (
    "semantic_metric_prior",
    "semantic_metric_temporal",
    "dynamic_reprojection",
    "track_3d_continuity",
    "direction_consistency",
    "relative_velocity",
    "point_reprojection",
    "boundary_reprojection",
    "depth_reprojection",
    "relation",
    "occlusion",
    "reappearance",
)
SPLITS = ("train", "validation", "test")
REQUIRED_MANIFEST_FIELDS = (
    "sample_id",
    "dataset_name",
    "source_video_id",
    "group_id",
    "split",
    "label",
    "residual_sequence_path",
)


@dataclass(frozen=True)
class ResidualSequence:
    residuals: np.ndarray
    availability: np.ndarray
    confidence: np.ndarray
    label: int
    sample_id: str = ""
    clip_ids: tuple[str, ...] = ()
    dataset_name: str = ""
    source_video_id: str = ""
    group_id: str = ""
    residual_sequence_path: str = ""


@dataclass(frozen=True)
class TrainingManifestBundle:
    samples: Mapping[str, tuple[ResidualSequence, ...]]
    sample_ids: Mapping[str, tuple[str, ...]]
    source_commit: str
    source_config_sha256: str
    channel_schema: Mapping[str, Any]
    manifest_path: str
    manifest_sha256: str
    manifest_rows: tuple[Mapping[str, str], ...]
    leakage_audit: Mapping[str, Any]
    runtime_path_manifest: str | None = None

    @property
    def config_sha256(self) -> str:
        """Backward-compatible name for the frozen producer config identity."""

        return self.source_config_sha256


class ResidualSequenceDataset(Dataset):
    def __init__(self, samples: Sequence[ResidualSequence]) -> None:
        if not samples:
            raise ValueError("ResidualSequenceDataset cannot be empty.")
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        return (
            torch.as_tensor(sample.residuals, dtype=torch.float32),
            torch.as_tensor(sample.availability, dtype=torch.bool),
            torch.as_tensor(sample.confidence, dtype=torch.float32),
            torch.tensor(sample.label, dtype=torch.float32),
            sample.sample_id,
        )


def residual_channel_schema(
    source_commit: str,
    source_config_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "channel_count": len(RESIDUAL_NAMES),
        "channel_names": list(RESIDUAL_NAMES),
        "value_dtype": "float32",
        "availability_dtype": "bool",
        "confidence_dtype": "float32",
        "missing_value_policy": (
            "unavailable values remain NaN with availability=false and confidence=0; "
            "zero filling occurs inside the model only after masking"
        ),
        "padding_policy": (
            "right padding is represented by a separate [B,T] boolean padding mask"
        ),
        "source_commit": str(source_commit),
        "source_config_sha256": str(source_config_sha256),
        "normalization": (
            "producer_normalized_value=1-exp(-raw_value), then arithmetic mean "
            "over valid evidence rows within each frozen clip"
        ),
        "timestep_definition": "one frozen inference clip ordered by start_frame",
        "label_is_input_channel": False,
    }


def validate_channel_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(schema)
    if "source_config_sha256" not in normalized and "config_sha256" in normalized:
        normalized["source_config_sha256"] = normalized.pop("config_sha256")
    required = {
        "schema_version",
        "channel_count",
        "channel_names",
        "value_dtype",
        "availability_dtype",
        "confidence_dtype",
        "missing_value_policy",
        "padding_policy",
        "source_commit",
        "source_config_sha256",
    }
    missing = sorted(required - set(normalized))
    if missing:
        raise ValueError(f"Residual channel schema is missing fields: {missing}.")
    if int(normalized["channel_count"]) != len(RESIDUAL_NAMES):
        raise ValueError("Residual channel schema must contain exactly 12 channels.")
    if tuple(normalized["channel_names"]) != RESIDUAL_NAMES:
        raise ValueError("Residual channel order does not match the frozen producer order.")
    if bool(normalized.get("label_is_input_channel", False)):
        raise ValueError("Authenticity label must not be a residual input channel.")
    expected_dtypes = {
        "value_dtype": "float32",
        "availability_dtype": "bool",
        "confidence_dtype": "float32",
    }
    for key, expected in expected_dtypes.items():
        if str(normalized[key]) != expected:
            raise ValueError(f"Unsupported {key}: {normalized[key]!r}.")
    return normalized


def load_channel_schema(path: str | Path) -> dict[str, Any]:
    return validate_channel_schema(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )


def _sequence_from_payload(
    payload: Mapping[str, Any],
    *,
    label: int,
    sample_id: str,
    dataset_name: str,
    source_video_id: str,
    group_id: str,
    residual_sequence_path: Path,
) -> ResidualSequence:
    clips = list(payload.get("clips", ()))
    if not clips:
        raise ValueError(f"{sample_id}: residual result contains no clips.")
    starts = [int(clip["start_frame"]) for clip in clips]
    if starts != sorted(starts) or len(set(starts)) != len(starts):
        raise ValueError(f"{sample_id}: clips are not in a unique temporal order.")
    residuals = np.full((len(clips), len(RESIDUAL_NAMES)), np.nan, dtype=np.float32)
    availability = np.zeros_like(residuals, dtype=bool)
    confidence = np.zeros_like(residuals, dtype=np.float32)
    for time_index, clip in enumerate(clips):
        by_name: dict[str, list[Mapping[str, Any]]] = {}
        for row in clip.get("residuals", ()):
            name = str(row.get("name", ""))
            if name not in RESIDUAL_NAMES:
                raise ValueError(f"{sample_id}: unknown residual channel {name!r}.")
            if not bool(row.get("valid_mask", False)):
                continue
            if row.get("availability") != "observed":
                raise ValueError(f"{sample_id}: valid residual is not marked observed.")
            value = float(row["normalized_value"])
            quality = float(row["confidence"])
            if not math.isfinite(value) or not math.isfinite(quality):
                raise ValueError(f"{sample_id}: valid residual contains NaN or Inf.")
            if not 0.0 <= quality <= 1.0:
                raise ValueError(f"{sample_id}: residual confidence is outside [0,1].")
            by_name.setdefault(name, []).append(row)
        for residual_index, name in enumerate(RESIDUAL_NAMES):
            rows = by_name.get(name, ())
            if rows:
                residuals[time_index, residual_index] = float(
                    np.mean([float(row["normalized_value"]) for row in rows])
                )
                availability[time_index, residual_index] = True
                confidence[time_index, residual_index] = float(
                    np.mean([float(row["confidence"]) for row in rows])
                )
    if not np.any(availability):
        raise ValueError(f"{sample_id}: training video has no valid residual.")
    if np.any(~np.isfinite(residuals[availability])):
        raise ValueError(f"{sample_id}: a model-valid position contains NaN or Inf.")
    if np.any(confidence[~availability] != 0.0):
        raise ValueError(f"{sample_id}: unavailable positions must have zero confidence.")
    return ResidualSequence(
        residuals=residuals,
        availability=availability,
        confidence=confidence,
        label=int(label),
        sample_id=sample_id,
        clip_ids=tuple(str(clip["clip_id"]) for clip in clips),
        dataset_name=dataset_name,
        source_video_id=source_video_id,
        group_id=group_id,
        residual_sequence_path=str(residual_sequence_path),
    )


def _read_training_manifest(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = set(reader.fieldnames or ())
    missing = sorted(set(REQUIRED_MANIFEST_FIELDS) - fields)
    if not rows or missing:
        raise ValueError(f"Training manifest is missing required fields: {missing}.")
    return rows


def _schema_from_rows(rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    commits = {row.get("source_commit", "").strip() for row in rows}
    config_hashes = {
        (row.get("source_config_sha256") or row.get("config_sha256") or "").strip()
        for row in rows
    }
    if len(commits) != 1 or not next(iter(commits)):
        raise ValueError(
            "Manifest must provide one source_commit when --channel-schema is omitted."
        )
    if len(config_hashes) != 1 or not next(iter(config_hashes)):
        raise ValueError(
            "Manifest must provide one source_config_sha256 when --channel-schema is omitted."
        )
    return residual_channel_schema(next(iter(commits)), next(iter(config_hashes)))


def build_manifest_samples(
    manifest_path: str | Path,
    channel_schema_path: str | Path | Mapping[str, Any] | None = None,
    *,
    runtime_path_manifest: str | Path | None = None,
    leakage_check: Mapping[str, Any] | None = None,
) -> TrainingManifestBundle:
    """Load frozen results without provider construction or random splitting."""

    resolved_manifest = Path(manifest_path).resolve()
    provenance_rows = _read_training_manifest(resolved_manifest)
    rows = (
        relocate_manifest_rows(provenance_rows, runtime_path_manifest)
        if runtime_path_manifest is not None
        else provenance_rows
    )
    if channel_schema_path is None:
        schema = _schema_from_rows(rows)
    elif isinstance(channel_schema_path, Mapping):
        schema = validate_channel_schema(channel_schema_path)
    else:
        schema = load_channel_schema(channel_schema_path)
    output: dict[str, list[ResidualSequence]] = {split: [] for split in SPLITS}
    leakage_records: list[dict[str, Any]] = []
    sample_splits: dict[str, set[str]] = {}
    source_commits: set[str] = set()
    config_hashes: set[str] = set()
    for row in rows:
        split = row["split"].strip()
        if split not in output:
            raise ValueError(f"Unsupported split {split!r}.")
        sample_id = row["sample_id"].strip()
        dataset_name = row["dataset_name"].strip()
        source_video_id = row["source_video_id"].strip()
        group_id = row["group_id"].strip()
        if not all((sample_id, dataset_name, source_video_id, group_id)):
            raise ValueError("Manifest identities must be non-empty.")
        seen_splits = sample_splits.setdefault(sample_id, set())
        if split in seen_splits:
            raise ValueError(f"Duplicate sample_id: {sample_id!r}.")
        seen_splits.add(split)
        label = int(row["label"])
        if label not in {0, 1}:
            raise ValueError("Training labels must be 0 or 1.")
        residual_path = Path(row["residual_sequence_path"]).expanduser().resolve()
        if not residual_path.is_file():
            raise FileNotFoundError(f"{sample_id}: residual sequence is missing.")
        payload = json.loads(residual_path.read_text(encoding="utf-8"))
        source_video_raw = row.get("source_video_path", "").strip()
        if source_video_raw:
            source_video = Path(source_video_raw).expanduser().resolve()
            if not source_video.is_file():
                raise FileNotFoundError(f"{sample_id}: optional source video is missing.")
            payload_video = str(payload.get("video_path", "")).strip()
            if (
                payload_video
                and row.get("_runtime_relocated") != "true"
                and Path(payload_video).resolve() != source_video
            ):
                raise ValueError(f"{sample_id}: result video path does not match manifest.")
        metadata = payload.get("metadata", {})
        if bool(metadata.get("authenticity_label_used", False)):
            raise ValueError(f"{sample_id}: inference output reports label use.")
        if bool(metadata.get("historical_csv_read", False)):
            raise ValueError(f"{sample_id}: historical CSV input is forbidden.")
        if bool(metadata.get("m6_to_a2_bridge_called", False)):
            raise ValueError(f"{sample_id}: M6-to-A2 bridge input is forbidden.")
        sequence = _sequence_from_payload(
            payload,
            label=label,
            sample_id=sample_id,
            dataset_name=dataset_name,
            source_video_id=source_video_id,
            group_id=group_id,
            residual_sequence_path=residual_path,
        )
        output[split].append(sequence)
        source_commit = row.get("source_commit", "").strip()
        source_config = (
            row.get("source_config_sha256") or row.get("config_sha256") or ""
        ).strip()
        if source_commit:
            source_commits.add(source_commit)
        if source_config:
            config_hashes.add(source_config)
        leakage_records.append(
            {
                "split": split,
                "sample_id": sample_id,
                "clip_ids": sequence.clip_ids,
                "dataset_name": dataset_name,
                "source_video_id": source_video_id,
                "group_id": group_id,
                "source_video_sha256": row.get("source_video_sha256", "").strip(),
                "source_video_path": str(source_video) if source_video_raw else "",
                "residual_sequence_path": str(residual_path),
            }
        )
    expected_commit = str(schema["source_commit"])
    expected_config = str(schema["source_config_sha256"])
    if source_commits and source_commits != {expected_commit}:
        raise ValueError("Manifest and channel schema source commits differ.")
    if config_hashes and config_hashes != {expected_config}:
        raise ValueError("Manifest and channel schema config identities differ.")
    leakage_audit = audit_split_leakage(leakage_records, leakage_check)
    if not output["train"] or not output["validation"]:
        raise ValueError("Manifest requires non-empty train and validation splits.")
    manifest_sha256 = hashlib.sha256(resolved_manifest.read_bytes()).hexdigest()
    return TrainingManifestBundle(
        samples={key: tuple(value) for key, value in output.items()},
        sample_ids={
            key: tuple(sample.sample_id for sample in value)
            for key, value in output.items()
        },
        source_commit=expected_commit,
        source_config_sha256=expected_config,
        channel_schema=schema,
        manifest_path=str(resolved_manifest),
        manifest_sha256=manifest_sha256,
        manifest_rows=tuple(dict(row) for row in rows),
        leakage_audit=leakage_audit,
        runtime_path_manifest=(
            str(Path(runtime_path_manifest).resolve())
            if runtime_path_manifest is not None
            else None
        ),
    )


def collate_residual_sequences(batch):
    max_steps = max(item[0].shape[0] for item in batch)
    residual_count = batch[0][0].shape[1]
    residuals = torch.full((len(batch), max_steps, residual_count), float("nan"))
    availability = torch.zeros((len(batch), max_steps, residual_count), dtype=torch.bool)
    confidence = torch.zeros((len(batch), max_steps, residual_count))
    padding_mask = torch.zeros((len(batch), max_steps), dtype=torch.bool)
    labels = torch.empty(len(batch))
    sample_ids: list[str] = []
    for index, (values, valid, quality, label, sample_id) in enumerate(batch):
        steps = values.shape[0]
        residuals[index, :steps] = values
        availability[index, :steps] = valid
        confidence[index, :steps] = quality
        padding_mask[index, :steps] = True
        labels[index] = label
        sample_ids.append(sample_id)
    return residuals, availability, confidence, padding_mask, labels, sample_ids


def sequence_statistics(samples: Sequence[ResidualSequence]) -> dict[str, Any]:
    values = np.concatenate([sample.residuals.reshape(-1) for sample in samples])
    valid = np.concatenate([sample.availability.reshape(-1) for sample in samples])
    return {
        "video_count": len(samples),
        "sequence_count": len(samples),
        "timestep_count": sum(sample.residuals.shape[0] for sample in samples),
        "channel_count": len(RESIDUAL_NAMES),
        "available_value_count": int(np.count_nonzero(valid)),
        "unavailable_value_count": int(np.count_nonzero(~valid)),
        "availability_rate": float(np.mean(valid)),
        "valid_nan_count": int(np.count_nonzero(np.isnan(values[valid]))),
        "valid_inf_count": int(np.count_nonzero(np.isinf(values[valid]))),
        "all_missing_sequence_count": sum(
            not bool(np.any(sample.availability)) for sample in samples
        ),
        "label_0_count": sum(sample.label == 0 for sample in samples),
        "label_1_count": sum(sample.label == 1 for sample in samples),
    }


__all__ = [
    "REQUIRED_MANIFEST_FIELDS",
    "RESIDUAL_NAMES",
    "ResidualSequence",
    "ResidualSequenceDataset",
    "TrainingManifestBundle",
    "audit_split_leakage",
    "build_manifest_samples",
    "collate_residual_sequences",
    "load_channel_schema",
    "residual_channel_schema",
    "sequence_statistics",
    "validate_channel_schema",
]
