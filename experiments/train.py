"""Minimal training directly from frozen residual-sequence JSON outputs."""

from __future__ import annotations

import csv
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from models.temporal_head import ResidualTemporalHead

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


@dataclass(frozen=True)
class ResidualSequence:
    residuals: np.ndarray
    availability: np.ndarray
    confidence: np.ndarray
    label: int
    sample_id: str = ""
    clip_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class TrainingManifestBundle:
    samples: Mapping[str, tuple[ResidualSequence, ...]]
    sample_ids: Mapping[str, tuple[str, ...]]
    source_commit: str
    config_sha256: str
    channel_schema: Mapping[str, Any]


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
        )


def residual_channel_schema(source_commit: str, config_sha256: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "channel_count": len(RESIDUAL_NAMES),
        "channel_names": list(RESIDUAL_NAMES),
        "source_commit": str(source_commit),
        "config_sha256": str(config_sha256),
        "normalization": (
            "producer_normalized_value=1-exp(-raw_value), then arithmetic mean "
            "over valid evidence rows within each frozen clip"
        ),
        "missing_value_policy": (
            "unavailable values remain NaN with availability=false and confidence=0; "
            "the model fills zero only after multiplying by the availability mask"
        ),
        "timestep_definition": "one frozen inference clip ordered by start_frame",
        "label_is_input_channel": False,
    }


def load_channel_schema(path: str | Path) -> dict[str, Any]:
    schema = json.loads(Path(path).read_text(encoding="utf-8"))
    if int(schema.get("channel_count", -1)) != len(RESIDUAL_NAMES):
        raise ValueError("Residual channel schema must contain exactly 12 channels.")
    if tuple(schema.get("channel_names", ())) != RESIDUAL_NAMES:
        raise ValueError("Residual channel order does not match the frozen producer order.")
    if bool(schema.get("label_is_input_channel", True)):
        raise ValueError("Authenticity label must not be a residual input channel.")
    return schema


def _sequence_from_payload(
    payload: Mapping[str, Any],
    *,
    label: int,
    sample_id: str,
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
            if name not in RESIDUAL_NAMES or not bool(row.get("valid_mask", False)):
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
        residuals,
        availability,
        confidence,
        int(label),
        sample_id=sample_id,
        clip_ids=tuple(str(clip["clip_id"]) for clip in clips),
    )


def _read_training_manifest(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "sample_id",
        "label",
        "split",
        "residual_sequence_path",
        "source_video_path",
        "source_commit",
        "config_sha256",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError("Mini-training manifest is missing required fields.")
    return rows


def build_manifest_samples(
    manifest_path: str | Path,
    channel_schema_path: str | Path,
) -> TrainingManifestBundle:
    """Load frozen result JSON files without constructing visual providers."""

    schema = load_channel_schema(channel_schema_path)
    rows = _read_training_manifest(manifest_path)
    output: dict[str, list[ResidualSequence]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    identities: dict[str, set[str]] = {
        "sample_id": set(),
        "residual_sequence_path": set(),
        "source_video_path": set(),
    }
    split_values: dict[str, dict[str, set[str]]] = {
        split: {key: set() for key in (*identities, "clip_id")}
        for split in output
    }
    source_commits: set[str] = set()
    config_hashes: set[str] = set()
    for row in rows:
        split = row["split"].strip()
        if split not in output:
            raise ValueError(f"Unsupported split {split!r}.")
        sample_id = row["sample_id"].strip()
        if not sample_id or sample_id in identities["sample_id"]:
            raise ValueError(f"Duplicate or empty sample_id: {sample_id!r}.")
        label = int(row["label"])
        if label not in {0, 1}:
            raise ValueError("Training labels must be 0 or 1.")
        residual_path = Path(row["residual_sequence_path"]).resolve()
        source_video = Path(row["source_video_path"]).resolve()
        if not residual_path.is_file() or not source_video.is_file():
            raise FileNotFoundError(f"{sample_id}: source path is missing.")
        payload = json.loads(residual_path.read_text(encoding="utf-8"))
        if Path(str(payload.get("video_path", ""))).resolve() != source_video:
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
        )
        output[split].append(sequence)
        source_commit = row["source_commit"].strip()
        config_sha256 = row["config_sha256"].strip()
        source_commits.add(source_commit)
        config_hashes.add(config_sha256)
        values = {
            "sample_id": sample_id,
            "residual_sequence_path": str(residual_path),
            "source_video_path": str(source_video),
        }
        for key, value in values.items():
            identities[key].add(value)
            split_values[split][key].add(value)
        split_values[split]["clip_id"].update(sequence.clip_ids)
    if len(source_commits) != 1 or source_commits != {str(schema["source_commit"])}:
        raise ValueError("Manifest and channel schema source commits differ.")
    if len(config_hashes) != 1 or config_hashes != {str(schema["config_sha256"])}:
        raise ValueError("Manifest and channel schema config identities differ.")
    for key in ("sample_id", "residual_sequence_path", "source_video_path", "clip_id"):
        if split_values["train"][key] & split_values["validation"][key]:
            raise ValueError(f"Train/validation leakage detected for {key}.")
    if output["test"]:
        raise ValueError("R4-B0 requires an empty test split.")
    if not output["train"] or not output["validation"]:
        raise ValueError("Manifest requires non-empty train and validation splits.")
    return TrainingManifestBundle(
        samples={key: tuple(value) for key, value in output.items()},
        sample_ids={
            key: tuple(sample.sample_id for sample in value)
            for key, value in output.items()
        },
        source_commit=next(iter(source_commits)),
        config_sha256=next(iter(config_hashes)),
        channel_schema=schema,
    )


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
        "nan_count": int(np.count_nonzero(np.isnan(values))),
        "inf_count": int(np.count_nonzero(np.isinf(values))),
        "valid_nan_count": int(np.count_nonzero(np.isnan(values[valid]))),
        "valid_inf_count": int(np.count_nonzero(np.isinf(values[valid]))),
        "all_missing_sequence_count": sum(
            not bool(np.any(sample.availability)) for sample in samples
        ),
        "label_0_count": sum(sample.label == 0 for sample in samples),
        "label_1_count": sum(sample.label == 1 for sample in samples),
    }


def _collate(batch):
    max_steps = max(item[0].shape[0] for item in batch)
    residual_count = batch[0][0].shape[1]
    residuals = torch.full((len(batch), max_steps, residual_count), float("nan"))
    availability = torch.zeros((len(batch), max_steps, residual_count), dtype=torch.bool)
    confidence = torch.zeros((len(batch), max_steps, residual_count))
    labels = torch.empty(len(batch))
    for index, (values, valid, quality, label) in enumerate(batch):
        steps = values.shape[0]
        residuals[index, :steps] = values
        availability[index, :steps] = valid
        confidence[index, :steps] = quality
        labels[index] = label
    return residuals, availability, confidence, labels


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = labels == 1
    negatives = labels == 0
    if not np.any(positives) or not np.any(negatives):
        return float("nan")
    comparisons = scores[positives, None] - scores[None, negatives]
    return float(np.mean((comparisons > 0) + 0.5 * (comparisons == 0)))


def binary_metrics(labels: Sequence[int], logits: Sequence[float]) -> dict[str, float]:
    truth = np.asarray(labels, dtype=int)
    raw = np.asarray(logits, dtype=float)
    predicted = raw >= 0.0
    tp = int(np.sum(predicted & (truth == 1)))
    fp = int(np.sum(predicted & (truth == 0)))
    fn = int(np.sum(~predicted & (truth == 1)))
    accuracy = float(np.mean(predicted == truth))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": _auc(truth, raw),
    }


def _loader(
    samples: Sequence[ResidualSequence],
    *,
    batch_size: int,
    shuffle: bool,
    random_seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(random_seed)
    return DataLoader(
        ResidualSequenceDataset(samples),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=_collate,
        generator=generator,
    )


def _run_epoch(
    model: ResidualTemporalHead,
    loader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp: bool,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    criterion = nn.BCEWithLogitsLoss()
    losses: list[float] = []
    labels: list[int] = []
    logits: list[float] = []
    for residuals, availability, confidence, target in loader:
        residuals = residuals.to(device)
        availability = availability.to(device)
        confidence = confidence.to(device)
        target = target.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp and device.type == "cuda",
            ):
                output = model(residuals, availability, confidence)
                loss = criterion(output, target)
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        losses.append(float(loss.detach().cpu()))
        labels.extend(target.detach().cpu().int().tolist())
        logits.extend(float(value) for value in output.detach().cpu().tolist())
    return {
        "loss": float(np.mean(losses)),
        **binary_metrics(labels, logits),
        "labels": labels,
        "logits": logits,
    }


def evaluate_residual_head(
    model: ResidualTemporalHead,
    samples: Sequence[ResidualSequence],
    *,
    batch_size: int,
    device: str,
    amp: bool,
) -> dict[str, Any]:
    torch_device = torch.device(device)
    loader = _loader(
        samples,
        batch_size=batch_size,
        shuffle=False,
        random_seed=0,
    )
    scaler = torch.amp.GradScaler(
        device=torch_device.type,
        enabled=amp and torch_device.type == "cuda",
    )
    return _run_epoch(
        model,
        loader,
        optimizer=None,
        scaler=scaler,
        device=torch_device,
        amp=amp,
    )


def _json_value(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _write_training_outputs(
    output: Path,
    history: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> None:
    fields = (
        "epoch",
        "train_loss",
        "validation_loss",
        "train_video_count",
        "validation_video_count",
        "learning_rate",
        "runtime_seconds",
        "peak_gpu_memory_mb",
        "checkpoint_path",
        "validation_accuracy",
        "validation_precision",
        "validation_recall",
        "validation_f1",
        "validation_auc",
    )
    with (output / "training_history.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in history:
            writer.writerow(
                {
                    key: (
                        "unavailable"
                        if isinstance(row.get(key), float)
                        and not math.isfinite(float(row[key]))
                        else row.get(key)
                    )
                    for key in fields
                }
            )
    (output / "training_summary.json").write_text(
        json.dumps(_json_value(dict(summary)), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def train_residual_head(
    train_samples: Sequence[ResidualSequence],
    validation_samples: Sequence[ResidualSequence],
    *,
    output_dir: str | Path,
    channel_schema: Mapping[str, Any],
    source_commit: str,
    config_sha256: str,
    epochs: int = 3,
    hidden_size: int = 32,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 2,
    random_seed: int = 42,
    resume: str | Path | None = None,
    device: str = "cuda",
    amp: bool = True,
) -> tuple[ResidualTemporalHead, list[dict[str, Any]]]:
    if not 3 <= epochs <= 5:
        raise ValueError("Paper-core training is intentionally limited to 3-5 epochs.")
    if tuple(channel_schema.get("channel_names", ())) != RESIDUAL_NAMES:
        raise ValueError("Training channel order differs from the frozen schema.")
    if str(channel_schema.get("source_commit")) != source_commit:
        raise ValueError("Training source commit differs from the channel schema.")
    if str(channel_schema.get("config_sha256")) != config_sha256:
        raise ValueError("Training config differs from the channel schema.")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("R4-B0 requires CUDA, but CUDA is unavailable.")
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    residual_count = int(train_samples[0].residuals.shape[-1])
    model = ResidualTemporalHead(residual_count, hidden_size=hidden_size)
    if model.parameter_count >= 100_000:
        raise ValueError("Temporal fusion head exceeds the 100k parameter limit.")
    torch_device = torch.device(device)
    model.to(torch_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scaler = torch.amp.GradScaler(
        device=torch_device.type,
        enabled=amp and torch_device.type == "cuda",
    )
    train_ids = [sample.sample_id for sample in train_samples]
    validation_ids = [sample.sample_id for sample in validation_samples]
    start_epoch = 0
    best_loss = math.inf
    if resume is not None:
        checkpoint = torch.load(
            Path(resume),
            map_location=torch_device,
            weights_only=False,
        )
        required = {
            "model_state",
            "optimizer_state",
            "epoch",
            "random_seed",
            "channel_schema",
            "source_commit",
            "config_sha256",
            "train_sample_ids",
            "validation_sample_ids",
        }
        if not required.issubset(checkpoint):
            raise ValueError("Resume checkpoint is missing required provenance.")
        if checkpoint["channel_schema"] != dict(channel_schema):
            raise ValueError("Resume checkpoint channel schema mismatch.")
        if checkpoint["source_commit"] != source_commit:
            raise ValueError("Resume checkpoint source commit mismatch.")
        if checkpoint["config_sha256"] != config_sha256:
            raise ValueError("Resume checkpoint config mismatch.")
        if checkpoint["train_sample_ids"] != train_ids:
            raise ValueError("Resume checkpoint train split mismatch.")
        if checkpoint["validation_sample_ids"] != validation_ids:
            raise ValueError("Resume checkpoint validation split mismatch.")
        if int(checkpoint["random_seed"]) != random_seed:
            raise ValueError("Resume checkpoint random seed mismatch.")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        if checkpoint.get("amp_scaler_state"):
            scaler.load_state_dict(checkpoint["amp_scaler_state"])
        start_epoch = int(checkpoint["epoch"])
        best_loss = float(checkpoint["best_validation_loss"])
        if epochs <= start_epoch:
            raise ValueError("Resume target epoch must exceed checkpoint epoch.")
    train_loader = _loader(
        train_samples,
        batch_size=batch_size,
        shuffle=True,
        random_seed=random_seed,
    )
    validation_loader = _loader(
        validation_samples,
        batch_size=batch_size,
        shuffle=False,
        random_seed=random_seed,
    )
    history: list[dict[str, Any]] = []
    for epoch in range(start_epoch + 1, epochs + 1):
        if torch_device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(torch_device)
            torch.cuda.synchronize(torch_device)
        started = time.perf_counter()
        train_metrics = _run_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=torch_device,
            amp=amp,
        )
        validation_metrics = _run_epoch(
            model,
            validation_loader,
            optimizer=None,
            scaler=scaler,
            device=torch_device,
            amp=amp,
        )
        if torch_device.type == "cuda":
            torch.cuda.synchronize(torch_device)
            peak_memory = torch.cuda.max_memory_allocated(torch_device) / 1024**2
        else:
            peak_memory = 0.0
        checkpoint_path = output / "last.pt"
        record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "validation_loss": validation_metrics["loss"],
            "train_video_count": len(train_samples),
            "validation_video_count": len(validation_samples),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "runtime_seconds": time.perf_counter() - started,
            "peak_gpu_memory_mb": float(peak_memory),
            "checkpoint_path": str(checkpoint_path.resolve()),
            "validation_accuracy": validation_metrics["accuracy"],
            "validation_precision": validation_metrics["precision"],
            "validation_recall": validation_metrics["recall"],
            "validation_f1": validation_metrics["f1"],
            "validation_auc": validation_metrics["auc"],
            "validation_logits": validation_metrics["logits"],
            "validation_labels": validation_metrics["labels"],
        }
        history.append(record)
        current_best = min(best_loss, validation_metrics["loss"])
        state = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "amp_scaler_state": scaler.state_dict(),
            "epoch": epoch,
            "random_seed": random_seed,
            "channel_schema": dict(channel_schema),
            "source_commit": source_commit,
            "config_sha256": config_sha256,
            "train_sample_ids": train_ids,
            "validation_sample_ids": validation_ids,
            "best_validation_loss": current_best,
            "model_config": {
                "residual_count": residual_count,
                "hidden_size": hidden_size,
                "parameter_count": model.parameter_count,
            },
            "optimizer_name": "AdamW",
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "amp": bool(amp and torch_device.type == "cuda"),
        }
        torch.save(state, checkpoint_path)
        if validation_metrics["loss"] < best_loss:
            best_loss = validation_metrics["loss"]
            state["best_validation_loss"] = best_loss
            torch.save(state, output / "best_validation_loss.pt")
    summary = {
        "status": "engineering_smoke_only",
        "performance_conclusion_allowed": False,
        "resume_from": str(Path(resume).resolve()) if resume else None,
        "epochs_completed_this_invocation": [row["epoch"] for row in history],
        "model_parameter_count": model.parameter_count,
        "random_seed": random_seed,
        "batch_size": batch_size,
        "shuffle_train": True,
        "shuffle_validation": False,
        "num_workers": 0,
        "device": str(torch_device),
        "amp": bool(amp and torch_device.type == "cuda"),
        "optimizer": "AdamW",
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "channel_schema": dict(channel_schema),
        "source_commit": source_commit,
        "config_sha256": config_sha256,
        "train_sample_ids": train_ids,
        "validation_sample_ids": validation_ids,
        "test_sample_ids": [],
        "train_statistics": sequence_statistics(train_samples),
        "validation_statistics": sequence_statistics(validation_samples),
        "history": history,
        "final_validation_logits": history[-1]["validation_logits"],
        "final_validation_labels": history[-1]["validation_labels"],
    }
    _write_training_outputs(output, history, summary)
    return model, history


__all__ = [
    "ResidualSequence",
    "ResidualSequenceDataset",
    "TrainingManifestBundle",
    "RESIDUAL_NAMES",
    "binary_metrics",
    "build_manifest_samples",
    "evaluate_residual_head",
    "load_channel_schema",
    "residual_channel_schema",
    "sequence_statistics",
    "train_residual_head",
]
