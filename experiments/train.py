"""Minimal residual-sequence training with checkpoint and resume."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from models.temporal_head import ResidualTemporalHead
from data.manifest import read_manifest

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


def video_result_to_sequence(result: Any, label: int) -> ResidualSequence:
    residuals = np.full((len(result.clip_results), len(RESIDUAL_NAMES)), np.nan, dtype=np.float32)
    availability = np.zeros_like(residuals, dtype=bool)
    confidence = np.zeros_like(residuals, dtype=np.float32)
    for time_index, clip in enumerate(result.clip_results):
        by_name: dict[str, list[Any]] = {}
        for row in clip.residuals:
            if row.valid_mask and row.name in RESIDUAL_NAMES:
                by_name.setdefault(row.name, []).append(row)
        for residual_index, name in enumerate(RESIDUAL_NAMES):
            rows = by_name.get(name, ())
            if rows:
                residuals[time_index, residual_index] = float(
                    np.mean([row.normalized_value for row in rows])
                )
                availability[time_index, residual_index] = True
                confidence[time_index, residual_index] = float(
                    np.mean([row.confidence for row in rows])
                )
    if not np.any(availability):
        raise ValueError("A training video must contain at least one valid residual.")
    return ResidualSequence(residuals, availability, confidence, int(label))


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


def build_manifest_samples(manifest_path: str | Path, pipeline: Any):
    manifest = Path(manifest_path).resolve()
    data_root = Path(os.environ.get("DATA_ROOT", manifest.parent))
    output: dict[str, list[ResidualSequence]] = {"train": [], "validation": [], "test": []}
    for row in read_manifest(manifest):
        if row.label is None:
            raise ValueError("Training/evaluation rows require labels.")
        path = Path(row.video_path)
        if not path.is_absolute():
            path = data_root / path
        output[row.split].append(video_result_to_sequence(pipeline.analyze_video(path), row.label))
    return output


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


def _run_epoch(
    model: ResidualTemporalHead,
    loader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> dict[str, float]:
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
        with torch.set_grad_enabled(training):
            output = model(residuals, availability, confidence)
            loss = criterion(output, target)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        losses.append(float(loss.detach().cpu()))
        labels.extend(target.detach().cpu().int().tolist())
        logits.extend(output.detach().cpu().tolist())
    return {"loss": float(np.mean(losses)), **binary_metrics(labels, logits)}


def train_residual_head(
    train_samples: Sequence[ResidualSequence],
    validation_samples: Sequence[ResidualSequence],
    *,
    output_dir: str | Path,
    epochs: int = 3,
    hidden_size: int = 32,
    learning_rate: float = 1e-3,
    batch_size: int = 8,
    resume: str | Path | None = None,
    device: str = "cpu",
) -> tuple[ResidualTemporalHead, list[dict[str, Any]]]:
    if not 3 <= epochs <= 5:
        raise ValueError("Paper-core training is intentionally limited to 3-5 epochs.")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    residual_count = int(train_samples[0].residuals.shape[-1])
    model = ResidualTemporalHead(residual_count, hidden_size=hidden_size)
    if model.parameter_count >= 100_000:
        raise ValueError("Temporal fusion head exceeds the 100k parameter limit.")
    torch_device = torch.device(device)
    model.to(torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    start_epoch = 0
    best_loss = math.inf
    if resume is not None:
        checkpoint = torch.load(Path(resume), map_location=torch_device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_loss = float(checkpoint["best_validation_loss"])
    train_loader = DataLoader(
        ResidualSequenceDataset(train_samples),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_collate,
    )
    validation_loader = DataLoader(
        ResidualSequenceDataset(validation_samples),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate,
    )
    history: list[dict[str, Any]] = []
    for epoch in range(start_epoch, start_epoch + epochs):
        train_metrics = _run_epoch(model, train_loader, optimizer=optimizer, device=torch_device)
        validation_metrics = _run_epoch(model, validation_loader, optimizer=None, device=torch_device)
        record = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(record)
        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_validation_loss": min(best_loss, validation_metrics["loss"]),
            "residual_count": residual_count,
            "hidden_size": hidden_size,
        }
        torch.save(state, output / "last.pt")
        if validation_metrics["loss"] < best_loss:
            best_loss = validation_metrics["loss"]
            state["best_validation_loss"] = best_loss
            torch.save(state, output / "best.pt")
    return model, history


__all__ = [
    "ResidualSequence",
    "ResidualSequenceDataset",
    "RESIDUAL_NAMES",
    "binary_metrics",
    "build_manifest_samples",
    "train_residual_head",
    "video_result_to_sequence",
]
