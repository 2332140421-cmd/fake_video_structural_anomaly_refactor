"""Missing-aware training from frozen paper-core residual sequences."""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from data.residual_dataset import (
    RESIDUAL_NAMES,
    ResidualSequence,
    ResidualSequenceDataset,
    TrainingManifestBundle,
    build_manifest_samples,
    collate_residual_sequences,
    load_channel_schema,
    residual_channel_schema,
    sequence_statistics,
    validate_channel_schema,
)
from experiments.artifacts import (
    prediction_rows,
    write_coverage_artifacts,
    write_metric_artifacts,
    write_training_figures,
)
from experiments.metrics import binary_classification_metrics, sigmoid
from models.temporal_head import ResidualTemporalHead
from utils.io import ensure_output_dir, json_safe, write_csv, write_json


def binary_metrics(
    labels: Sequence[int],
    logits: Sequence[float],
    classification_threshold: float = 0.5,
) -> dict[str, Any]:
    """Compatibility wrapper around the complete split-level metrics."""

    metrics = binary_classification_metrics(
        labels,
        logits,
        classification_threshold=classification_threshold,
    )
    return {**metrics, "auc": metrics["roc_auc"]}


def _collate(batch):
    """Kept importable for focused padding tests."""

    return collate_residual_sequences(batch)


def _loader(
    samples: Sequence[ResidualSequence],
    *,
    batch_size: int,
    shuffle: bool,
    random_seed: int,
    num_workers: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(random_seed)
    return DataLoader(
        ResidualSequenceDataset(samples),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_residual_sequences,
        generator=generator,
    )


def _gpu_memory(device: torch.device) -> tuple[float, float]:
    if device.type != "cuda":
        return 0.0, 0.0
    return (
        torch.cuda.memory_allocated(device) / 1024**2,
        torch.cuda.memory_reserved(device) / 1024**2,
    )


def _run_epoch(
    model: ResidualTemporalHead,
    loader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp: bool,
    classification_threshold: float,
    epoch: int,
    total_epochs: int,
    global_step: int,
    log_every: int,
    phase_name: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    training = optimizer is not None
    model.train(training)
    criterion = nn.BCEWithLogitsLoss()
    weighted_loss = 0.0
    labels: list[int] = []
    logits: list[float] = []
    sample_ids: list[str] = []
    batch_rows: list[dict[str, Any]] = []
    samples_seen = 0
    iterator = iter(loader)
    for batch_index in range(1, len(loader) + 1):
        data_started = time.perf_counter()
        (
            residuals,
            availability,
            confidence,
            padding_mask,
            target,
            batch_sample_ids,
        ) = next(iterator)
        data_seconds = time.perf_counter() - data_started
        step_started = time.perf_counter()
        residuals = residuals.to(device)
        availability = availability.to(device)
        confidence = confidence.to(device)
        padding_mask = padding_mask.to(device)
        target = target.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp and device.type == "cuda",
            ):
                output = model(
                    residuals,
                    availability,
                    confidence,
                    padding_mask,
                )
                loss = criterion(output, target)
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                global_step += 1
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        step_seconds = time.perf_counter() - step_started
        batch_size = len(batch_sample_ids)
        samples_seen += batch_size
        weighted_loss += float(loss.detach().cpu()) * batch_size
        labels.extend(target.detach().cpu().int().tolist())
        logits.extend(float(value) for value in output.detach().cpu().tolist())
        sample_ids.extend(batch_sample_ids)
        real_positions = padding_mask.unsqueeze(-1).expand_as(availability)
        available_count = torch.count_nonzero(availability & real_positions).item()
        real_count = torch.count_nonzero(real_positions).item()
        allocated, reserved = _gpu_memory(device)
        row = {
            "phase": (
                str(phase_name).lower()
                if phase_name
                else ("train" if training else "validation")
            ),
            "epoch": epoch,
            "total_epochs": total_epochs,
            "batch": batch_index,
            "total_batches": len(loader),
            "samples_seen": samples_seen,
            "total_samples": len(loader.dataset),
            "batch_loss": float(loss.detach().cpu()),
            "running_loss": weighted_loss / samples_seen,
            "learning_rate": (
                optimizer.param_groups[0]["lr"] if optimizer is not None else None
            ),
            "availability_rate": available_count / max(real_count, 1),
            "data_time_seconds": data_seconds,
            "step_time_seconds": step_seconds,
            "batch_time_seconds": step_seconds,
            "samples_per_second": batch_size / max(step_seconds, 1e-12),
            "gpu_memory_allocated_mb": allocated,
            "gpu_memory_reserved_mb": reserved,
            "gpu_peak_memory_mb": (
                torch.cuda.max_memory_allocated(device) / 1024**2
                if device.type == "cuda"
                else 0.0
            ),
            "global_step": global_step,
        }
        batch_rows.append(row)
    metrics = binary_classification_metrics(
        labels,
        logits,
        classification_threshold=classification_threshold,
    )
    return (
        {
            "loss": weighted_loss / len(labels),
            **metrics,
            "labels": labels,
            "logits": logits,
            "probabilities": sigmoid(logits).tolist(),
            "sample_ids": sample_ids,
        },
        batch_rows,
        global_step,
    )


def evaluate_residual_head(
    model: ResidualTemporalHead,
    samples: Sequence[ResidualSequence],
    *,
    batch_size: int,
    device: str,
    amp: bool,
    num_workers: int = 0,
    classification_threshold: float = 0.5,
) -> dict[str, Any]:
    torch_device = torch.device(device)
    loader = _loader(
        samples,
        batch_size=batch_size,
        shuffle=False,
        random_seed=0,
        num_workers=num_workers,
    )
    scaler = torch.amp.GradScaler(
        device=torch_device.type,
        enabled=amp and torch_device.type == "cuda",
    )
    metrics, _, _ = _run_epoch(
        model,
        loader,
        optimizer=None,
        scaler=scaler,
        device=torch_device,
        amp=amp,
        classification_threshold=classification_threshold,
        epoch=0,
        total_epochs=0,
        global_step=0,
        log_every=1,
    )
    return metrics


def _metric_display(value: Any) -> str:
    return "unavailable" if value is None else f"{float(value):.6f}"


def _print_epoch(
    epoch: int,
    total_epochs: int,
    train: Mapping[str, Any],
    validation: Mapping[str, Any],
    *,
    learning_rate: float,
    runtime: float,
    peak_memory: float,
) -> None:
    def metrics(prefix: str, values: Mapping[str, Any]) -> str:
        return " ".join(
            (
                f"{prefix} loss={_metric_display(values['loss'])}",
                f"acc={_metric_display(values['accuracy'])}",
                f"precision={_metric_display(values['precision'])}",
                f"recall={_metric_display(values['recall'])}",
                f"specificity={_metric_display(values['specificity'])}",
                f"f1={_metric_display(values['f1'])}",
                f"roc_auc={_metric_display(values['roc_auc'])}",
                f"pr_auc={_metric_display(values['pr_auc'])}",
            )
        )

    print(
        f"Epoch {epoch:03d}/{total_epochs:03d} | "
        f"{metrics('train', train)} | "
        f"{metrics('val', validation)} | "
        f"lr={learning_rate:.6e} time={runtime:.3f}s "
        f"gpu_peak={peak_memory:.1f}MiB",
        flush=True,
    )


def _checkpoint_state(
    *,
    model: ResidualTemporalHead,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    epoch: int,
    global_step: int,
    random_seed: int,
    model_config: Mapping[str, Any],
    training_config: Mapping[str, Any],
    channel_schema: Mapping[str, Any],
    source_commit: str,
    source_config_sha256: str,
    manifest_sha256: str,
    train_sample_ids: Sequence[str],
    validation_sample_ids: Sequence[str],
    classification_threshold: float,
    best_validation_loss: float,
) -> dict[str, Any]:
    return {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "amp_scaler_state": scaler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "random_seed": random_seed,
        "model_config": dict(model_config),
        "training_config": dict(training_config),
        "channel_schema": dict(channel_schema),
        "source_commit": source_commit,
        "source_config_sha256": source_config_sha256,
        "manifest_sha256": manifest_sha256,
        "train_sample_ids": list(train_sample_ids),
        "validation_sample_ids": list(validation_sample_ids),
        "classification_threshold": classification_threshold,
        "best_validation_loss": best_validation_loss,
    }


def train_residual_head(
    train_samples: Sequence[ResidualSequence],
    validation_samples: Sequence[ResidualSequence],
    *,
    output_dir: str | Path,
    channel_schema: Mapping[str, Any],
    source_commit: str,
    config_sha256: str | None = None,
    source_config_sha256: str | None = None,
    manifest_sha256: str = "unit-test-manifest",
    epochs: int = 3,
    hidden_size: int = 32,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 2,
    random_seed: int = 42,
    resume: str | Path | None = None,
    device: str = "cuda",
    amp: bool = True,
    num_workers: int = 0,
    log_every: int = 1,
    checkpoint_every: int = 1,
    classification_threshold: float = 0.5,
    progress_mode: str = "epoch",
    bundle: TrainingManifestBundle | None = None,
) -> tuple[ResidualTemporalHead, list[dict[str, Any]]]:
    if epochs < 1:
        raise ValueError("epochs must be an integer greater than or equal to 1.")
    if log_every < 1 or checkpoint_every < 1:
        raise ValueError("log_every and checkpoint_every must be positive.")
    if progress_mode not in {"epoch", "none"}:
        raise ValueError("progress_mode must be 'epoch' or 'none'.")
    schema = validate_channel_schema(channel_schema)
    producer_config = source_config_sha256 or config_sha256
    if not producer_config:
        raise ValueError("source_config_sha256 is required.")
    if str(schema["source_commit"]) != source_commit:
        raise ValueError("Training source commit differs from the channel schema.")
    if str(schema["source_config_sha256"]) != producer_config:
        raise ValueError("Training producer config differs from the channel schema.")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but CUDA is unavailable.")
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    output = ensure_output_dir(output_dir)
    checkpoints = ensure_output_dir(output / "checkpoints")
    for directory in ("logs", "metrics", "predictions", "coverage", "efficiency", "figures"):
        ensure_output_dir(output / directory)
    residual_count = int(train_samples[0].residuals.shape[-1])
    model_config = {
        "name": "ResidualTemporalHead",
        "residual_count": residual_count,
        "hidden_size": hidden_size,
        "expected_input_channels": len(RESIDUAL_NAMES),
    }
    training_config = {
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "seed": random_seed,
        "num_workers": num_workers,
        "log_every": log_every,
        "checkpoint_every": checkpoint_every,
        "amp": bool(amp),
        "device": device,
        "classification_threshold": classification_threshold,
        "progress": {"mode": progress_mode},
    }
    model = ResidualTemporalHead(residual_count, hidden_size=hidden_size)
    if model.parameter_count >= 100_000:
        raise ValueError("Temporal fusion head exceeds the 100k parameter limit.")
    print(
        f"[RUN] output={output.resolve()} device={device} "
        f"train_samples={len(train_samples)} "
        f"validation_samples={len(validation_samples)} "
        f"batch_size={batch_size} epochs={epochs} "
        f"parameter_count={model.parameter_count}",
        flush=True,
    )
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
    global_step = 0
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
            "amp_scaler_state",
            "epoch",
            "global_step",
            "random_seed",
            "model_config",
            "training_config",
            "channel_schema",
            "source_commit",
            "source_config_sha256",
            "manifest_sha256",
            "train_sample_ids",
            "validation_sample_ids",
            "classification_threshold",
            "best_validation_loss",
        }
        missing = sorted(required - set(checkpoint))
        if missing:
            raise ValueError(f"Resume checkpoint is missing provenance: {missing}.")
        exact = {
            "model_config": model_config,
            "training_config": training_config,
            "channel_schema": schema,
            "source_commit": source_commit,
            "source_config_sha256": producer_config,
            "manifest_sha256": manifest_sha256,
            "train_sample_ids": train_ids,
            "validation_sample_ids": validation_ids,
            "classification_threshold": classification_threshold,
            "random_seed": random_seed,
        }
        for key, expected in exact.items():
            if checkpoint[key] != expected:
                raise ValueError(f"Resume checkpoint {key} mismatch.")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scaler.load_state_dict(checkpoint["amp_scaler_state"])
        start_epoch = int(checkpoint["epoch"])
        global_step = int(checkpoint["global_step"])
        best_loss = float(checkpoint["best_validation_loss"])
        if epochs <= start_epoch:
            raise ValueError("Resume target epoch must exceed checkpoint epoch.")
    train_loader = _loader(
        train_samples,
        batch_size=batch_size,
        shuffle=True,
        random_seed=random_seed,
        num_workers=num_workers,
    )
    train_evaluation_loader = _loader(
        train_samples,
        batch_size=batch_size,
        shuffle=False,
        random_seed=random_seed,
        num_workers=num_workers,
    )
    validation_loader = _loader(
        validation_samples,
        batch_size=batch_size,
        shuffle=False,
        random_seed=random_seed,
        num_workers=num_workers,
    )
    history: list[dict[str, Any]] = []
    batch_history: list[dict[str, Any]] = []
    final_train: dict[str, Any] = {}
    final_validation: dict[str, Any] = {}
    invocation_started = time.perf_counter()
    peak_overall = 0.0
    forward_count = 0
    backward_count = 0
    nonfinite_loss_count = 0
    data_loading_seconds = 0.0
    epoch_metrics_path = output / "logs" / "epoch_metrics.jsonl"
    if resume is None:
        epoch_metrics_path.write_text("", encoding="utf-8")
    for epoch in range(start_epoch + 1, epochs + 1):
        if torch_device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(torch_device)
            torch.cuda.synchronize(torch_device)
        started = time.perf_counter()
        _, train_batches, global_step = _run_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=torch_device,
            amp=amp,
            classification_threshold=classification_threshold,
            epoch=epoch,
            total_epochs=epochs,
            global_step=global_step,
            log_every=log_every,
            phase_name="Train",
        )
        train_metrics, train_evaluation_batches, global_step = _run_epoch(
            model,
            train_evaluation_loader,
            optimizer=None,
            scaler=scaler,
            device=torch_device,
            amp=amp,
            classification_threshold=classification_threshold,
            epoch=epoch,
            total_epochs=epochs,
            global_step=global_step,
            log_every=log_every,
            phase_name="TrainEval",
        )
        validation_metrics, validation_batches, global_step = _run_epoch(
            model,
            validation_loader,
            optimizer=None,
            scaler=scaler,
            device=torch_device,
            amp=amp,
            classification_threshold=classification_threshold,
            epoch=epoch,
            total_epochs=epochs,
            global_step=global_step,
            log_every=log_every,
            phase_name="Validation",
        )
        if torch_device.type == "cuda":
            torch.cuda.synchronize(torch_device)
            peak_memory = torch.cuda.max_memory_allocated(torch_device) / 1024**2
        else:
            peak_memory = 0.0
        peak_overall = max(peak_overall, peak_memory)
        runtime = time.perf_counter() - started
        epoch_rows = (
            train_batches + train_evaluation_batches + validation_batches
        )
        epoch_data_loading_seconds = sum(
            float(row["data_time_seconds"]) for row in epoch_rows
        )
        epoch_nonfinite_loss_count = sum(
            not math.isfinite(float(row["batch_loss"])) for row in epoch_rows
        )
        epoch_forward_count = len(epoch_rows)
        epoch_backward_count = len(train_batches)
        data_loading_seconds += epoch_data_loading_seconds
        nonfinite_loss_count += epoch_nonfinite_loss_count
        forward_count += epoch_forward_count
        backward_count += epoch_backward_count
        if epoch_nonfinite_loss_count:
            raise ValueError("Training or validation loss contains NaN or Inf.")
        current_best = min(best_loss, float(validation_metrics["loss"]))
        state = _checkpoint_state(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            global_step=global_step,
            random_seed=random_seed,
            model_config=model_config,
            training_config=training_config,
            channel_schema=schema,
            source_commit=source_commit,
            source_config_sha256=producer_config,
            manifest_sha256=manifest_sha256,
            train_sample_ids=train_ids,
            validation_sample_ids=validation_ids,
            classification_threshold=classification_threshold,
            best_validation_loss=current_best,
        )
        checkpoint_path = checkpoints / "last.pt"
        if epoch % checkpoint_every == 0 or epoch == epochs:
            torch.save(state, checkpoint_path)
        if float(validation_metrics["loss"]) < best_loss:
            best_loss = float(validation_metrics["loss"])
            state["best_validation_loss"] = best_loss
            torch.save(state, checkpoints / "best_validation_loss.pt")
        record: dict[str, Any] = {
            "epoch": epoch,
            "total_epochs": epochs,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "runtime_seconds": runtime,
            "peak_gpu_memory_mb": peak_memory,
            "checkpoint_path": str(checkpoint_path.resolve()),
            "global_step": global_step,
            "forward_count": epoch_forward_count,
            "backward_count": epoch_backward_count,
            "optimizer_step_count": len(train_batches),
            "nan_inf_loss_count": epoch_nonfinite_loss_count,
            "data_loading_seconds": epoch_data_loading_seconds,
        }
        for prefix, metrics in (
            ("train", train_metrics),
            ("validation", validation_metrics),
        ):
            for name in (
                "loss",
                "accuracy",
                "precision",
                "recall",
                "specificity",
                "f1",
                "roc_auc",
                "pr_auc",
                "tp",
                "tn",
                "fp",
                "fn",
            ):
                record[f"{prefix}_{name}"] = metrics[name]
        history.append(record)
        with epoch_metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(json_safe(record), ensure_ascii=False) + "\n")
            handle.flush()
        batch_history.extend(train_batches)
        batch_history.extend(train_evaluation_batches)
        batch_history.extend(validation_batches)
        final_train = train_metrics
        final_validation = validation_metrics
        if progress_mode == "epoch":
            _print_epoch(
                epoch,
                epochs,
                train_metrics,
                validation_metrics,
                learning_rate=optimizer.param_groups[0]["lr"],
                runtime=runtime,
                peak_memory=peak_memory,
            )
    batch_fields = list(batch_history[0])
    write_csv(output / "logs" / "batch_history.csv", batch_history, batch_fields)
    epoch_fields = list(history[0])
    write_csv(output / "logs" / "epoch_history.csv", history, epoch_fields)
    train_predictions = prediction_rows(
        final_train["sample_ids"],
        final_train["labels"],
        final_train["logits"],
        final_train["probabilities"],
        threshold=classification_threshold,
    )
    validation_predictions = prediction_rows(
        final_validation["sample_ids"],
        final_validation["labels"],
        final_validation["logits"],
        final_validation["probabilities"],
        threshold=classification_threshold,
    )
    train_result = {
        key: value
        for key, value in final_train.items()
        if key not in {"labels", "logits", "probabilities", "sample_ids"}
    }
    validation_result = {
        key: value
        for key, value in final_validation.items()
        if key not in {"labels", "logits", "probabilities", "sample_ids"}
    }
    write_metric_artifacts(
        output,
        split="train",
        metrics=train_result,
        predictions=train_predictions,
    )
    write_metric_artifacts(
        output,
        split="validation",
        metrics=validation_result,
        predictions=validation_predictions,
        write_shared_curves=True,
    )
    final_metrics = {
        "scope": "engineering_smoke_not_paper_performance",
        "performance_conclusion_allowed": False,
        "classification_threshold": classification_threshold,
        "train": train_result,
        "validation": validation_result,
    }
    write_json(output / "metrics" / "final_metrics.json", final_metrics)
    runtime_seconds = time.perf_counter() - invocation_started
    write_json(
        output / "efficiency" / "runtime_summary.json",
        {
            "invocation_runtime_seconds": runtime_seconds,
            "epochs_completed_this_invocation": [row["epoch"] for row in history],
        },
    )
    write_json(
        output / "efficiency" / "gpu_memory_summary.json",
        {
            "device": str(torch_device),
            "peak_gpu_memory_mb": peak_overall,
        },
    )
    write_json(
        output / "efficiency" / "model_summary.json",
        {
            **model_config,
            "parameter_count": model.parameter_count,
        },
    )
    write_json(
        output / "efficiency" / "execution_summary.json",
        {
            "forward_count": forward_count,
            "backward_count": backward_count,
            "optimizer_step_count": global_step,
            "nan_inf_loss_count": nonfinite_loss_count,
            "data_loading_seconds": data_loading_seconds,
            "train_batch_count": len(train_loader),
            "train_evaluation_batch_count": len(train_evaluation_loader),
            "validation_batch_count": len(validation_loader),
        },
    )
    summary = {
        "status": "engineering_smoke_only",
        "performance_conclusion_allowed": False,
        "resume_from": str(Path(resume).resolve()) if resume else None,
        "epochs_completed_this_invocation": [row["epoch"] for row in history],
        "global_step": global_step,
        "model_parameter_count": model.parameter_count,
        "training_config": training_config,
        "channel_schema": schema,
        "source_commit": source_commit,
        "source_config_sha256": producer_config,
        "manifest_sha256": manifest_sha256,
        "train_sample_ids": train_ids,
        "validation_sample_ids": validation_ids,
        "train_statistics": sequence_statistics(train_samples),
        "validation_statistics": sequence_statistics(validation_samples),
        "history": history,
        "execution_counts": {
            "forward": forward_count,
            "backward": backward_count,
            "optimizer_step": global_step,
            "nan_inf_loss": nonfinite_loss_count,
        },
        "final_train_logits": final_train["logits"],
        "final_validation_logits": final_validation["logits"],
    }
    write_json(output / "training_summary.json", summary)
    write_csv(output / "training_history.csv", history, epoch_fields)
    if bundle is not None:
        write_coverage_artifacts(output, bundle)
        write_training_figures(output, history, validation_predictions)
    return model, history


__all__ = [
    "RESIDUAL_NAMES",
    "ResidualSequence",
    "ResidualSequenceDataset",
    "TrainingManifestBundle",
    "_collate",
    "binary_metrics",
    "build_manifest_samples",
    "evaluate_residual_head",
    "load_channel_schema",
    "residual_channel_schema",
    "sequence_statistics",
    "train_residual_head",
]
