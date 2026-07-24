"""Configuration, train/validation loop, logging, and resume for A2."""

from __future__ import annotations

import json
import platform
import random
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml

from .checkpoint import restore_training_checkpoint, save_training_checkpoint
from .contracts import (
    TRAINING_CONFIG_SCHEMA_VERSION,
    FeatureContract,
    load_feature_contract,
)
from .data import TrainingDataBundle, build_training_dataloaders
from .loss import MaskedBinaryLoss
from .metrics import binary_validation_metrics
from .model import MinimalMissingAwareEvidenceHead


@dataclass(frozen=True)
class TrainingConfig:
    schema_version: str
    project_root: Path
    feature_contract: Path
    train_formal_manifest: Path | None
    train_manifest: Path | None
    validation_formal_manifest: Path | None
    validation_manifest: Path | None
    output_dir: Path
    resume_checkpoint: Path | None
    seed: int
    device: str
    epochs: int
    batch_size: int
    num_workers: int
    learning_rate: float
    weight_decay: float
    gradient_clip_norm: float
    amp: bool
    deterministic: bool
    scheduler: str
    scheduler_step_size: int
    scheduler_gamma: float
    pos_weight: str | None
    log_interval: int
    max_train_batches: int | None
    max_validation_batches: int | None
    hidden_dim: int
    dropout: float

    def snapshot(self) -> dict[str, Any]:
        payload = asdict(self)
        for key, value in list(payload.items()):
            if isinstance(value, Path):
                payload[key] = str(value)
        return payload


def _resolve(project_root: Path, value: Any) -> Path | None:
    if value is None or not str(value).strip():
        return None
    path = Path(str(value))
    return path if path.is_absolute() else project_root / path


def load_training_config(
    path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> TrainingConfig:
    """Load the checked-in YAML without guessing missing data paths."""

    config_path = Path(path).resolve()
    root = (
        Path(project_root).resolve()
        if project_root is not None
        else config_path.resolve().parents[1]
    )
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != TRAINING_CONFIG_SCHEMA_VERSION:
        raise ValueError("Unsupported A2 training config schema_version.")
    data = payload.get("data", {})
    training = payload.get("training", {})
    model = payload.get("model", {})
    return TrainingConfig(
        schema_version=str(payload["schema_version"]),
        project_root=root,
        feature_contract=_resolve(root, payload["feature_contract"]),
        train_formal_manifest=_resolve(root, data.get("train_formal_manifest")),
        train_manifest=_resolve(root, data.get("train_manifest")),
        validation_formal_manifest=_resolve(
            root, data.get("validation_formal_manifest")
        ),
        validation_manifest=_resolve(root, data.get("validation_manifest")),
        output_dir=_resolve(root, payload["output_dir"]),
        resume_checkpoint=_resolve(root, payload.get("resume_checkpoint")),
        seed=int(training.get("seed", 0)),
        device=str(training.get("device", "auto")),
        epochs=int(training.get("epochs", 1)),
        batch_size=int(training.get("batch_size", 1)),
        num_workers=int(training.get("num_workers", 0)),
        learning_rate=float(training.get("learning_rate", 1e-3)),
        weight_decay=float(training.get("weight_decay", 0.0)),
        gradient_clip_norm=float(training.get("gradient_clip_norm", 1.0)),
        amp=bool(training.get("amp", False)),
        deterministic=bool(training.get("deterministic", True)),
        scheduler=str(training.get("scheduler", "none")),
        scheduler_step_size=int(training.get("scheduler_step_size", 1)),
        scheduler_gamma=float(training.get("scheduler_gamma", 0.5)),
        pos_weight=(
            None
            if training.get("pos_weight") is None
            else str(training["pos_weight"])
        ),
        log_interval=int(training.get("log_interval", 1)),
        max_train_batches=(
            None
            if training.get("max_train_batches") is None
            else int(training["max_train_batches"])
        ),
        max_validation_batches=(
            None
            if training.get("max_validation_batches") is None
            else int(training["max_validation_batches"])
        ),
        hidden_dim=int(model.get("hidden_dim", 16)),
        dropout=float(model.get("dropout", 0.0)),
    )


def with_config_overrides(config: TrainingConfig, **overrides: Any) -> TrainingConfig:
    """Apply explicit CLI/test overrides to immutable configuration."""

    allowed = set(TrainingConfig.__dataclass_fields__)
    unknown = sorted(set(overrides) - allowed)
    if unknown:
        raise ValueError(f"Unknown training config overrides: {unknown}")
    converted = {}
    path_fields = {
        "feature_contract",
        "train_formal_manifest",
        "train_manifest",
        "validation_formal_manifest",
        "validation_manifest",
        "output_dir",
        "resume_checkpoint",
    }
    for key, value in overrides.items():
        converted[key] = None if value is None else Path(value) if key in path_fields else value
    return replace(config, **converted)


def _validate_config(config: TrainingConfig) -> None:
    required_paths = {
        "feature_contract": config.feature_contract,
        "train_formal_manifest": config.train_formal_manifest,
        "train_manifest": config.train_manifest,
        "validation_formal_manifest": config.validation_formal_manifest,
        "validation_manifest": config.validation_manifest,
    }
    missing = [name for name, path in required_paths.items() if path is None]
    if missing:
        raise ValueError(f"Training configuration is missing paths: {missing}.")
    for name, path in required_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name} does not exist: {path}")
    if config.epochs <= 0 or config.batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive.")
    if config.num_workers < 0:
        raise ValueError("num_workers must be non-negative.")
    if config.learning_rate <= 0 or config.weight_decay < 0:
        raise ValueError("learning_rate must be positive and weight_decay non-negative.")
    if config.gradient_clip_norm <= 0:
        raise ValueError("gradient_clip_norm must be positive.")
    if config.log_interval <= 0:
        raise ValueError("log_interval must be positive.")
    if config.scheduler not in {"none", "step"}:
        raise ValueError("scheduler must be 'none' or 'step'.")
    if config.pos_weight not in {None, "train_split"}:
        raise ValueError("pos_weight must be null or 'train_split'.")


def set_reproducible_seed(seed: int, deterministic: bool) -> None:
    """Seed Python, NumPy, torch, CUDA, and deterministic kernels."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic


def _resolve_device(requested: str) -> torch.device:
    normalized = requested.strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(normalized)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def _git_commit(project_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "git_unavailable"


def _environment_snapshot(device: torch.device, amp_enabled: bool) -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else platform.processor() or "cpu"
        ),
        "amp_enabled": amp_enabled,
        "provider_inference_executed": False,
        "video_decoded": False,
        "formal_training_executed": False,
    }


def _move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    output = dict(batch)
    for name in (
        "features",
        "feature_mask",
        "missing_mask",
        "observability",
        "reliability",
        "labels",
        "label_mask",
    ):
        output[name] = batch[name].to(device)
    return output


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), sort_keys=True, allow_nan=False) + "\n")


def _build_components(
    config: TrainingConfig,
    contract: FeatureContract,
    device: torch.device,
    train_dataset: Any,
) -> tuple[
    MinimalMissingAwareEvidenceHead,
    torch.optim.Optimizer,
    Any,
    Any,
    MaskedBinaryLoss,
]:
    model = MinimalMissingAwareEvidenceHead(
        branch_count=contract.branch_count,
        feature_dim=contract.feature_dim,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = (
        None
        if config.scheduler == "none"
        else torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.scheduler_step_size,
            gamma=config.scheduler_gamma,
        )
    )
    amp_enabled = config.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    pos_weight_value = None
    if config.pos_weight == "train_split":
        labels = [
            int(item["label"])
            for item in train_dataset
            if item["label"] is not None
        ]
        positives = sum(label == 1 for label in labels)
        negatives = sum(label == 0 for label in labels)
        if positives == 0 or negatives == 0:
            raise ValueError(
                "train_split pos_weight requires both known positive and negative labels."
            )
        pos_weight_value = negatives / positives
    criterion = MaskedBinaryLoss(pos_weight_value).to(device)
    return model, optimizer, scheduler, scaler, criterion


def train_one_epoch(
    *,
    model: MinimalMissingAwareEvidenceHead,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: MaskedBinaryLoss,
    scaler: Any,
    device: torch.device,
    amp_enabled: bool,
    gradient_clip_norm: float,
    global_step: int,
    max_batches: int | None = None,
    log_interval: int = 1,
) -> tuple[dict[str, Any], int]:
    """Train one epoch; empty-supervision batches never update parameters."""

    model.train()
    loss_sum = 0.0
    supervised_count = 0
    missing_label_count = 0
    no_evidence_count = 0
    skipped_batches = 0
    processed_batches = 0
    maximum_gradient_norm = 0.0
    logged_optimizer_step_count = 0
    for batch_index, raw_batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        processed_batches += 1
        batch = _move_batch(raw_batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            output = model(
                features=batch["features"],
                feature_mask=batch["feature_mask"],
                observability=batch["observability"],
                reliability=batch["reliability"],
            )
            loss_result = criterion(
                logits=output["logits"],
                labels=batch["labels"],
                label_mask=batch["label_mask"],
                valid_sample_mask=output["valid_sample_mask"],
            )
        missing_label_count += loss_result.missing_label_count
        no_evidence_count += loss_result.no_evidence_count
        if loss_result.loss is None:
            skipped_batches += 1
            continue
        scaler.scale(loss_result.loss).backward()
        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), gradient_clip_norm
        )
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError("Gradient norm is NaN or Inf.")
        maximum_gradient_norm = max(maximum_gradient_norm, float(gradient_norm))
        scaler.step(optimizer)
        scaler.update()
        global_step += 1
        if global_step % log_interval == 0:
            logged_optimizer_step_count += 1
        loss_sum += float(loss_result.loss.detach()) * loss_result.supervised_count
        supervised_count += loss_result.supervised_count
    return (
        {
            "loss": loss_sum / supervised_count if supervised_count else None,
            "supervised_sample_count": supervised_count,
            "missing_label_count": missing_label_count,
            "no_evidence_count": no_evidence_count,
            "skipped_batch_count": skipped_batches,
            "processed_batch_count": processed_batches,
            "max_preclip_gradient_norm": maximum_gradient_norm,
            "logged_optimizer_step_count": logged_optimizer_step_count,
            "log_interval": log_interval,
        },
        global_step,
    )


def validate_one_epoch(
    *,
    model: MinimalMissingAwareEvidenceHead,
    loader: torch.utils.data.DataLoader,
    criterion: MaskedBinaryLoss,
    device: torch.device,
    amp_enabled: bool,
    max_batches: int | None = None,
) -> dict[str, Any]:
    """Evaluate validation only; no state or statistic is updated."""

    model.eval()
    logits: list[float] = []
    labels: list[int] = []
    loss_sum = 0.0
    valid_sample_count = 0
    missing_label_count = 0
    no_evidence_count = 0
    skipped_batches = 0
    with torch.inference_mode():
        for batch_index, raw_batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            batch = _move_batch(raw_batch, device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                output = model(
                    features=batch["features"],
                    feature_mask=batch["feature_mask"],
                    observability=batch["observability"],
                    reliability=batch["reliability"],
                )
                loss_result = criterion(
                    logits=output["logits"],
                    labels=batch["labels"],
                    label_mask=batch["label_mask"],
                    valid_sample_mask=output["valid_sample_mask"],
                )
            valid_sample_count += int(output["valid_sample_mask"].sum().item())
            missing_label_count += loss_result.missing_label_count
            no_evidence_count += loss_result.no_evidence_count
            if loss_result.loss is None:
                skipped_batches += 1
                continue
            loss_sum += float(loss_result.loss) * loss_result.supervised_count
            selected = loss_result.supervised_mask
            logits.extend(output["logits"][selected].float().cpu().tolist())
            labels.extend(batch["labels"][selected].long().cpu().tolist())
    return binary_validation_metrics(
        logits=logits,
        labels=labels,
        loss_sum=loss_sum,
        valid_sample_count=valid_sample_count,
        missing_label_count=missing_label_count,
        no_evidence_count=no_evidence_count,
        skipped_batch_count=skipped_batches,
    )


def run_training(config: TrainingConfig) -> dict[str, Any]:
    """Run the bounded A2 train/validation engineering loop."""

    _validate_config(config)
    device = _resolve_device(config.device)
    if config.amp and device.type != "cuda":
        raise ValueError("A2 AMP is CUDA-only; disable amp for CPU runs.")
    set_reproducible_seed(config.seed, config.deterministic)
    contract = load_feature_contract(config.feature_contract)
    data: TrainingDataBundle = build_training_dataloaders(
        train_formal_manifest=config.train_formal_manifest,
        train_manifest=config.train_manifest,
        validation_formal_manifest=config.validation_formal_manifest,
        validation_manifest=config.validation_manifest,
        feature_contract=contract,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        seed=config.seed,
    )
    model, optimizer, scheduler, scaler, criterion = _build_components(
        config, contract, device, data.train_dataset
    )
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    if config.resume_checkpoint is None:
        metrics_path.write_text("", encoding="utf-8")
    snapshot = config.snapshot()
    (output_dir / "config_snapshot.yaml").write_text(
        yaml.safe_dump(snapshot, sort_keys=True),
        encoding="utf-8",
    )
    amp_enabled = config.amp and device.type == "cuda"
    (output_dir / "environment_snapshot.json").write_text(
        json.dumps(
            _environment_snapshot(device, amp_enabled),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    start_epoch = 1
    global_step = 0
    best_validation_metric: float | None = None
    if config.resume_checkpoint is not None:
        checkpoint = restore_training_checkpoint(
            config.resume_checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            expected_feature_contract=contract.descriptor(),
            expected_model_config=model.config_dict(),
            expected_train_manifest_checksum=data.train_dataset.manifest_checksum,
            expected_validation_manifest_checksum=(
                data.validation_dataset.manifest_checksum
            ),
            map_location=device,
        )
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])
        best_validation_metric = checkpoint["best_validation_metric"]

    git_commit = _git_commit(config.project_root)
    completed_epochs: list[int] = []
    last_validation: dict[str, Any] | None = None
    for epoch in range(start_epoch, config.epochs + 1):
        train_metrics, global_step = train_one_epoch(
            model=model,
            loader=data.train_loader,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            device=device,
            amp_enabled=amp_enabled,
            gradient_clip_norm=config.gradient_clip_norm,
            global_step=global_step,
            max_batches=config.max_train_batches,
            log_interval=config.log_interval,
        )
        validation_metrics = validate_one_epoch(
            model=model,
            loader=data.validation_loader,
            criterion=criterion,
            device=device,
            amp_enabled=amp_enabled,
            max_batches=config.max_validation_batches,
        )
        if scheduler is not None:
            scheduler.step()
        train_row = {
            "record_type": "train",
            "epoch": epoch,
            "global_step": global_step,
            **train_metrics,
        }
        validation_row = {
            "record_type": "validation",
            "epoch": epoch,
            "global_step": global_step,
            **validation_metrics,
        }
        _append_jsonl(metrics_path, train_row)
        _append_jsonl(metrics_path, validation_row)
        current_metric = validation_metrics["loss"]
        improved = current_metric is not None and (
            best_validation_metric is None or current_metric < best_validation_metric
        )
        if improved:
            best_validation_metric = float(current_metric)
        checkpoint_arguments = {
            "model": model,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "scaler": scaler,
            "epoch": epoch,
            "global_step": global_step,
            "best_validation_metric": best_validation_metric,
            "configuration": snapshot,
            "feature_contract": contract.descriptor(),
            "model_config": model.config_dict(),
            "git_commit": git_commit,
            "train_manifest_checksum": data.train_dataset.manifest_checksum,
            "validation_manifest_checksum": (
                data.validation_dataset.manifest_checksum
            ),
        }
        save_training_checkpoint(
            output_dir / "last_checkpoint.pt",
            **checkpoint_arguments,
        )
        if improved:
            save_training_checkpoint(
                output_dir / "best_checkpoint.pt",
                **checkpoint_arguments,
            )
        completed_epochs.append(epoch)
        last_validation = validation_metrics
    return {
        "completed_epochs": completed_epochs,
        "global_step": global_step,
        "best_validation_metric": best_validation_metric,
        "last_validation": last_validation,
        "output_dir": str(output_dir),
        "device": str(device),
        "amp_enabled": amp_enabled,
        "train_split": "train",
        "validation_split": "validation",
        "test_split_loaded": False,
        "synthetic_metrics_are_performance_claims": False,
        "formal_training_executed": False,
    }
