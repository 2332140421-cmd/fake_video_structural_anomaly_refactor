"""Strict checkpoint and resume support for the A2 engineering loop."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .contracts import TRAINING_CHECKPOINT_SCHEMA_VERSION


def capture_random_states() -> dict[str, Any]:
    """Capture Python, NumPy, and PyTorch RNG states."""

    return {
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_cpu_random_state": torch.get_rng_state(),
        "torch_cuda_random_state": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
    }


def restore_random_states(states: Mapping[str, Any]) -> None:
    """Restore all saved RNG states available on the current host."""

    random.setstate(states["python_random_state"])
    np.random.set_state(states["numpy_random_state"])
    torch.set_rng_state(states["torch_cpu_random_state"].cpu())
    cuda_states = states.get("torch_cuda_random_state", [])
    if cuda_states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([state.cpu() for state in cuda_states])


def save_training_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    epoch: int,
    global_step: int,
    best_validation_metric: float | None,
    configuration: Mapping[str, Any],
    feature_contract: Mapping[str, Any],
    model_config: Mapping[str, Any],
    git_commit: str,
    train_manifest_checksum: str,
    validation_manifest_checksum: str,
) -> Path:
    """Atomically save all state needed for a strict A2 resume."""

    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": TRAINING_CHECKPOINT_SCHEMA_VERSION,
        "feature_contract_version": feature_contract["feature_contract_version"],
        "feature_contract": dict(feature_contract),
        "model_config": dict(model_config),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": None if scheduler is None else scheduler.state_dict(),
        "amp_scaler_state": None if scaler is None else scaler.state_dict(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_validation_metric": best_validation_metric,
        "configuration": dict(configuration),
        "random_states": capture_random_states(),
        "git_commit": git_commit,
        "train_manifest_checksum": train_manifest_checksum,
        "validation_manifest_checksum": validation_manifest_checksum,
        "splits": {"train": "train", "validation": "validation"},
    }
    temporary = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(checkpoint_path)
    return checkpoint_path


def restore_training_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    expected_feature_contract: Mapping[str, Any],
    expected_model_config: Mapping[str, Any],
    expected_train_manifest_checksum: str,
    expected_validation_manifest_checksum: str,
    map_location: torch.device | str,
) -> dict[str, Any]:
    """Validate identity fields before restoring any mutable training state."""

    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=False,
    )
    if checkpoint.get("schema_version") != TRAINING_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("Checkpoint schema version mismatch.")
    if (
        checkpoint.get("feature_contract_version")
        != expected_feature_contract["feature_contract_version"]
    ):
        raise ValueError("Checkpoint feature contract version mismatch.")
    if checkpoint.get("feature_contract") != dict(expected_feature_contract):
        raise ValueError("Checkpoint feature contract mismatch.")
    if checkpoint.get("model_config") != dict(expected_model_config):
        raise ValueError("Checkpoint model config mismatch.")
    if checkpoint.get("train_manifest_checksum") != expected_train_manifest_checksum:
        raise ValueError("Checkpoint train manifest checksum mismatch.")
    if (
        checkpoint.get("validation_manifest_checksum")
        != expected_validation_manifest_checksum
    ):
        raise ValueError("Checkpoint validation manifest checksum mismatch.")
    if checkpoint.get("splits") != {"train": "train", "validation": "validation"}:
        raise ValueError("Checkpoint split identity mismatch.")
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    saved_scheduler = checkpoint.get("scheduler_state")
    if (saved_scheduler is None) != (scheduler is None):
        raise ValueError("Checkpoint scheduler configuration mismatch.")
    if scheduler is not None:
        scheduler.load_state_dict(saved_scheduler)
    saved_scaler = checkpoint.get("amp_scaler_state")
    if scaler is not None and saved_scaler is not None:
        scaler.load_state_dict(saved_scaler)
    restore_random_states(checkpoint["random_states"])
    return checkpoint
