"""Reproducible, file-based artifacts for small paper-core experiments."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from data.residual_dataset import RESIDUAL_NAMES, TrainingManifestBundle
from experiments.metrics import (
    pr_curve_points,
    roc_curve_points,
    threshold_curve_points,
)
from utils.io import ensure_output_dir, write_csv, write_json


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(project_root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def write_git_state(path: str | Path, project_root: str | Path) -> Path:
    root = Path(project_root).resolve()
    tracked_diff = subprocess.run(
        ["git", "diff", "--binary"],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    untracked = [
        name
        for name in _git(root, "ls-files", "--others", "--exclude-standard").splitlines()
        if name
    ]
    return write_json(
        path,
        {
            "branch": _git(root, "branch", "--show-current"),
            "commit": _git(root, "rev-parse", "HEAD"),
            "status_short": _git(root, "status", "--short"),
            "tracked_diff_sha256": hashlib.sha256(tracked_diff).hexdigest(),
            "untracked_files": [
                {"path": name, "sha256": sha256_file(root / name)}
                for name in untracked
            ],
        },
    )


def initialize_run_artifacts(
    output_dir: str | Path,
    *,
    bundle: TrainingManifestBundle,
    run_config: Mapping[str, Any],
    project_root: str | Path,
) -> Path:
    output = ensure_output_dir(output_dir)
    for name in (
        "logs",
        "metrics",
        "predictions",
        "coverage",
        "efficiency",
        "checkpoints",
        "figures",
    ):
        ensure_output_dir(output / name)
    config_payload = dict(run_config)
    (output / "run_config.yaml").write_text(
        yaml.safe_dump(config_payload, sort_keys=False),
        encoding="utf-8",
    )
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(config_payload, sort_keys=False),
        encoding="utf-8",
    )
    write_json(
        output / "run_manifest.json",
        {
            "manifest_path": bundle.manifest_path,
            "manifest_sha256": bundle.manifest_sha256,
            "rows": bundle.manifest_rows,
        },
    )
    root = Path(project_root).resolve()
    write_git_state(output / "git_state.json", root)
    write_json(
        output / "environment.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "executable": sys.executable,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_runtime": torch.version.cuda,
            "cuda_device": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            "pid": os.getpid(),
        },
    )
    input_paths = sorted(
        {
            str(Path(row["residual_sequence_path"]).expanduser().resolve())
            for row in bundle.manifest_rows
        }
    )
    write_json(
        output / "input_artifacts.json",
        [
            {"path": path, "sha256": sha256_file(path), "copied": False}
            for path in input_paths
        ],
    )
    write_json(output / "residual_channel_schema.json", bundle.channel_schema)
    eligibility_rows = [
        report.as_dict() for report in bundle.eligibility_reports
    ]
    write_json(
        output / "eligibility_summary.json",
        bundle.eligibility_summary or {},
    )
    write_csv(
        output / "eligibility_report.csv",
        eligibility_rows,
        fieldnames=list(eligibility_rows[0]) if eligibility_rows else [
            "sample_id",
            "split",
            "label",
            "residual_path",
            "clip_count",
            "residual_record_count",
            "valid_count",
            "observed_count",
            "blocked_by_input_count",
            "not_applicable_count",
            "eligibility_status",
            "exclusion_reason",
            "model_eligible",
        ],
    )
    write_json(output / "leakage_audit.json", bundle.leakage_audit)
    write_csv(
        output / "split_snapshot.csv",
        bundle.manifest_rows,
        fieldnames=list(bundle.manifest_rows[0].keys()),
    )
    return output


def prediction_rows(
    sample_ids: Sequence[str],
    labels: Sequence[int],
    logits: Sequence[float],
    probabilities: Sequence[float],
    *,
    threshold: float,
) -> list[dict[str, Any]]:
    return [
        {
            "sample_id": sample_id,
            "label": int(label),
            "logit": float(logit),
            "probability": float(probability),
            "prediction": int(probability >= threshold),
            "classification_threshold": float(threshold),
        }
        for sample_id, label, logit, probability in zip(
            sample_ids, labels, logits, probabilities, strict=True
        )
    ]


def write_metric_artifacts(
    output_dir: str | Path,
    *,
    split: str,
    metrics: Mapping[str, Any],
    predictions: Sequence[Mapping[str, Any]],
    write_shared_curves: bool = False,
) -> None:
    output = Path(output_dir)
    write_json(output / "metrics" / f"{split}_metrics.json", metrics)
    write_csv(
        output / "predictions" / f"{split}_video_predictions.csv",
        predictions,
        fieldnames=[
            "sample_id",
            "label",
            "logit",
            "probability",
            "prediction",
            "classification_threshold",
        ],
    )
    if write_shared_curves:
        labels = [int(row["label"]) for row in predictions]
        probabilities = [float(row["probability"]) for row in predictions]
        write_csv(
            output / "metrics" / "confusion_matrix.csv",
            [
                {"actual": 0, "predicted_0": metrics["tn"], "predicted_1": metrics["fp"]},
                {"actual": 1, "predicted_0": metrics["fn"], "predicted_1": metrics["tp"]},
            ],
            fieldnames=["actual", "predicted_0", "predicted_1"],
        )
        write_csv(
            output / "metrics" / "roc_curve.csv",
            roc_curve_points(labels, probabilities),
            fieldnames=[
                "threshold",
                "false_positive_rate",
                "true_positive_rate",
                "tp",
                "tn",
                "fp",
                "fn",
            ],
        )
        write_csv(
            output / "metrics" / "pr_curve.csv",
            pr_curve_points(labels, probabilities),
            fieldnames=["threshold", "recall", "precision", "tp", "tn", "fp", "fn"],
        )
        write_csv(
            output / "metrics" / "threshold_curve.csv",
            threshold_curve_points(labels, probabilities),
            fieldnames=[
                "threshold",
                "accuracy",
                "precision",
                "recall",
                "specificity",
                "f1",
                "tp",
                "tn",
                "fp",
                "fn",
            ],
        )


def write_coverage_artifacts(
    output_dir: str | Path,
    bundle: TrainingManifestBundle,
) -> None:
    output = Path(output_dir)
    all_samples = [
        sample
        for split in ("train", "validation", "test")
        for sample in bundle.samples[split]
    ]
    channel_rows = []
    total_timesteps = sum(sample.residuals.shape[0] for sample in all_samples)
    for index, name in enumerate(RESIDUAL_NAMES):
        available = sum(
            int(np.count_nonzero(sample.availability[:, index]))
            for sample in all_samples
        )
        channel_rows.append(
            {
                "channel_index": index,
                "channel_name": name,
                "available_count": available,
                "total_count": total_timesteps,
                "coverage": available / max(total_timesteps, 1),
            }
        )
    write_csv(
        output / "coverage" / "residual_channel_coverage.csv",
        channel_rows,
        fieldnames=[
            "channel_index",
            "channel_name",
            "available_count",
            "total_count",
            "coverage",
        ],
    )
    sample_rows = []
    reason_counts: dict[tuple[str, str], int] = {}
    for split in ("train", "validation", "test"):
        for sample in bundle.samples[split]:
            sample_rows.append(
                {
                    "sample_id": sample.sample_id,
                    "split": split,
                    "timestep_count": sample.residuals.shape[0],
                    "available_count": int(np.count_nonzero(sample.availability)),
                    "total_count": int(sample.availability.size),
                    "availability_rate": float(np.mean(sample.availability)),
                }
            )
            payload = json.loads(
                Path(sample.residual_sequence_path).read_text(encoding="utf-8")
            )
            for clip in payload.get("clips", ()):
                for row in clip.get("residuals", ()):
                    if bool(row.get("valid_mask", False)):
                        continue
                    name = str(row.get("name", "unknown"))
                    reason = str(
                        row.get("unavailable_reason")
                        or row.get("availability")
                        or "unspecified"
                    )
                    reason_counts[(name, reason)] = (
                        reason_counts.get((name, reason), 0) + 1
                    )
    write_csv(
        output / "coverage" / "sample_availability.csv",
        sample_rows,
        fieldnames=[
            "sample_id",
            "split",
            "timestep_count",
            "available_count",
            "total_count",
            "availability_rate",
        ],
    )
    write_csv(
        output / "coverage" / "unavailable_reason_counts.csv",
        [
            {"channel_name": key[0], "unavailable_reason": key[1], "count": value}
            for key, value in sorted(reason_counts.items())
        ],
        fieldnames=["channel_name", "unavailable_reason", "count"],
    )


def _curve_figure(
    path: Path,
    x_values: Sequence[float],
    series: Mapping[str, Sequence[float]],
    *,
    xlabel: str,
    ylabel: str,
) -> None:
    figure, axis = plt.subplots(figsize=(5.5, 3.5))
    for label, values in series.items():
        axis.plot(x_values, values, marker="o", label=label)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.25)
    if len(series) > 1:
        axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def write_training_figures(
    output_dir: str | Path,
    epoch_history: Sequence[Mapping[str, Any]],
    validation_predictions: Sequence[Mapping[str, Any]],
) -> None:
    output = Path(output_dir)
    epochs = [int(row["epoch"]) for row in epoch_history]
    for metric, filename in (
        ("loss", "loss_curve.png"),
        ("accuracy", "accuracy_curve.png"),
        ("f1", "f1_curve.png"),
    ):
        _curve_figure(
            output / "figures" / filename,
            epochs,
            {
                "train": [float(row[f"train_{metric}"]) for row in epoch_history],
                "validation": [
                    float(row[f"validation_{metric}"]) for row in epoch_history
                ],
            },
            xlabel="epoch",
            ylabel=metric,
        )
    labels = [int(row["label"]) for row in validation_predictions]
    probabilities = [float(row["probability"]) for row in validation_predictions]
    roc = roc_curve_points(labels, probabilities)
    pr = pr_curve_points(labels, probabilities)
    _curve_figure(
        output / "figures" / "roc_curve.png",
        [float(row["false_positive_rate"]) for row in roc],
        {"validation": [float(row["true_positive_rate"]) for row in roc]},
        xlabel="false positive rate",
        ylabel="true positive rate",
    )
    _curve_figure(
        output / "figures" / "pr_curve.png",
        [float(row["recall"]) for row in pr],
        {"validation": [float(row["precision"]) for row in pr]},
        xlabel="recall",
        ylabel="precision",
    )
    coverage_path = output / "coverage" / "residual_channel_coverage.csv"
    rows = np.genfromtxt(coverage_path, delimiter=",", names=True, dtype=None, encoding=None)
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.bar(
        np.arange(len(rows)),
        [float(row["coverage"]) for row in rows],
    )
    axis.set_xticks(np.arange(len(rows)), RESIDUAL_NAMES, rotation=70, ha="right")
    axis.set_ylabel("coverage")
    figure.tight_layout()
    figure.savefig(output / "figures" / "residual_coverage.png", dpi=140)
    plt.close(figure)


__all__ = [
    "initialize_run_artifacts",
    "prediction_rows",
    "sha256_file",
    "write_git_state",
    "write_coverage_artifacts",
    "write_metric_artifacts",
    "write_training_figures",
]
