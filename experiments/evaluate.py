"""Independent evaluation of a frozen residual temporal-head checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import torch

from data.residual_dataset import build_manifest_samples
from experiments.artifacts import prediction_rows
from experiments.metrics import (
    binary_classification_metrics,
    pr_curve_points,
    roc_curve_points,
    sigmoid,
    threshold_curve_points,
)
from experiments.train import binary_metrics, evaluate_residual_head
from models.temporal_head import ResidualTemporalHead
from utils.io import ensure_output_dir, write_csv, write_json


def evaluate_video_scores(
    labels: Sequence[int],
    logits: Sequence[float],
    *,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        **binary_metrics(labels, logits),
        "scope": "engineering_smoke_not_paper_performance",
    }
    if output_path is not None:
        write_json(output_path, metrics)
    return metrics


def interval_iou(
    predicted: Sequence[tuple[int, int]],
    target: Sequence[tuple[int, int]],
) -> float:
    predicted_frames = {
        frame for start, end in predicted for frame in range(start, end + 1)
    }
    target_frames = {
        frame for start, end in target for frame in range(start, end + 1)
    }
    union = predicted_frames | target_frames
    return len(predicted_frames & target_frames) / len(union) if union else 1.0


def evaluate_checkpoint(
    *,
    manifest_path: str | Path,
    checkpoint_path: str | Path,
    split: str,
    output_dir: str | Path,
    batch_size: int = 2,
    device: str = "cuda",
    num_workers: int = 0,
    classification_threshold: float | None = None,
) -> dict[str, Any]:
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"Unsupported evaluation split: {split!r}.")
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but CUDA is unavailable.")
    checkpoint = torch.load(
        Path(checkpoint_path),
        map_location=torch_device,
        weights_only=False,
    )
    required = {
        "model_state",
        "model_config",
        "channel_schema",
        "source_commit",
        "source_config_sha256",
        "manifest_sha256",
        "train_sample_ids",
        "validation_sample_ids",
        "classification_threshold",
    }
    missing = sorted(required - set(checkpoint))
    if missing:
        raise ValueError(f"Checkpoint is missing evaluation provenance: {missing}.")
    bundle = build_manifest_samples(manifest_path, checkpoint["channel_schema"])
    if bundle.manifest_sha256 != checkpoint["manifest_sha256"]:
        raise ValueError("Evaluation manifest identity differs from checkpoint.")
    if bundle.source_commit != checkpoint["source_commit"]:
        raise ValueError("Evaluation source commit differs from checkpoint.")
    if bundle.source_config_sha256 != checkpoint["source_config_sha256"]:
        raise ValueError("Evaluation producer config differs from checkpoint.")
    expected_ids = (
        checkpoint["train_sample_ids"]
        if split == "train"
        else checkpoint["validation_sample_ids"]
        if split == "validation"
        else bundle.sample_ids["test"]
    )
    if list(bundle.sample_ids[split]) != list(expected_ids):
        raise ValueError(f"Evaluation {split} sample order differs from checkpoint.")
    if not bundle.samples[split]:
        raise ValueError(f"Evaluation split {split!r} is empty.")
    model_config = checkpoint["model_config"]
    if int(model_config["expected_input_channels"]) != 12:
        raise ValueError("Checkpoint model input channel count is not 12.")
    model = ResidualTemporalHead(
        int(model_config["residual_count"]),
        hidden_size=int(model_config["hidden_size"]),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(torch_device)
    threshold = (
        float(checkpoint["classification_threshold"])
        if classification_threshold is None
        else float(classification_threshold)
    )
    result = evaluate_residual_head(
        model,
        bundle.samples[split],
        batch_size=batch_size,
        device=device,
        amp=bool(
            checkpoint.get("training_config", {}).get("amp", False)
            and torch_device.type == "cuda"
        ),
        num_workers=num_workers,
        classification_threshold=threshold,
    )
    probabilities = sigmoid(result["logits"]).tolist()
    predictions = prediction_rows(
        result["sample_ids"],
        result["labels"],
        result["logits"],
        probabilities,
        threshold=threshold,
    )
    metrics = binary_classification_metrics(
        result["labels"],
        result["logits"],
        classification_threshold=threshold,
    )
    metrics["loss"] = result["loss"]
    metrics["scope"] = "engineering_smoke_not_paper_performance"
    output = ensure_output_dir(output_dir)
    write_json(output / "metrics.json", metrics)
    write_csv(
        output / "predictions.csv",
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
    write_csv(
        output / "confusion_matrix.csv",
        [
            {"actual": 0, "predicted_0": metrics["tn"], "predicted_1": metrics["fp"]},
            {"actual": 1, "predicted_0": metrics["fn"], "predicted_1": metrics["tp"]},
        ],
        fieldnames=["actual", "predicted_0", "predicted_1"],
    )
    labels = result["labels"]
    write_csv(
        output / "roc_curve.csv",
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
        output / "pr_curve.csv",
        pr_curve_points(labels, probabilities),
        fieldnames=["threshold", "recall", "precision", "tp", "tn", "fp", "fn"],
    )
    write_csv(
        output / "threshold_curve.csv",
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
    summary = {
        "split": split,
        "checkpoint_path": str(Path(checkpoint_path).resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "manifest_path": bundle.manifest_path,
        "manifest_sha256": bundle.manifest_sha256,
        "channel_schema": bundle.channel_schema,
        "sample_ids": result["sample_ids"],
        "classification_threshold": threshold,
        "training_performed": False,
        "residual_recomputed": False,
        "metrics": metrics,
    }
    write_json(output / "evaluation_summary.json", summary)
    print(f"[EVALUATE] split={split} samples={len(predictions)}")
    for name in (
        "loss",
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "roc_auc",
        "pr_auc",
    ):
        value = metrics[name]
        print(f"{name}={'unavailable' if value is None else f'{float(value):.6f}'}")
    print(f"predictions={output / 'predictions.csv'}", flush=True)
    return {"metrics": metrics, "predictions": predictions, "summary": summary}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", required=True, choices=("train", "validation", "test"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--classification-threshold", type=float)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    evaluate_checkpoint(
        manifest_path=arguments.manifest,
        checkpoint_path=arguments.checkpoint,
        split=arguments.split,
        output_dir=arguments.output,
        batch_size=arguments.batch_size,
        device=arguments.device,
        num_workers=arguments.num_workers,
        classification_threshold=arguments.classification_threshold,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "evaluate_checkpoint",
    "evaluate_video_scores",
    "interval_iou",
]
