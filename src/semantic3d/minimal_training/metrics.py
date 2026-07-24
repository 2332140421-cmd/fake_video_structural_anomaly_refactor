"""Dependency-light binary validation metrics for engineering checks."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def _roc_auc(labels: np.ndarray, probabilities: np.ndarray) -> float | None:
    positive = probabilities[labels == 1]
    negative = probabilities[labels == 0]
    if positive.size == 0 or negative.size == 0:
        return None
    comparisons = positive[:, None] - negative[None, :]
    return float(
        (np.sum(comparisons > 0) + 0.5 * np.sum(comparisons == 0))
        / comparisons.size
    )


def binary_validation_metrics(
    *,
    logits: Sequence[float],
    labels: Sequence[int],
    loss_sum: float,
    valid_sample_count: int,
    missing_label_count: int,
    no_evidence_count: int,
    skipped_batch_count: int,
) -> dict[str, Any]:
    """Compute fixed-threshold metrics without threshold search."""

    logits_array = np.asarray(logits, dtype=np.float64)
    labels_array = np.asarray(labels, dtype=np.int64)
    if logits_array.shape != labels_array.shape:
        raise ValueError("Metric logits and labels must have identical shape.")
    count = int(labels_array.size)
    probabilities = 1.0 / (1.0 + np.exp(-logits_array))
    predictions = (probabilities >= 0.5).astype(np.int64)
    tp = int(np.sum((predictions == 1) & (labels_array == 1)))
    tn = int(np.sum((predictions == 0) & (labels_array == 0)))
    fp = int(np.sum((predictions == 1) & (labels_array == 0)))
    fn = int(np.sum((predictions == 0) & (labels_array == 1)))
    accuracy = (tp + tn) / count if count else None
    precision = tp / (tp + fp) if tp + fp else 0.0 if count else None
    recall = tp / (tp + fn) if tp + fn else 0.0 if count else None
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if count and precision is not None and recall is not None and precision + recall
        else 0.0 if count else None
    )
    return {
        "loss": loss_sum / count if count else None,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": _roc_auc(labels_array, probabilities) if count else None,
        "supervised_sample_count": count,
        "valid_sample_count": int(valid_sample_count),
        "missing_label_count": int(missing_label_count),
        "no_evidence_count": int(no_evidence_count),
        "skipped_batch_count": int(skipped_batch_count),
        "classification_threshold": 0.5,
        "threshold_role": "engineering_validation_only",
    }
