"""Deterministic binary metrics and raw curve points without sklearn."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def sigmoid(logits: Sequence[float]) -> np.ndarray:
    values = np.asarray(logits, dtype=float)
    return np.where(
        values >= 0.0,
        1.0 / (1.0 + np.exp(-values)),
        np.exp(values) / (1.0 + np.exp(values)),
    )


def _counts(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> tuple[int, int, int, int]:
    predicted = probabilities >= threshold
    positive = labels == 1
    tp = int(np.count_nonzero(predicted & positive))
    tn = int(np.count_nonzero(~predicted & ~positive))
    fp = int(np.count_nonzero(predicted & ~positive))
    fn = int(np.count_nonzero(~predicted & positive))
    return tp, tn, fp, fn


def roc_curve_points(
    labels: Sequence[int],
    probabilities: Sequence[float],
) -> list[dict[str, float]]:
    truth = np.asarray(labels, dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    thresholds = [float("inf"), *sorted(set(scores.tolist()), reverse=True), float("-inf")]
    points = []
    positives = max(int(np.count_nonzero(truth == 1)), 1)
    negatives = max(int(np.count_nonzero(truth == 0)), 1)
    for threshold in thresholds:
        tp, tn, fp, fn = _counts(truth, scores, threshold)
        points.append(
            {
                "threshold": threshold,
                "false_positive_rate": fp / negatives,
                "true_positive_rate": tp / positives,
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
            }
        )
    return points


def pr_curve_points(
    labels: Sequence[int],
    probabilities: Sequence[float],
) -> list[dict[str, float]]:
    truth = np.asarray(labels, dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    thresholds = [float("inf"), *sorted(set(scores.tolist()), reverse=True), float("-inf")]
    points = []
    for threshold in thresholds:
        tp, tn, fp, fn = _counts(truth, scores, threshold)
        predicted_positive = tp + fp
        points.append(
            {
                "threshold": threshold,
                "recall": tp / max(tp + fn, 1),
                "precision": tp / predicted_positive if predicted_positive else 1.0,
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
            }
        )
    return points


def threshold_curve_points(
    labels: Sequence[int],
    probabilities: Sequence[float],
) -> list[dict[str, float]]:
    truth = np.asarray(labels, dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    thresholds = sorted({0.0, 0.5, 1.0, *scores.tolist()})
    output = []
    for threshold in thresholds:
        tp, tn, fp, fn = _counts(truth, scores, threshold)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        output.append(
            {
                "threshold": float(threshold),
                "accuracy": (tp + tn) / max(len(truth), 1),
                "precision": precision,
                "recall": recall,
                "specificity": tn / max(tn + fp, 1),
                "f1": 2.0 * precision * recall / max(precision + recall, 1e-12),
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
            }
        )
    return output


def binary_classification_metrics(
    labels: Sequence[int],
    logits: Sequence[float],
    *,
    classification_threshold: float = 0.5,
) -> dict[str, Any]:
    truth = np.asarray(labels, dtype=int)
    raw = np.asarray(logits, dtype=float)
    if truth.ndim != 1 or raw.shape != truth.shape or not len(truth):
        raise ValueError("labels and logits must be aligned non-empty vectors.")
    if not np.isfinite(raw).all():
        raise ValueError("Logits must be finite.")
    if not 0.0 <= classification_threshold <= 1.0:
        raise ValueError("classification_threshold must be in [0,1].")
    probabilities = sigmoid(raw)
    tp, tn, fp, fn = _counts(truth, probabilities, classification_threshold)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    has_both_classes = bool(np.any(truth == 0) and np.any(truth == 1))
    unavailable_reason: dict[str, str] = {}
    roc_auc: float | None
    pr_auc: float | None
    if has_both_classes:
        roc = roc_curve_points(truth, probabilities)
        roc_auc = float(
            np.trapezoid(
                [row["true_positive_rate"] for row in roc],
                [row["false_positive_rate"] for row in roc],
            )
        )
        pr = pr_curve_points(truth, probabilities)
        recall_values = np.asarray([row["recall"] for row in pr], dtype=float)
        precision_values = np.asarray([row["precision"] for row in pr], dtype=float)
        order = np.argsort(recall_values, kind="stable")
        pr_auc = float(
            np.trapezoid(precision_values[order], recall_values[order])
        )
    else:
        roc_auc = None
        pr_auc = None
        unavailable_reason = {
            "roc_auc": "split_contains_only_one_class",
            "pr_auc": "split_contains_only_one_class",
        }
    return {
        "classification_threshold": float(classification_threshold),
        "accuracy": (tp + tn) / len(truth),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": 2.0 * precision * recall / max(precision + recall, 1e-12),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "unavailable_reason": unavailable_reason,
    }


__all__ = [
    "binary_classification_metrics",
    "pr_curve_points",
    "roc_curve_points",
    "sigmoid",
    "threshold_curve_points",
]
