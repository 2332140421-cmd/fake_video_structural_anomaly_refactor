#!/usr/bin/env python
"""Batch synthetic experiment for object-level scale-depth residual R_sd.

This is not a semantic common-sense experiment and not a complete forged-video
detector. It validates one object-level projection-geometry residual in a 3D
structural constraint system:

    object class scale prior + image projection area + estimated depth

No YOLO, SAM, Depth Anything, RAFT, CoTracker, or any real vision model is
called by this script.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


def _ensure_project_environment() -> Path:
    """Re-run with the project-local Python environment if needed."""

    project_root = Path(__file__).resolve().parents[1]
    project_python = project_root / ".venv" / "bin" / "python"
    current_python = Path(sys.executable).resolve()

    if project_python.exists() and current_python != project_python.resolve():
        os.execv(str(project_python), [str(project_python), *sys.argv])

    return project_root


PROJECT_ROOT = _ensure_project_environment()
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from semantic3d.scale_depth import (  # noqa: E402
    ObjectObservation,
    ScalePrior,
    pairwise_scale_depth_residuals,
)


DEFAULT_SCALE_PRIORS: Mapping[str, ScalePrior] = {
    "soccer_ball": ScalePrior(min_size=0.20, max_size=0.24),
    "elephant": ScalePrior(min_size=2.40, max_size=3.40),
    "person": ScalePrior(min_size=1.50, max_size=1.90),
    "car": ScalePrior(min_size=1.40, max_size=1.80),
    "cup": ScalePrior(min_size=0.08, max_size=0.15),
    "dog": ScalePrior(min_size=0.40, max_size=0.80),
    "small_object": ScalePrior(min_size=0.10, max_size=0.25),
    "large_object": ScalePrior(min_size=2.00, max_size=3.00),
}


CSV_FIELDS = [
    "case_id",
    "expected_label",
    "status",
    "max_R_sd",
    "mean_R_sd",
    "topk_mean_R_sd",
    "max_R_sd_log",
    "mean_R_sd_log",
    "topk_mean_R_sd_log",
    "predicted_label",
    "is_correct",
    "error_message",
    "num_objects",
    "num_pairs",
    "pairwise_R_sd",
    "pairwise_R_sd_log",
    "source_json",
]


def load_observation_json(path: Path) -> Dict[str, Any]:
    """Load one synthetic observation JSON file."""

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def objects_from_sample(
    sample: Mapping[str, Any], min_confidence: float = 0.5
) -> List[ObjectObservation]:
    """Convert one JSON sample into object observations.

    Args:
        sample: Parsed JSON sample.
        min_confidence: Minimum acceptable observation confidence. Low confidence
            samples are treated as invalid synthetic inputs for this experiment.

    Returns:
        List of ObjectObservation instances.
    """

    frame_width = float(sample["frame_width"])
    frame_height = float(sample["frame_height"])
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError(
            f"frame_width and frame_height must be positive, got "
            f"{frame_width} x {frame_height}."
        )

    frame_area_default = frame_width * frame_height
    objects = []
    for item in sample["objects"]:
        confidence = float(item.get("confidence", 1.0))
        if confidence < min_confidence:
            raise ValueError(
                f"Object '{item.get('object_id')}' has confidence={confidence}; "
                f"confidence must be >= {min_confidence}."
            )
        objects.append(
            ObjectObservation(
                object_id=str(item["object_id"]),
                label=str(item["label"]),
                mask_area=float(item["mask_area"]),
                frame_area=float(item.get("frame_area", frame_area_default)),
                depth=float(item["depth"]),
                confidence=confidence,
            )
        )

    if len(objects) < 2:
        raise ValueError(f"case_id={sample.get('case_id')} requires at least 2 objects.")
    return objects


def off_diagonal_values(matrix: np.ndarray) -> np.ndarray:
    """Return all non-diagonal values from a square pairwise matrix."""

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Expected a square residual matrix, got shape {matrix.shape}.")
    return matrix[~np.eye(matrix.shape[0], dtype=bool)]


def topk_mean(values: Sequence[float], top_k: int) -> float:
    """Compute the mean of the largest k residual values."""

    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return 0.0
    k = min(max(int(top_k), 1), array.size)
    return float(np.sort(array)[-k:].mean())


def _pairwise_residual_dict(details: Mapping[tuple[str, str], Mapping[str, float]]) -> str:
    """Serialize pairwise residual details as a compact JSON string."""

    pairwise = {
        f"{object_a}->{object_b}": float(pair_details["residual"])
        for (object_a, object_b), pair_details in details.items()
    }
    return json.dumps(pairwise, sort_keys=True)


def evaluate_sample(
    sample_path: Path,
    scale_priors: Mapping[str, ScalePrior] = DEFAULT_SCALE_PRIORS,
    top_k: int = 3,
    threshold: float = 0.1,
    min_confidence: float = 0.5,
) -> Dict[str, Any]:
    """Evaluate one JSON case and return a CSV-ready result row.

    Invalid input is converted to a row with status="invalid" and a clear
    error_message, so one bad case does not interrupt the batch experiment.
    """

    try:
        sample = load_observation_json(sample_path)
        case_id = str(sample["case_id"])
        expected_label = int(sample["expected_label"])
        objects = objects_from_sample(sample, min_confidence=min_confidence)

        ratio_matrix, ratio_details = pairwise_scale_depth_residuals(
            objects, scale_priors, use_log=False
        )
        log_matrix, log_details = pairwise_scale_depth_residuals(
            objects, scale_priors, use_log=True
        )

        ratio_values = off_diagonal_values(ratio_matrix)
        log_values = off_diagonal_values(log_matrix)
        max_log = float(log_values.max())
        predicted_label = int(max_log >= threshold)
        is_correct = int(predicted_label == expected_label)

        return {
            "case_id": case_id,
            "expected_label": expected_label,
            "status": "ok",
            "max_R_sd": float(ratio_values.max()),
            "mean_R_sd": float(ratio_values.mean()),
            "topk_mean_R_sd": topk_mean(ratio_values, top_k),
            "max_R_sd_log": max_log,
            "mean_R_sd_log": float(log_values.mean()),
            "topk_mean_R_sd_log": topk_mean(log_values, top_k),
            "predicted_label": predicted_label,
            "is_correct": is_correct,
            "error_message": "",
            "num_objects": len(objects),
            "num_pairs": int(ratio_values.size),
            "pairwise_R_sd": _pairwise_residual_dict(ratio_details),
            "pairwise_R_sd_log": _pairwise_residual_dict(log_details),
            "source_json": str(sample_path.relative_to(PROJECT_ROOT)),
        }
    except Exception as exc:
        fallback_case_id = sample_path.stem
        fallback_label: str | int = ""
        try:
            sample = load_observation_json(sample_path)
            fallback_case_id = str(sample.get("case_id", fallback_case_id))
            fallback_label = sample.get("expected_label", "")
        except Exception:
            pass

        return {
            "case_id": fallback_case_id,
            "expected_label": fallback_label,
            "status": "invalid",
            "max_R_sd": "",
            "mean_R_sd": "",
            "topk_mean_R_sd": "",
            "max_R_sd_log": "",
            "mean_R_sd_log": "",
            "topk_mean_R_sd_log": "",
            "predicted_label": "",
            "is_correct": "",
            "error_message": f"{type(exc).__name__}: {exc}",
            "num_objects": 0,
            "num_pairs": 0,
            "pairwise_R_sd": "{}",
            "pairwise_R_sd_log": "{}",
            "source_json": str(sample_path.relative_to(PROJECT_ROOT)),
        }


def compute_classification_metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    """Compute accuracy, precision, recall, and F1 over valid rows."""

    valid_rows = [row for row in rows if row["status"] == "ok"]
    if not valid_rows:
        raise ValueError("No valid rows available for metric computation.")

    y_true = np.asarray([int(row["expected_label"]) for row in valid_rows], dtype=int)
    y_pred = np.asarray([int(row["predicted_label"]) for row in valid_rows], dtype=int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    accuracy = (tp + tn) / len(valid_rows)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "num_total": float(len(rows)),
        "num_valid": float(len(valid_rows)),
        "num_invalid": float(len(rows) - len(valid_rows)),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def run_experiment(
    input_dir: Path,
    output_csv: Path,
    output_plot: Path,
    output_metrics: Path,
    top_k: int = 3,
    threshold: float = 0.1,
    min_confidence: float = 0.5,
) -> List[Dict[str, Any]]:
    """Run the full synthetic R_sd experiment and write output artifacts."""

    sample_paths = sorted(input_dir.glob("*.json"))
    if not sample_paths:
        raise FileNotFoundError(f"No JSON samples found in {input_dir}.")

    rows = [
        evaluate_sample(
            path,
            top_k=top_k,
            threshold=threshold,
            min_confidence=min_confidence,
        )
        for path in sample_paths
    ]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_plot.parent.mkdir(parents=True, exist_ok=True)
    output_metrics.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows({field: row[field] for field in CSV_FIELDS} for row in rows)

    metrics = compute_classification_metrics(rows)
    metrics["threshold"] = float(threshold)
    metrics["min_confidence"] = float(min_confidence)
    with output_metrics.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, sort_keys=True)

    plot_scores(rows, output_plot, threshold)
    return rows


def plot_scores(rows: Iterable[Mapping[str, Any]], output_plot: Path, threshold: float) -> None:
    """Generate a bar chart of per-case max_R_sd_log scores."""

    valid_rows = [row for row in rows if row["status"] == "ok"]
    labels = [str(row["case_id"]) for row in valid_rows]
    scores = [float(row["max_R_sd_log"]) for row in valid_rows]

    fig, ax = plt.subplots(figsize=(15, 5.5))
    for index, row in enumerate(valid_rows):
        expected_label = int(row["expected_label"])
        color = "#1b9e77" if expected_label == 0 else "#d95f02"
        hatch = "" if expected_label == 0 else "///"
        ax.bar(
            labels[index],
            scores[index],
            color=color,
            edgecolor="black",
            linewidth=0.4,
            hatch=hatch,
        )

    ax.axhline(threshold, color="black", linestyle="--", linewidth=1.2, label="threshold")
    ax.set_ylabel("max_R_sd_log")
    ax.set_title("Scale-Depth Consistency Residual on Synthetic Cases")
    ax.tick_params(axis="x", labelrotation=65)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    handles = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            color="#1b9e77",
            ec="black",
            label="expected_label=0",
        ),
        plt.Rectangle(
            (0, 0),
            1,
            1,
            color="#d95f02",
            ec="black",
            hatch="///",
            label="expected_label=1",
        ),
    ]
    ax.legend(handles=handles + [ax.lines[0]])
    fig.tight_layout()
    fig.savefig(output_plot, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "synthetic_observations",
        help="Directory containing synthetic observation JSON files.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "results" / "scale_depth_results.csv",
        help="Path of the CSV result file.",
    )
    parser.add_argument(
        "--output-plot",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "visualizations" / "scale_depth_scores.png",
        help="Path of the PNG bar chart.",
    )
    parser.add_argument(
        "--output-metrics",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "results" / "scale_depth_metrics.json",
        help="Path of the JSON metric summary.",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Top-k used by top-k mean.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="Threshold on max_R_sd_log for predicted_label.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.5,
        help="Minimum confidence required for synthetic observations.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    rows = run_experiment(
        input_dir=args.input_dir,
        output_csv=args.output_csv,
        output_plot=args.output_plot,
        output_metrics=args.output_metrics,
        top_k=args.top_k,
        threshold=args.threshold,
        min_confidence=args.min_confidence,
    )
    metrics = compute_classification_metrics(rows)
    print(f"Processed {len(rows)} synthetic samples.")
    print(
        f"valid={int(metrics['num_valid'])}, invalid={int(metrics['num_invalid'])}, "
        f"accuracy={metrics['accuracy']:.3f}, precision={metrics['precision']:.3f}, "
        f"recall={metrics['recall']:.3f}, f1={metrics['f1']:.3f}"
    )
    print(f"CSV written to: {args.output_csv}")
    print(f"Metrics written to: {args.output_metrics}")
    print(f"Plot written to: {args.output_plot}")


if __name__ == "__main__":
    main()
