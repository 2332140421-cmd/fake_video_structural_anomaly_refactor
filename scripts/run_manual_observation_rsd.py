#!/usr/bin/env python3
"""Run scale-depth residual analysis on manually authored observation JSON."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


def _ensure_project_environment() -> Path:
    """Re-execute this script with the project .venv Python when available."""

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

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from semantic3d.observations import FrameObservationJSON  # noqa: E402
from semantic3d.scale_depth import (  # noqa: E402
    ObjectObservation,
    ScalePrior,
    scale_depth_residual,
    scale_depth_residual_log,
)


DEFAULT_SCALE_PRIORS: Mapping[str, ScalePrior] = {
    "soccer_ball": ScalePrior(min_size=0.20, max_size=0.24),
    "elephant": ScalePrior(min_size=2.40, max_size=3.40),
    "person": ScalePrior(min_size=1.50, max_size=1.90),
    "car": ScalePrior(min_size=1.40, max_size=1.80),
    "cup": ScalePrior(min_size=0.08, max_size=0.15),
}

CSV_FIELDS = [
    "case_id",
    "frame_id",
    "expected_label",
    "max_R_sd",
    "mean_R_sd",
    "topk_mean_R_sd",
    "max_R_sd_log",
    "mean_R_sd_log",
    "topk_mean_R_sd_log",
    "predicted_label",
    "is_correct",
    "pairwise_R_sd",
    "pairwise_R_sd_log",
    "source_json",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Compute R_sd on manually authored observation JSON files."
    )
    parser.add_argument(
        "--input_dir",
        default=str(PROJECT_ROOT / "data" / "manual_observations"),
        help="Directory containing manual observation JSON files.",
    )
    parser.add_argument(
        "--output_csv",
        default=str(
            PROJECT_ROOT / "outputs" / "results" / "manual_observation_rsd_results.csv"
        ),
        help="CSV path for batch residual results.",
    )
    parser.add_argument(
        "--output_png",
        default=str(
            PROJECT_ROOT
            / "outputs"
            / "visualizations"
            / "manual_observation_rsd_scores.png"
        ),
        help="PNG path for the score bar chart.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="Threshold on max_R_sd_log for predicted_label.",
    )
    parser.add_argument("--top_k", type=int, default=3, help="Top-k mean size.")
    return parser.parse_args()


def load_manual_observation(path: Path) -> tuple[str, int, FrameObservationJSON]:
    """Load one manual observation JSON and return metadata plus frame data."""

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    if "case_id" not in data:
        raise ValueError(f"Missing required field in manual observation: case_id.")
    if "expected_label" not in data:
        raise ValueError(f"Missing required field in manual observation: expected_label.")
    return str(data["case_id"]), int(data["expected_label"]), FrameObservationJSON.from_dict(data)


def topk_mean(values: Sequence[float], top_k: int = 3) -> float:
    """Return the mean of the largest k values."""

    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return 0.0
    k = min(max(int(top_k), 1), array.size)
    return float(np.sort(array)[-k:].mean())


def _object_pairs(
    objects: Sequence[ObjectObservation],
) -> list[tuple[ObjectObservation, ObjectObservation]]:
    """Return unique unordered object pairs for a frame."""

    return [
        (objects[i], objects[j])
        for i in range(len(objects))
        for j in range(i + 1, len(objects))
    ]


def evaluate_manual_observation(
    path: Path,
    scale_priors: Mapping[str, ScalePrior] = DEFAULT_SCALE_PRIORS,
    threshold: float = 0.1,
    top_k: int = 3,
) -> dict[str, Any]:
    """Evaluate one manual observation file and return a CSV-ready row."""

    case_id, expected_label, frame = load_manual_observation(path)
    objects = [obj.to_scale_depth_observation() for obj in frame.objects]
    if len(objects) < 2:
        raise ValueError(f"Manual observation '{case_id}' requires at least two objects.")

    ratio_values: list[float] = []
    log_values: list[float] = []
    ratio_pairs: dict[str, float] = {}
    log_pairs: dict[str, float] = {}

    for obj_a, obj_b in _object_pairs(objects):
        residual, _details = scale_depth_residual(obj_a, obj_b, scale_priors)
        residual_log, _details_log = scale_depth_residual_log(
            obj_a, obj_b, scale_priors
        )
        pair_name = f"{obj_a.object_id}->{obj_b.object_id}"
        ratio_values.append(float(residual))
        log_values.append(float(residual_log))
        ratio_pairs[pair_name] = float(residual)
        log_pairs[pair_name] = float(residual_log)

    max_log = max(log_values)
    predicted_label = int(max_log >= threshold)
    return {
        "case_id": case_id,
        "frame_id": frame.frame_id,
        "expected_label": expected_label,
        "max_R_sd": max(ratio_values),
        "mean_R_sd": float(np.mean(ratio_values)),
        "topk_mean_R_sd": topk_mean(ratio_values, top_k),
        "max_R_sd_log": max_log,
        "mean_R_sd_log": float(np.mean(log_values)),
        "topk_mean_R_sd_log": topk_mean(log_values, top_k),
        "predicted_label": predicted_label,
        "is_correct": int(predicted_label == expected_label),
        "pairwise_R_sd": json.dumps(ratio_pairs, sort_keys=True),
        "pairwise_R_sd_log": json.dumps(log_pairs, sort_keys=True),
        "source_json": str(path),
    }


def run_manual_batch(
    input_dir: Path,
    output_csv: Path,
    output_png: Path,
    threshold: float = 0.1,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """Evaluate all manual observation JSON files and save CSV plus PNG."""

    json_paths = sorted(input_dir.glob("*.json"))
    if not json_paths:
        raise FileNotFoundError(f"No manual observation JSON files found in {input_dir}.")

    rows = [
        evaluate_manual_observation(
            path, threshold=threshold, top_k=top_k
        )
        for path in json_paths
    ]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in CSV_FIELDS})

    save_score_plot(rows, output_png)
    return rows


def save_score_plot(rows: Sequence[Mapping[str, Any]], output_png: Path) -> None:
    """Save a bar chart of max_R_sd_log for manual observations."""

    output_png.parent.mkdir(parents=True, exist_ok=True)
    case_ids = [str(row["case_id"]) for row in rows]
    scores = [float(row["max_R_sd_log"]) for row in rows]
    labels = [int(row["expected_label"]) for row in rows]
    colors = ["#4C78A8" if label == 0 else "#E45756" for label in labels]
    hatches = ["" if label == 0 else "//" for label in labels]

    fig, ax = plt.subplots(figsize=(10, 4.8))
    bars = ax.bar(case_ids, scores, color=colors, edgecolor="black", linewidth=0.8)
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    ax.set_title("Manual Observation Scale-Depth Residual Scores")
    ax.set_xlabel("case_id")
    ax.set_ylabel("max_R_sd_log")
    ax.tick_params(axis="x", labelrotation=45)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_png, dpi=160)
    plt.close(fig)


def main() -> None:
    """Run manual observation R_sd analysis from the command line."""

    args = parse_args()
    rows = run_manual_batch(
        input_dir=Path(args.input_dir),
        output_csv=Path(args.output_csv),
        output_png=Path(args.output_png),
        threshold=args.threshold,
        top_k=args.top_k,
    )
    valid = len(rows)
    correct = sum(int(row["is_correct"]) for row in rows)
    print(f"Saved {valid} manual observation result row(s) to {args.output_csv}")
    print(f"Saved score visualization to {args.output_png}")
    print(f"Simple threshold accuracy: {correct}/{valid} = {correct / valid:.3f}")


if __name__ == "__main__":
    main()
