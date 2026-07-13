#!/usr/bin/env python3
"""Run R_depth_cons tolerance sensitivity analysis."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _ensure_project_environment() -> None:
    """Re-execute with the project .venv Python when available."""

    project_python = PROJECT_ROOT / ".venv" / "bin" / "python"
    current_python = Path(sys.executable).resolve()
    if project_python.exists() and current_python != project_python.resolve():
        os.execv(str(project_python), [str(project_python), *sys.argv])


_ensure_project_environment()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Analyze R_depth_cons tolerance sensitivity.")
    parser.add_argument("--observation_dir", required=True)
    parser.add_argument("--depth_map_dir", required=True)
    parser.add_argument(
        "--output_dir",
        default=str(PROJECT_ROOT / "outputs" / "results" / "depth_consistency_sensitivity"),
    )
    parser.add_argument(
        "--visualization_dir",
        default=str(PROJECT_ROOT / "outputs" / "visualizations"),
    )
    parser.add_argument(
        "--tolerances",
        nargs="+",
        type=float,
        default=[0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.10],
    )
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument(
        "--depth_mode",
        choices=["auto", "no_depth", "real_depth_no_invert", "real_depth_invert"],
        default="real_depth_invert",
    )
    return parser.parse_args()


def run_pipeline_for_tolerance(
    args: argparse.Namespace,
    tolerance: float,
    work_dir: Path,
) -> Path:
    """Run the depth consistency pipeline for one tolerance and return pair CSV."""

    case_dir = work_dir / f"tol_{_format_float_for_path(tolerance)}"
    pair_csv = case_dir / "depth_consistency_pairs.csv"
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_depth_consistency_pipeline.py"),
        "--observation_dir",
        str(args.observation_dir),
        "--depth_map_dir",
        str(args.depth_map_dir),
        "--depth_mode",
        str(args.depth_mode),
        "--output_pair_csv",
        str(pair_csv),
        "--output_track_csv",
        str(case_dir / "depth_consistency_tracks.csv"),
        "--output_clip_csv",
        str(case_dir / "depth_consistency_clips.csv"),
        "--associated_observation_dir",
        str(case_dir / "observations_with_tracks"),
        "--raw_residual_visualization_path",
        str(case_dir / "raw_residual.png"),
        "--thresholded_residual_visualization_path",
        str(case_dir / "thresholded_residual.png"),
        "--combined_visualization_path",
        str(case_dir / "combined_analysis.png"),
        "--visualization_path",
        str(case_dir / "legacy_tracks.png"),
        "--tolerance",
        str(tolerance),
        "--topk",
        str(args.topk),
    ]
    if args.max_files is not None:
        command.extend(["--max_files", str(args.max_files)])
    subprocess.run(command, cwd=PROJECT_ROOT, check=True, capture_output=True, text=True)
    return pair_csv


def summarize_pair_csv(pair_csv: Path, tolerance: float, depth_mode: str) -> dict[str, Any]:
    """Summarize one pair CSV for sensitivity analysis."""

    rows = _read_valid_rows(pair_csv)
    raw = np.asarray([float(row["raw_residual"]) for row in rows], dtype=float)
    residual = np.asarray([float(row["residual"]) for row in rows], dtype=float)
    tracks = {str(row["video_id"]) + ":" + str(row["track_id"]) for row in rows}
    nonzero = residual > 0.0
    return {
        "tolerance": tolerance,
        "depth_mode": depth_mode,
        "num_tracks": len(tracks),
        "valid_transitions": len(rows),
        "nonzero_residual_count": int(np.sum(nonzero)) if residual.size else 0,
        "nonzero_residual_ratio": float(np.mean(nonzero)) if residual.size else 0.0,
        "mean_raw_residual": _stat(raw, "mean"),
        "max_raw_residual": _stat(raw, "max"),
        "mean_R_depth_cons": _stat(residual, "mean"),
        "max_R_depth_cons": _stat(residual, "max"),
        "p50_raw_residual": _percentile(raw, 50),
        "p90_raw_residual": _percentile(raw, 90),
        "p95_raw_residual": _percentile(raw, 95),
        "p99_raw_residual": _percentile(raw, 99),
    }


def build_threshold_recommendations(
    rows: list[dict[str, Any]],
    depth_mode: str,
) -> dict[str, Any]:
    """Build preliminary threshold recommendations from raw residual distribution."""

    zero_row = next((row for row in rows if float(row["tolerance"]) == 0.0), None)
    if zero_row is None:
        return {
            "depth_mode": depth_mode,
            "num_valid_transitions": 0,
            "note": "No tolerance=0 row was available.",
        }
    # Reconstruct enough from summary. p90/p95/p99 are exact from raw distribution.
    p90 = float(zero_row["p90_raw_residual"])
    p95 = float(zero_row["p95_raw_residual"])
    p99 = float(zero_row["p99_raw_residual"])
    mean = float(zero_row["mean_raw_residual"])
    # std needs raw rows; use approximate conservative fallback if unavailable.
    return {
        "depth_mode": depth_mode,
        "num_valid_transitions": int(zero_row["valid_transitions"]),
        "mean": mean,
        "std": float(zero_row.get("std_raw_residual", 0.0)),
        "p90": p90,
        "p95": p95,
        "p99": p99,
        "mean_plus_2std": float(zero_row.get("mean_plus_2std", mean)),
        "mean_plus_3std": float(zero_row.get("mean_plus_3std", mean)),
        "recommended_for_debug": p95,
        "note": (
            "This is a preliminary threshold from normal small-sample videos, "
            "not the final experimental threshold. Do not tune the final paper "
            "threshold on the test set."
        ),
    }


def enrich_with_raw_std(summary: dict[str, Any], pair_csv: Path) -> None:
    """Add raw residual std and mean+k*std fields to one summary row."""

    raw = np.asarray(
        [float(row["raw_residual"]) for row in _read_valid_rows(pair_csv)],
        dtype=float,
    )
    std = float(np.std(raw)) if raw.size else 0.0
    mean = float(summary["mean_raw_residual"])
    summary["std_raw_residual"] = std
    summary["mean_plus_2std"] = mean + 2.0 * std
    summary["mean_plus_3std"] = mean + 3.0 * std


def save_summary_csv(rows: list[dict[str, Any]], output_csv: Path) -> None:
    """Save sensitivity summary rows."""

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "tolerance",
        "depth_mode",
        "num_tracks",
        "valid_transitions",
        "nonzero_residual_count",
        "nonzero_residual_ratio",
        "mean_raw_residual",
        "max_raw_residual",
        "mean_R_depth_cons",
        "max_R_depth_cons",
        "p50_raw_residual",
        "p90_raw_residual",
        "p95_raw_residual",
        "p99_raw_residual",
        "std_raw_residual",
        "mean_plus_2std",
        "mean_plus_3std",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def save_sensitivity_plot(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Save tolerance sensitivity curves."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tolerances = [float(row["tolerance"]) for row in rows]
    nonzero_ratios = [float(row["nonzero_residual_ratio"]) for row in rows]
    means = [float(row["mean_R_depth_cons"]) for row in rows]
    max_values = [float(row["max_R_depth_cons"]) for row in rows]

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    axes[0].plot(tolerances, nonzero_ratios, marker="o")
    axes[0].set_ylabel("nonzero ratio")
    axes[1].plot(tolerances, means, marker="o")
    axes[1].set_ylabel("mean R_depth_cons")
    axes[2].plot(tolerances, max_values, marker="o")
    axes[2].set_ylabel("max R_depth_cons")
    axes[2].set_xlabel("tolerance")
    for axis in axes:
        axis.grid(True, alpha=0.25)
    fig.suptitle("R_depth_cons Tolerance Sensitivity")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _read_valid_rows(pair_csv: Path) -> list[dict[str, str]]:
    if not pair_csv.exists():
        return []
    with pair_csv.open("r", encoding="utf-8", newline="") as file:
        return [row for row in csv.DictReader(file) if row.get("valid") == "True"]


def _stat(values: np.ndarray, name: str) -> float:
    if values.size == 0:
        return 0.0
    if name == "mean":
        return float(np.mean(values))
    if name == "max":
        return float(np.max(values))
    raise ValueError(name)


def _percentile(values: np.ndarray, percentile: float) -> float:
    return float(np.percentile(values, percentile)) if values.size else 0.0


def _format_float_for_path(value: float) -> str:
    return str(value).replace(".", "_")


def main() -> None:
    """Run tolerance sensitivity analysis."""

    args = parse_args()
    output_dir = Path(args.output_dir)
    work_dir = output_dir / "runs"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for tolerance in args.tolerances:
        pair_csv = run_pipeline_for_tolerance(args, tolerance, work_dir)
        row = summarize_pair_csv(pair_csv, tolerance, args.depth_mode)
        enrich_with_raw_std(row, pair_csv)
        summary_rows.append(row)

    summary_csv = output_dir / "depth_consistency_sensitivity.csv"
    plot_path = output_dir / "depth_consistency_tolerance_sensitivity.png"
    visualization_plot_path = (
        Path(args.visualization_dir) / "depth_consistency_tolerance_sensitivity.png"
    )
    recommendation_path = output_dir / "depth_consistency_threshold_recommendations.json"
    root_recommendation_path = (
        PROJECT_ROOT / "outputs" / "results" / "depth_consistency_threshold_recommendations.json"
    )
    save_summary_csv(summary_rows, summary_csv)
    save_sensitivity_plot(summary_rows, plot_path)
    save_sensitivity_plot(summary_rows, visualization_plot_path)
    recommendations = build_threshold_recommendations(summary_rows, args.depth_mode)
    if summary_rows:
        zero_row = next((row for row in summary_rows if float(row["tolerance"]) == 0.0), summary_rows[0])
        recommendations["std"] = float(zero_row.get("std_raw_residual", 0.0))
        recommendations["mean_plus_2std"] = float(zero_row.get("mean_plus_2std", recommendations.get("mean", 0.0)))
        recommendations["mean_plus_3std"] = float(zero_row.get("mean_plus_3std", recommendations.get("mean", 0.0)))
    with recommendation_path.open("w", encoding="utf-8") as file:
        json.dump(recommendations, file, ensure_ascii=False, indent=2)
        file.write("\n")
    root_recommendation_path.parent.mkdir(parents=True, exist_ok=True)
    with root_recommendation_path.open("w", encoding="utf-8") as file:
        json.dump(recommendations, file, ensure_ascii=False, indent=2)
        file.write("\n")

    print(f"Saved sensitivity CSV: {summary_csv}")
    print(f"Saved sensitivity plot: {plot_path}")
    print(f"Saved visualization plot: {visualization_plot_path}")
    print(f"Saved threshold recommendations: {recommendation_path}")
    print(f"Saved root threshold recommendations: {root_recommendation_path}")
    for row in summary_rows:
        print(
            f"tolerance={row['tolerance']}: nonzero={row['nonzero_residual_count']}/"
            f"{row['valid_transitions']}, mean_R={row['mean_R_depth_cons']:.6f}, "
            f"max_R={row['max_R_depth_cons']:.6f}"
        )


if __name__ == "__main__":
    main()
