#!/usr/bin/env python3
"""Analyze exploratory real/fake structural residual distributions."""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"


def _ensure_project_environment() -> None:
    """Re-execute with the project-local interpreter when needed."""

    if not PROJECT_PYTHON.exists():
        raise RuntimeError(f"Project environment is missing: {PROJECT_PYTHON}")
    if Path(sys.executable).resolve() != PROJECT_PYTHON.resolve():
        os.execv(str(PROJECT_PYTHON), [str(PROJECT_PYTHON), *sys.argv])


_ensure_project_environment()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


GROUP_FIELDS = [
    "label_name",
    "num_videos",
    "videos_with_rsd_evidence",
    "videos_with_depth_evidence",
    "total_valid_rsd_pairs",
    "total_valid_depth_transitions",
    "rsd_log_mean",
    "rsd_log_max",
    "rsd_log_topk_mean",
    "depth_cons_raw_mean",
    "depth_cons_raw_max",
    "depth_cons_raw_p95",
    "mean_scale_prior_coverage_ratio",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Create exploratory plots from one structural evaluation run."
    )
    parser.add_argument(
        "--evaluation_dir",
        default=str(PROJECT_ROOT / "outputs/evaluation/pilot_6video"),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Plot directory; defaults to <evaluation_dir>/analysis.",
    )
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read a required CSV file."""

    if not path.exists():
        raise FileNotFoundError(f"Required evaluation CSV is missing: {path}")
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _float(value: object) -> float:
    """Convert CSV values to float, preserving missing evidence as NaN."""

    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def _finite(values: Iterable[object]) -> np.ndarray:
    """Return only finite numeric values."""

    array = np.asarray([_float(value) for value in values], dtype=float)
    return array[np.isfinite(array)]


def _mean(values: Iterable[object]) -> float:
    """Return a finite mean or NaN."""

    array = _finite(values)
    return float(np.mean(array)) if array.size else math.nan


def _max(values: Iterable[object]) -> float:
    """Return a finite maximum or NaN."""

    array = _finite(values)
    return float(np.max(array)) if array.size else math.nan


def _is_true(value: object) -> bool:
    """Parse CSV booleans."""

    return str(value).strip().lower() in {"1", "true", "yes"}


def _collect_pair_values(
    evaluation_dir: Path,
    video_rows: Sequence[Mapping[str, str]],
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """Collect pair/transition values grouped by real/fake labels."""

    rsd: dict[str, list[float]] = {"real": [], "fake": []}
    depth: dict[str, list[float]] = {"real": [], "fake": []}
    for video in video_rows:
        video_id = str(video["video_id"])
        group = str(video["label_name"])
        if group not in rsd:
            continue
        rsd_path = evaluation_dir / "videos" / video_id / "rsd_pairs.csv"
        if rsd_path.exists():
            for row in _read_csv(rsd_path):
                value = _float(row.get("R_sd_log"))
                if math.isfinite(value):
                    rsd[group].append(value)
        depth_path = evaluation_dir / "videos" / video_id / "depth_consistency_pairs.csv"
        if depth_path.exists():
            for row in _read_csv(depth_path):
                if not _is_true(row.get("valid")):
                    continue
                value = _float(row.get("raw_residual"))
                if math.isfinite(value):
                    depth[group].append(value)
    return rsd, depth


def _save_distribution(
    grouped: Mapping[str, Sequence[float]],
    output_path: Path,
    title: str,
    xlabel: str,
) -> None:
    """Save an overlaid finite-value histogram with an empty-data message."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    colors = {"real": "#4C78A8", "fake": "#E45756"}
    plotted = False
    all_values = [value for values in grouped.values() for value in values]
    if all_values:
        low, high = min(all_values), max(all_values)
        if math.isclose(low, high):
            bins = np.linspace(low - 0.05, high + 0.05, 12)
        else:
            bins = np.linspace(low, high, 20)
        for group in ("real", "fake"):
            values = list(grouped.get(group, []))
            if values:
                ax.hist(
                    values,
                    bins=bins,
                    alpha=0.58,
                    label=f"{group} (n={len(values)})",
                    color=colors[group],
                    edgecolor="white",
                )
                plotted = True
    if not plotted:
        ax.text(0.5, 0.5, "No valid evidence", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.legend()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _bar_colors(rows: Sequence[Mapping[str, str]]) -> list[str]:
    """Return stable real/fake colors for video bars."""

    return ["#4C78A8" if row["label_name"] == "real" else "#E45756" for row in rows]


def _save_count_bars(
    rows: Sequence[Mapping[str, str]],
    field: str,
    output_path: Path,
    title: str,
    ylabel: str,
) -> None:
    """Save per-video evidence counts without converting missing evidence to scores."""

    names = [row["video_id"] for row in rows]
    values = [max(0.0, _float(row.get(field))) for row in rows]
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    bars = ax.bar(names, values, color=_bar_colors(rows), edgecolor="black", linewidth=0.7)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{int(value)}", ha="center", va="bottom", fontsize=8)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", labelrotation=35)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _save_scale_prior_coverage(
    rows: Sequence[Mapping[str, str]], output_path: Path
) -> None:
    """Save stacked exact/alias/missing/unreliable object coverage counts."""

    names = [row["video_id"] for row in rows]
    fields = [
        ("exact_prior_objects", "exact", "#59A14F"),
        ("alias_prior_objects", "alias", "#76B7B2"),
        ("missing_prior_objects", "missing", "#F28E2B"),
        ("unreliable_prior_objects", "unreliable", "#B07AA1"),
    ]
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    bottom = np.zeros(len(rows), dtype=float)
    for field, label, color in fields:
        values = np.asarray([max(0.0, _float(row.get(field))) for row in rows])
        ax.bar(names, values, bottom=bottom, label=label, color=color, edgecolor="white")
        bottom += values
    ax.set_title("Scale Prior Coverage by Video")
    ax.set_ylabel("object observations")
    ax.tick_params(axis="x", labelrotation=35)
    ax.legend(ncols=2)
    ax.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _save_track_quality(rows: Sequence[Mapping[str, str]], output_path: Path) -> None:
    """Save track evidence and association-error diagnostics in two panels."""

    names = [row["video_id"] for row in rows]
    x = np.arange(len(rows), dtype=float)
    width = 0.36
    tracks = np.asarray([max(0.0, _float(row.get("num_tracks"))) for row in rows])
    transitions = np.asarray(
        [max(0.0, _float(row.get("valid_depth_transitions"))) for row in rows]
    )
    duplicates = np.asarray(
        [max(0.0, _float(row.get("duplicate_track_frame_count"))) for row in rows]
    )
    one_to_many = np.asarray(
        [max(0.0, _float(row.get("one_to_many_assignment_count"))) for row in rows]
    )

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    axes[0].bar(x - width / 2, tracks, width, label="tracks", color="#4C78A8")
    axes[0].bar(x + width / 2, transitions, width, label="valid transitions", color="#59A14F")
    axes[0].set_title("Track Evidence")
    axes[0].legend()
    axes[1].bar(x - width / 2, duplicates, width, label="duplicate track-frame", color="#F28E2B")
    axes[1].bar(x + width / 2, one_to_many, width, label="one-to-many", color="#E15759")
    axes[1].set_title("Association Diagnostics")
    axes[1].legend()
    for ax in axes:
        ax.set_xticks(x, names, rotation=35, ha="right")
        ax.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_group_summary(rows: Sequence[Mapping[str, str]]) -> list[dict[str, object]]:
    """Aggregate video-level features by label without treating clips as samples."""

    summaries: list[dict[str, object]] = []
    for group in ("real", "fake"):
        group_rows = [row for row in rows if row["label_name"] == group]
        summaries.append(
            {
                "label_name": group,
                "num_videos": len(group_rows),
                "videos_with_rsd_evidence": sum(int(row["valid_rsd_pairs"]) > 0 for row in group_rows),
                "videos_with_depth_evidence": sum(int(row["valid_depth_transitions"]) > 0 for row in group_rows),
                "total_valid_rsd_pairs": sum(int(row["valid_rsd_pairs"]) for row in group_rows),
                "total_valid_depth_transitions": sum(int(row["valid_depth_transitions"]) for row in group_rows),
                "rsd_log_mean": _mean(row["rsd_log_mean"] for row in group_rows),
                "rsd_log_max": _max(row["rsd_log_max"] for row in group_rows),
                "rsd_log_topk_mean": _mean(row["rsd_log_topk_mean"] for row in group_rows),
                "depth_cons_raw_mean": _mean(row["depth_cons_raw_mean"] for row in group_rows),
                "depth_cons_raw_max": _max(row["depth_cons_raw_max"] for row in group_rows),
                "depth_cons_raw_p95": _mean(row["depth_cons_raw_p95"] for row in group_rows),
                "mean_scale_prior_coverage_ratio": _mean(row["scale_prior_coverage_ratio"] for row in group_rows),
            }
        )
    return summaries


def _write_group_summary(rows: Sequence[Mapping[str, object]], output_path: Path) -> None:
    """Save exploratory group summary with NaN preserved."""

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=GROUP_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: (
                        "NaN"
                        if isinstance(row.get(field), float) and math.isnan(float(row[field]))
                        else row.get(field, "")
                    )
                    for field in GROUP_FIELDS
                }
            )


def _format_metric(value: object) -> str:
    """Format a metric for the text report."""

    number = _float(value)
    return "NaN" if not math.isfinite(number) else f"{number:.6f}"


def _write_text_summary(
    video_rows: Sequence[Mapping[str, str]],
    group_rows: Sequence[Mapping[str, object]],
    output_path: Path,
) -> None:
    """Write a compact evidence and quality report for human review."""

    lines = [
        "Pilot structural residual evaluation summary",
        "",
        "Depth values are monocular relative depth, not metric meters.",
        "Project convention: larger depth means farther.",
        "This is exploratory: 2 fake videos cannot define final thresholds or formal metrics.",
        "Clips are not treated as independent video samples.",
        "",
    ]
    for group in ("real", "fake"):
        lines.append(f"[{group} videos]")
        for row in video_rows:
            if row["label_name"] != group:
                continue
            lines.append(
                f"{row['video_id']}: status={row['status']}, "
                f"R_sd(mean/max/topk)={_format_metric(row['rsd_log_mean'])}/"
                f"{_format_metric(row['rsd_log_max'])}/"
                f"{_format_metric(row['rsd_log_topk_mean'])}, "
                f"R_depth_raw(mean/max/p95)={_format_metric(row['depth_cons_raw_mean'])}/"
                f"{_format_metric(row['depth_cons_raw_max'])}/"
                f"{_format_metric(row['depth_cons_raw_p95'])}, "
                f"evidence={row['valid_rsd_pairs']} pairs, "
                f"{row['valid_depth_transitions']} transitions, "
                f"quality={row['quality_reason']}"
            )
        lines.append("")
    lines.append("[group-level exploratory means over videos]")
    for row in group_rows:
        lines.append(
            f"{row['label_name']}: n={row['num_videos']}, "
            f"R_sd_mean={_format_metric(row['rsd_log_mean'])}, "
            f"R_depth_raw_mean={_format_metric(row['depth_cons_raw_mean'])}, "
            f"pairs={row['total_valid_rsd_pairs']}, "
            f"transitions={row['total_valid_depth_transitions']}"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(evaluation_dir: Path, output_dir: Path) -> list[dict[str, object]]:
    """Generate six figures and video-level exploratory summaries."""

    video_rows = _read_csv(evaluation_dir / "per_video_features.csv")
    output_dir.mkdir(parents=True, exist_ok=True)
    rsd_values, depth_values = _collect_pair_values(evaluation_dir, video_rows)

    _save_distribution(
        rsd_values,
        output_dir / "rsd_log_distribution.png",
        "R_sd Log Residual Distribution",
        "R_sd_log",
    )
    _save_distribution(
        depth_values,
        output_dir / "depth_cons_raw_distribution.png",
        "R_depth_cons Raw Residual Distribution",
        "raw residual",
    )
    _save_count_bars(
        video_rows,
        "valid_rsd_pairs",
        output_dir / "valid_rsd_pairs_per_video.png",
        "Valid R_sd Pairs per Video",
        "valid pairs",
    )
    _save_count_bars(
        video_rows,
        "valid_depth_transitions",
        output_dir / "valid_depth_transitions_per_video.png",
        "Valid Depth Transitions per Video",
        "valid transitions",
    )
    _save_scale_prior_coverage(video_rows, output_dir / "scale_prior_coverage.png")
    _save_track_quality(video_rows, output_dir / "track_quality.png")

    group_rows = build_group_summary(video_rows)
    _write_group_summary(group_rows, output_dir / "group_summary.csv")
    _write_text_summary(video_rows, group_rows, output_dir / "analysis_summary.txt")
    print(f"Saved six exploratory figures and summaries under: {output_dir}")
    print("No AUC, accuracy, optimal threshold, or clip-as-sample metric was computed.")
    return group_rows


def main() -> None:
    """Run exploratory structural residual distribution analysis."""

    args = parse_args()
    evaluation_dir = Path(args.evaluation_dir)
    output_dir = Path(args.output_dir) if args.output_dir else evaluation_dir / "analysis"
    analyze(evaluation_dir, output_dir)


if __name__ == "__main__":
    main()
