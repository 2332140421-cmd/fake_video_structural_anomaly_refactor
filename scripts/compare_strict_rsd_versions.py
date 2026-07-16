#!/usr/bin/env python3
"""Compare strict physical R_sd v1 and dimension-aligned v2 coverage."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_PYTHON = PROJECT_ROOT / ".venv/bin/python"


def _ensure_project_environment() -> None:
    if not PROJECT_PYTHON.exists():
        raise RuntimeError(f"Project Python is missing: {PROJECT_PYTHON}")
    if Path(sys.executable).resolve() != PROJECT_PYTHON.resolve():
        os.execv(str(PROJECT_PYTHON), [str(PROJECT_PYTHON), *sys.argv])


_ensure_project_environment()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


FIELDS = [
    "prior_version",
    "evidence_tier",
    "total_candidate_pairs",
    "valid_pairs",
    "valid_pair_ratio",
    "videos_with_valid_rsd",
    "videos_without_valid_rsd",
    "top_skip_reasons",
    "real_valid_pairs",
    "fake_valid_pairs",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare strict R_sd v1 and v2 evidence coverage.")
    parser.add_argument("--v1_dir", required=True)
    parser.add_argument("--v2_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _is_true(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _comparison_row(
    prior_version: str,
    tier: str,
    pairs: Sequence[Mapping[str, str]],
    video_labels: Mapping[str, str],
    selected_tier: str | None,
) -> dict[str, object]:
    candidates = [row for row in pairs if row.get("skip_reason") != "insufficient_objects"]
    valid = [
        row
        for row in candidates
        if _is_true(row.get("valid"))
        and (selected_tier is None or row.get("evidence_tier") == selected_tier)
    ]
    valid_videos = {str(row["video_id"]) for row in valid}
    skip_counts = Counter(str(row.get("skip_reason")) for row in candidates if row.get("skip_reason"))
    return {
        "prior_version": prior_version,
        "evidence_tier": tier,
        "total_candidate_pairs": len(candidates),
        "valid_pairs": len(valid),
        "valid_pair_ratio": float(len(valid) / len(candidates)) if candidates else 0.0,
        "videos_with_valid_rsd": len(valid_videos),
        "videos_without_valid_rsd": len(video_labels) - len(valid_videos),
        "top_skip_reasons": json.dumps(skip_counts.most_common(5), ensure_ascii=False),
        "real_valid_pairs": sum(video_labels.get(str(row["video_id"])) == "real" for row in valid),
        "fake_valid_pairs": sum(video_labels.get(str(row["video_id"])) == "fake" for row in valid),
    }


def build_comparison_rows(v1_dir: Path, v2_dir: Path) -> list[dict[str, object]]:
    """Build four coverage rows without calculating classification metrics."""

    v1_pairs = _read_csv(v1_dir / "per_pair_rsd_details.csv")
    v2_pairs = _read_csv(v2_dir / "per_pair_rsd_details.csv")
    v1_videos = _read_csv(v1_dir / "per_video_rsd_features.csv")
    v2_videos = _read_csv(v2_dir / "per_video_rsd_features.csv")
    v1_labels = {row["video_id"]: row["label_name"] for row in v1_videos}
    v2_labels = {row["video_id"]: row["label_name"] for row in v2_videos}
    if v1_labels != v2_labels:
        raise ValueError("v1 and v2 must contain the same video-level labels.")
    return [
        _comparison_row("strict_physical_v1", "physical_all", v1_pairs, v1_labels, None),
        _comparison_row("strict_physical_v2", "strict_high", v2_pairs, v2_labels, "strict_high"),
        _comparison_row(
            "strict_physical_v2", "conditional_physical", v2_pairs, v2_labels, "conditional_physical"
        ),
        _comparison_row("strict_physical_v2", "all_available", v2_pairs, v2_labels, None),
    ]


def save_comparison(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _labels(rows: Sequence[Mapping[str, object]]) -> list[str]:
    return [f"{row['prior_version']}\n{row['evidence_tier']}" for row in rows]


def save_plots(
    rows: Sequence[Mapping[str, object]],
    v2_pairs: Sequence[Mapping[str, str]],
    output_dir: Path,
) -> None:
    """Save pair coverage, video coverage, and v2 skip-reason figures."""

    visual_dir = output_dir / "visualizations"
    visual_dir.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(rows))
    labels = _labels(rows)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ratios = [float(row["valid_pair_ratio"]) for row in rows]
    ax.bar(x, ratios, color=["#9C9C9C", "#4C78A8", "#F28E2B", "#59A14F"])
    for index, row in enumerate(rows):
        ax.text(index, ratios[index] + 0.01, f"{row['valid_pairs']}/{row['total_candidate_pairs']}", ha="center")
    ax.set_ylim(0, max(0.12, max(ratios, default=0.0) * 1.35))
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylabel("valid pair ratio")
    ax.set_title("Strict R_sd v1/v2 Pair Evidence Coverage")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(visual_dir / "v1_v2_pair_coverage.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    video_counts = [int(row["videos_with_valid_rsd"]) for row in rows]
    ax.bar(x, video_counts, color=["#9C9C9C", "#4C78A8", "#F28E2B", "#59A14F"])
    ax.set_ylim(0, max(6.5, max(video_counts, default=0) + 1))
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylabel("videos with valid R_sd")
    ax.set_title("Strict R_sd v1/v2 Video Evidence Coverage (6 videos)")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(visual_dir / "v1_v2_video_coverage.png", dpi=180)
    plt.close(fig)

    skip_counts = Counter(str(row.get("skip_reason")) for row in v2_pairs if row.get("skip_reason"))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    if skip_counts:
        names, values = zip(*skip_counts.most_common())
        ax.bar(names, values, color="#E15759")
        ax.tick_params(axis="x", labelrotation=40)
    else:
        ax.text(0.5, 0.5, "No skipped v2 pairs", transform=ax.transAxes, ha="center")
    ax.set_ylabel("count")
    ax.set_title("Dimension-Aligned Strict v2 Skip Reasons")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(visual_dir / "v2_skip_reasons.png", dpi=180)
    plt.close(fig)


def compare(v1_dir: Path, v2_dir: Path, output_dir: Path) -> list[dict[str, object]]:
    rows = build_comparison_rows(v1_dir, v2_dir)
    save_comparison(rows, output_dir / "v1_v2_comparison.csv")
    save_plots(rows, _read_csv(v2_dir / "per_pair_rsd_details.csv"), output_dir)
    print("Saved v1/v2 engineering coverage comparison. No AUC, accuracy, or threshold was computed.")
    for row in rows:
        print(
            f"  {row['prior_version']} / {row['evidence_tier']}: "
            f"pairs={row['valid_pairs']}/{row['total_candidate_pairs']}, "
            f"videos={row['videos_with_valid_rsd']}/6"
        )
    return rows


def main() -> None:
    args = parse_args()
    compare(Path(args.v1_dir), Path(args.v2_dir), Path(args.output_dir))


if __name__ == "__main__":
    main()

