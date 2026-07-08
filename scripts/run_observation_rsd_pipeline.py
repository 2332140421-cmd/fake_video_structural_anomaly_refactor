#!/usr/bin/env python3
"""Compute scale-depth residuals from clip observation JSON files."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path


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

from semantic3d.io import load_clip_observation  # noqa: E402
from semantic3d.scale_prior import (  # noqa: E402
    ResolvedScalePrior,
    ScalePriorResolver,
    default_scale_prior_resolver,
)
from semantic3d.scale_depth import (  # noqa: E402
    ObjectObservation,
    scale_depth_residual,
    scale_depth_residual_log,
)


CSV_FIELDS = [
    "video_id",
    "clip_id",
    "frame_index",
    "object_pair",
    "R_sd",
    "R_sd_log",
    "clip_score",
    "expected_mode",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Run scale-depth residual analysis on observation JSON clips."
    )
    parser.add_argument(
        "--observation_dir",
        default=str(PROJECT_ROOT / "outputs" / "observations"),
        help="Directory containing clip observation JSON files.",
    )
    parser.add_argument(
        "--output_csv",
        default=str(PROJECT_ROOT / "outputs" / "results" / "video_rsd_results.csv"),
        help="Path to the output CSV file.",
    )
    parser.add_argument(
        "--visualization_path",
        default=str(
            PROJECT_ROOT / "outputs" / "visualizations" / "video_rsd_clip_scores.png"
        ),
        help="Path to the output PNG score visualization.",
    )
    parser.add_argument(
        "--scale_prior_config",
        default=str(PROJECT_ROOT / "configs" / "scale_priors.yaml"),
        help="YAML file containing scale_priors and aliases.",
    )
    return parser.parse_args()


def _to_scale_depth_objects(frame_objects: list[object]) -> list[ObjectObservation]:
    """Convert JSON object records to scale-depth residual inputs."""

    return [obj.to_scale_depth_observation() for obj in frame_objects]  # type: ignore[attr-defined]


def _resolve_objects(
    objects: list[ObjectObservation],
    resolver: ScalePriorResolver,
    stats: dict[str, int],
) -> list[tuple[ObjectObservation, ResolvedScalePrior]]:
    """Resolve object labels to exact/coarse scale-prior labels with status."""

    resolved_objects: list[tuple[ObjectObservation, ResolvedScalePrior]] = []
    for obj in objects:
        stats["total_objects"] += 1
        resolved = resolver.resolve(obj.label, require_reliable=True)
        if resolved.source == "missing":
            stats["skipped_missing_prior_objects"] += 1
            print(
                f"Skipping object {obj.object_id}: missing scale prior for label "
                f"'{obj.label}'.",
                file=sys.stderr,
            )
        elif resolved.source == "unreliable":
            stats["skipped_unreliable_prior_objects"] += 1
            print(
                f"Skipping object {obj.object_id}: unreliable scale prior for label "
                f"'{obj.label}' resolved to '{resolved.resolved_label}'.",
                file=sys.stderr,
            )
        elif resolved.source == "exact":
            stats["exact_prior_objects"] += 1
        elif resolved.source == "alias":
            stats["alias_prior_objects"] += 1

        converted = (
            ObjectObservation(
                object_id=obj.object_id,
                label=resolved.resolved_label,
                mask_area=obj.mask_area,
                frame_area=obj.frame_area,
                depth=obj.depth,
                confidence=obj.confidence,
            )
        )
        resolved_objects.append((converted, resolved))
    return resolved_objects


def compute_rows(
    observation_dir: Path,
    resolver: ScalePriorResolver | None = None,
    return_stats: bool = False,
) -> list[dict[str, object]] | tuple[list[dict[str, object]], dict[str, int]]:
    """Read observation JSON files and compute frame-level pairwise R_sd rows."""

    resolver = resolver or default_scale_prior_resolver(PROJECT_ROOT)
    scale_priors = resolver.to_scale_prior_map()
    stats = {
        "total_objects": 0,
        "exact_prior_objects": 0,
        "alias_prior_objects": 0,
        "skipped_missing_prior_objects": 0,
        "skipped_unreliable_prior_objects": 0,
        "total_candidate_pairs": 0,
        "computed_pairs": 0,
        "skipped_pairs_missing_prior": 0,
        "skipped_pairs_unreliable_prior": 0,
    }

    json_paths = sorted(observation_dir.rglob("*.json"))
    if not json_paths:
        raise FileNotFoundError(f"No observation JSON files found in {observation_dir}.")

    all_rows: list[dict[str, object]] = []
    for json_path in json_paths:
        clip_obs = load_clip_observation(json_path)
        expected_mode = str(
            clip_obs.metadata.get(
                "expected_mode", clip_obs.metadata.get("mock_mode", "unknown")
            )
        )

        clip_rows: list[dict[str, object]] = []
        for frame in clip_obs.frames:
            raw_objects = _to_scale_depth_objects(frame.objects)
            resolved_objects = _resolve_objects(raw_objects, resolver, stats)
            stats["total_candidate_pairs"] += max(
                0, len(resolved_objects) * (len(resolved_objects) - 1) // 2
            )
            for i, (obj_a, resolved_a) in enumerate(resolved_objects):
                for j, (obj_b, resolved_b) in enumerate(resolved_objects):
                    if i >= j:
                        continue
                    if "missing" in {resolved_a.source, resolved_b.source}:
                        stats["skipped_pairs_missing_prior"] += 1
                        continue
                    if "unreliable" in {resolved_a.source, resolved_b.source}:
                        stats["skipped_pairs_unreliable_prior"] += 1
                        continue
                    try:
                        residual, _details = scale_depth_residual(
                            obj_a, obj_b, scale_priors
                        )
                        residual_log, _details_log = scale_depth_residual_log(
                            obj_a, obj_b, scale_priors
                        )
                    except KeyError as exc:
                        stats["skipped_pairs_missing_prior"] += 1
                        print(
                            f"Skipping pair {obj_a.object_id}->{obj_b.object_id}: {exc}",
                            file=sys.stderr,
                        )
                        continue
                    stats["computed_pairs"] += 1
                    clip_rows.append(
                        {
                            "video_id": clip_obs.video_id,
                            "clip_id": clip_obs.clip_id,
                            "frame_index": frame.frame_index,
                            "object_pair": f"{obj_a.object_id}->{obj_b.object_id}",
                            "R_sd": float(residual),
                            "R_sd_log": float(residual_log),
                            "clip_score": 0.0,
                            "expected_mode": expected_mode,
                        }
                    )

        clip_score = max((float(row["R_sd_log"]) for row in clip_rows), default=0.0)
        for row in clip_rows:
            row["clip_score"] = clip_score
        all_rows.extend(clip_rows)

    _print_stats(stats)
    if return_stats:
        return all_rows, stats
    return all_rows


def _print_stats(stats: dict[str, int]) -> None:
    """Print scale-prior resolver and pair-computation statistics."""

    print("Scale-prior resolution stats:")
    for key in [
        "total_objects",
        "exact_prior_objects",
        "alias_prior_objects",
        "skipped_missing_prior_objects",
        "skipped_unreliable_prior_objects",
        "total_candidate_pairs",
        "computed_pairs",
        "skipped_pairs_missing_prior",
        "skipped_pairs_unreliable_prior",
    ]:
        print(f"  {key}: {stats[key]}")


def save_rows_csv(rows: list[dict[str, object]], output_csv: Path) -> None:
    """Save residual rows to CSV."""

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in CSV_FIELDS})


def save_clip_score_plot(rows: list[dict[str, object]], output_path: Path) -> None:
    """Save a basic bar chart of clip-level scale-depth scores."""

    clip_scores: dict[str, tuple[float, str]] = {}
    for row in rows:
        clip_scores[str(row["clip_id"])] = (
            float(row["clip_score"]),
            str(row["expected_mode"]),
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not clip_scores:
        fig, ax = plt.subplots(figsize=(8.0, 4.8))
        ax.set_title("Scale-Depth Residual from Observation JSON Clips")
        ax.text(
            0.5,
            0.5,
            "No valid object pairs with scale priors",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
        return

    clip_ids = list(clip_scores)
    scores = [clip_scores[clip_id][0] for clip_id in clip_ids]
    modes = [clip_scores[clip_id][1] for clip_id in clip_ids]
    colors = ["#4C78A8" if mode == "reasonable" else "#E45756" for mode in modes]
    hatches = ["" if mode == "reasonable" else "//" for mode in modes]

    fig, ax = plt.subplots(figsize=(max(8.0, len(clip_ids) * 0.8), 4.8))
    bars = ax.bar(clip_ids, scores, color=colors, edgecolor="black", linewidth=0.8)
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    ax.set_title("Scale-Depth Residual from Observation JSON Clips")
    ax.set_xlabel("clip_id")
    ax.set_ylabel("clip_score = max R_sd_log")
    ax.tick_params(axis="x", labelrotation=45)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def run_pipeline(
    observation_dir: Path,
    output_csv: Path,
    visualization_path: Path,
    scale_prior_config: Path | None = None,
) -> list[dict[str, object]]:
    """Compute residual rows, save CSV, and save a score visualization."""

    resolver = (
        load_resolver_from_path(scale_prior_config)
        if scale_prior_config is not None
        else default_scale_prior_resolver(PROJECT_ROOT)
    )
    rows = compute_rows(observation_dir, resolver=resolver)
    save_rows_csv(rows, output_csv)
    save_clip_score_plot(rows, visualization_path)
    return rows


def load_resolver_from_path(path: Path | None) -> ScalePriorResolver:
    """Load a resolver from a path, falling back to project default."""

    if path is None:
        return default_scale_prior_resolver(PROJECT_ROOT)
    from semantic3d.scale_prior import load_scale_prior_resolver

    return load_scale_prior_resolver(path)


def main() -> None:
    """Run the observation-to-R_sd pipeline from the command line."""

    args = parse_args()
    rows = run_pipeline(
        observation_dir=Path(args.observation_dir),
        output_csv=Path(args.output_csv),
        visualization_path=Path(args.visualization_path),
        scale_prior_config=Path(args.scale_prior_config),
    )
    print(f"Saved {len(rows)} residual row(s) to {args.output_csv}")
    print(f"Saved visualization to {args.visualization_path}")


if __name__ == "__main__":
    main()
