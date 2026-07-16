#!/usr/bin/env python3
"""Run manifest-driven R_sd and R_depth_cons pilot evaluation."""

from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"


def _ensure_project_environment() -> None:
    """Re-execute with the project-local interpreter when needed."""

    if not PROJECT_PYTHON.exists():
        raise RuntimeError(f"Project environment is missing: {PROJECT_PYTHON}")
    if Path(sys.executable).resolve() != PROJECT_PYTHON.resolve():
        os.execv(str(PROJECT_PYTHON), [str(PROJECT_PYTHON), *sys.argv])


_ensure_project_environment()

SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from scripts.build_real_object_observations_from_video import (  # noqa: E402
    build_real_object_observations_from_video,
)
from scripts.run_depth_consistency_pipeline import run_pipeline as run_depth_pipeline  # noqa: E402
from scripts.run_observation_rsd_pipeline import (  # noqa: E402
    compute_rows as compute_rsd_rows,
    save_clip_score_plot,
    save_rows_csv,
)
from semantic3d.io import load_clip_observation  # noqa: E402
from semantic3d.scale_prior import (  # noqa: E402
    ScalePriorResolver,
    load_scale_prior_resolver,
    normalize_label,
)


PER_VIDEO_FIELDS = [
    "video_id",
    "video_path",
    "label",
    "label_name",
    "split",
    "status",
    "num_frames",
    "num_objects",
    "num_tracks",
    "valid_rsd_pairs",
    "rsd_log_mean",
    "rsd_log_max",
    "rsd_log_topk_mean",
    "valid_depth_transitions",
    "depth_cons_raw_mean",
    "depth_cons_raw_max",
    "depth_cons_raw_p95",
    "depth_cons_mean",
    "depth_cons_max",
    "scale_prior_coverage_ratio",
    "exact_prior_objects",
    "alias_prior_objects",
    "missing_prior_objects",
    "unreliable_prior_objects",
    "duplicate_frame_count",
    "duplicate_track_frame_count",
    "one_to_many_assignment_count",
    "skipped_invalid_depth",
    "skipped_missing_reference",
    "quality_reason",
    "error_message",
]

PER_CLIP_FIELDS = [
    "video_id",
    "label",
    "label_name",
    "split",
    "clip_id",
    "num_frames",
    "valid_rsd_pairs",
    "rsd_log_mean",
    "rsd_log_max",
    "rsd_log_topk_mean",
    "valid_depth_transitions",
    "depth_cons_raw_mean",
    "depth_cons_raw_max",
    "depth_cons_raw_p95",
    "depth_cons_mean",
    "depth_cons_max",
]

QUALITY_FIELDS = [
    "video_id",
    "label",
    "label_name",
    "status",
    "num_objects",
    "num_tracks",
    "valid_rsd_pairs",
    "valid_depth_transitions",
    "scale_prior_coverage_ratio",
    "duplicate_track_frame_count",
    "one_to_many_assignment_count",
    "skipped_invalid_depth",
    "skipped_missing_reference",
    "quality_reason",
    "error_message",
]

COVERAGE_FIELDS = [
    "video_id",
    "label",
    "count",
    "has_exact_prior",
    "has_alias_prior",
    "resolved_label",
    "reliable",
    "status",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Run structural residual evaluation from a validated manifest."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs/structural_residual_eval.yaml"),
    )
    parser.add_argument(
        "--output_dir",
        default=str(PROJECT_ROOT / "outputs/evaluation/pilot_6video"),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing evaluation output directory.",
    )
    parser.add_argument(
        "--allow_test_providers",
        action="store_true",
        help="Allow mock providers in temporary test configurations.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> dict[str, Any]:
    """Load and validate an evaluation YAML configuration."""

    with config_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Evaluation config must be a mapping: {config_path}")
    for section in (
        "video",
        "object_detection",
        "depth",
        "scale_depth",
        "depth_consistency",
    ):
        if not isinstance(data.get(section), dict):
            raise ValueError(f"Evaluation config requires mapping section '{section}'.")
    return data


def _resolve_project_path(value: str | Path) -> Path:
    """Resolve a repository-relative or absolute path."""

    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def validate_config(config: Mapping[str, Any], allow_test_providers: bool = False) -> None:
    """Enforce the real pilot settings unless explicitly running tests."""

    object_cfg = config["object_detection"]
    depth_cfg = config["depth"]
    if not allow_test_providers:
        expected = {
            "object_detection.provider": (object_cfg.get("provider"), "real_detector"),
            "depth.provider": (depth_cfg.get("provider"), "real_depth"),
            "depth.depth_mode": (depth_cfg.get("depth_mode"), "real_depth_invert"),
            "depth.invert_depth": (depth_cfg.get("invert_depth"), True),
            "depth.convention": (depth_cfg.get("convention"), "larger_is_farther"),
            "depth.metric_depth": (depth_cfg.get("metric_depth"), False),
        }
        invalid = [
            f"{name}={actual!r} (expected {wanted!r})"
            for name, (actual, wanted) in expected.items()
            if actual != wanted
        ]
        if invalid:
            raise ValueError("Invalid pilot configuration: " + "; ".join(invalid))
    tolerance = float(config["depth_consistency"].get("tolerance", -1.0))
    if tolerance < 0:
        raise ValueError("depth_consistency.tolerance must be non-negative.")


def load_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    """Read and validate a real/fake evaluation manifest."""

    with manifest_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"Manifest is empty: {manifest_path}")
    seen: set[str] = set()
    validated: list[dict[str, Any]] = []
    for row in rows:
        video_id = str(row.get("video_id", "")).strip()
        if not video_id:
            raise ValueError("Manifest contains an empty video_id.")
        if video_id in seen:
            raise ValueError(f"Duplicate video_id in manifest: {video_id}")
        seen.add(video_id)
        label = int(row["label"])
        if label not in {0, 1}:
            raise ValueError(f"Invalid label for {video_id}: {label}")
        if str(row.get("split")) != "val":
            raise ValueError(f"Pilot split must be val for {video_id}.")
        expected_name = "real" if label == 0 else "fake"
        if str(row.get("label_name")) != expected_name:
            raise ValueError(f"Invalid label_name for {video_id}.")
        video_path = _resolve_project_path(str(row["video_path"]))
        if not video_path.is_file():
            raise FileNotFoundError(f"Video path does not exist: {video_path}")
        validated.append({**row, "label": label, "video_path": video_path})
    return validated


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV if it exists, otherwise return an empty list."""

    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _is_true(value: object) -> bool:
    """Interpret common CSV boolean representations."""

    return str(value).strip().lower() in {"1", "true", "yes"}


def _finite(values: Iterable[object]) -> np.ndarray:
    """Convert values to a finite float array."""

    converted: list[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            converted.append(number)
    return np.asarray(converted, dtype=float)


def _summary(values: Iterable[object], topk: int) -> dict[str, float]:
    """Return NaN-aware mean, max, top-k mean, and p95 statistics."""

    array = _finite(values)
    if array.size == 0:
        return {"mean": math.nan, "max": math.nan, "topk_mean": math.nan, "p95": math.nan}
    k = min(max(1, int(topk)), int(array.size))
    largest = np.sort(array)[-k:]
    return {
        "mean": float(np.mean(array)),
        "max": float(np.max(array)),
        "topk_mean": float(np.mean(largest)),
        "p95": float(np.percentile(array, 95)),
    }


def _write_csv(rows: Sequence[Mapping[str, object]], path: Path, fields: list[str]) -> None:
    """Write stable CSV headers and serialize missing evidence as NaN."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            serialized: dict[str, object] = {}
            for field in fields:
                value = row.get(field, "")
                if isinstance(value, float) and math.isnan(value):
                    value = "NaN"
                serialized[field] = value
            writer.writerow(serialized)


def _coverage_rows(
    video_id: str,
    observation_dir: Path,
    resolver: ScalePriorResolver,
) -> tuple[list[dict[str, object]], Counter[str], int]:
    """Count per-object exact, alias, missing, and unreliable prior status."""

    label_counts: Counter[str] = Counter()
    for path in sorted(observation_dir.rglob("*.json")):
        clip = load_clip_observation(path)
        for frame in clip.frames:
            label_counts.update(normalize_label(obj.label) for obj in frame.objects)

    status_counts: Counter[str] = Counter()
    rows: list[dict[str, object]] = []
    for label, count in sorted(label_counts.items()):
        resolved = resolver.resolve(label, require_reliable=True)
        status_counts[resolved.source] += count
        rows.append(
            {
                "video_id": video_id,
                "label": label,
                "count": count,
                "has_exact_prior": label in resolver.scale_priors,
                "has_alias_prior": label in resolver.aliases,
                "resolved_label": (
                    "" if resolved.source == "missing" else resolved.resolved_label
                ),
                "reliable": resolved.reliable,
                "status": resolved.source,
            }
        )
    return rows, status_counts, sum(label_counts.values())


def _depth_args(
    config: Mapping[str, Any],
    observation_dir: Path,
    depth_dir: Path,
    video_output: Path,
) -> SimpleNamespace:
    """Build the argument namespace expected by the existing depth pipeline."""

    cfg = config["depth_consistency"]
    visual_dir = video_output / "visualizations"
    return SimpleNamespace(
        observation_dir=str(observation_dir),
        depth_map_dir=str(depth_dir),
        output_pair_csv=str(video_output / "depth_consistency_pairs.csv"),
        output_track_csv=str(video_output / "depth_consistency_tracks.csv"),
        output_clip_csv=str(video_output / "depth_consistency_clips.csv"),
        associated_observation_dir=str(video_output / "associated_observations"),
        visualization_path=str(visual_dir / "depth_consistency_tracks.png"),
        raw_residual_visualization_path=str(visual_dir / "depth_cons_raw.png"),
        thresholded_residual_visualization_path=str(visual_dir / "depth_cons_thresholded.png"),
        combined_visualization_path=str(visual_dir / "depth_cons_combined.png"),
        depth_mode=str(config["depth"]["depth_mode"]),
        iou_threshold=float(cfg.get("iou_threshold", 0.1)),
        center_distance_threshold=float(cfg.get("center_distance_threshold", 0.25)),
        max_area_ratio=float(cfg.get("max_area_ratio", 3.0)),
        max_frame_gap=int(cfg.get("max_frame_gap", 1)),
        tolerance=float(cfg.get("tolerance", 0.02)),
        topk=int(cfg.get("topk", 3)),
        max_files=None,
    )


def _build_observations(
    manifest_row: Mapping[str, Any],
    config: Mapping[str, Any],
    observation_dir: Path,
    depth_dir: Path,
) -> None:
    """Build real or test observations according to the evaluation config."""

    video_cfg = config["video"]
    object_cfg = config["object_detection"]
    depth_cfg = config["depth"]
    build_real_object_observations_from_video(
        video_path=Path(manifest_row["video_path"]),
        output_dir=observation_dir,
        fps=video_cfg.get("fps"),
        max_frames=video_cfg.get("max_frames"),
        clip_len=int(video_cfg.get("clip_len", 8)),
        stride=int(video_cfg.get("stride", 4)),
        object_provider=str(object_cfg.get("provider", "real_detector")),
        confidence_threshold=float(object_cfg.get("confidence_threshold", 0.3)),
        default_depth=float(depth_cfg.get("default_depth", 5.0)),
        model_path=str(_resolve_project_path(object_cfg.get("model_path", "checkpoints/yolov8n.pt"))),
        device=str(object_cfg.get("device", "cpu")),
        skip_unknown_scale_prior=not bool(object_cfg.get("keep_unknown_scale_prior", True)),
        mock_mode=str(object_cfg.get("mock_mode", "reasonable")),
        depth_provider=str(depth_cfg.get("provider", "real_depth")),
        depth_model_name=str(depth_cfg.get("model_name", "depth-anything/Depth-Anything-V2-Small")),
        invert_depth=bool(depth_cfg.get("invert_depth", True)),
        save_depth_maps=bool(depth_cfg.get("save_depth_maps", True)),
        depth_output_dir=str(depth_dir),
    )


def _quality_status(valid_rsd: int, valid_depth: int, error: str = "") -> str:
    """Classify processing/evidence availability without treating missing as normal."""

    if error:
        return "failed"
    if valid_rsd > 0 and valid_depth > 0:
        return "ok"
    if valid_rsd > 0 or valid_depth > 0:
        return "partial_evidence"
    return "no_evidence"


def _quality_reasons(
    num_objects: int,
    valid_rsd: int,
    valid_depth: int,
    coverage_ratio: float,
    depth_stats: Mapping[str, Any],
) -> str:
    """Return semicolon-separated evidence and association diagnostics."""

    reasons: list[str] = []
    if num_objects == 0:
        reasons.append("no_detected_objects")
    if valid_rsd == 0:
        reasons.append("no_valid_rsd_pairs")
    if valid_depth == 0:
        reasons.append("no_valid_depth_transitions")
    if math.isfinite(coverage_ratio) and coverage_ratio < 1.0:
        reasons.append("incomplete_scale_prior_coverage")
    for key in (
        "duplicate_track_frame_count",
        "one_to_many_assignment_count",
        "skipped_invalid_depth",
        "skipped_missing_reference",
    ):
        if int(depth_stats.get(key, 0)) > 0:
            reasons.append(key)
    return ";".join(reasons) if reasons else "ok"


def _clip_feature_rows(
    manifest_row: Mapping[str, Any],
    observation_dir: Path,
    rsd_rows: list[dict[str, object]],
    depth_pair_rows: list[dict[str, str]],
    rsd_topk: int,
    depth_topk: int,
) -> list[dict[str, object]]:
    """Aggregate original clip windows while keeping videos as the sample unit."""

    rsd_by_clip: dict[str, list[dict[str, object]]] = {}
    for row in rsd_rows:
        rsd_by_clip.setdefault(str(row["clip_id"]), []).append(row)
    valid_depth_rows = [row for row in depth_pair_rows if _is_true(row.get("valid"))]
    output: list[dict[str, object]] = []
    for path in sorted(observation_dir.glob("*.json")):
        clip = load_clip_observation(path)
        frame_set = {int(index) for index in clip.frame_indices}
        clip_rsd = rsd_by_clip.get(clip.clip_id, [])
        clip_depth = [
            row
            for row in valid_depth_rows
            if int(row["previous_frame_index"]) in frame_set
            and int(row["current_frame_index"]) in frame_set
        ]
        rsd_summary = _summary((row["R_sd_log"] for row in clip_rsd), rsd_topk)
        raw_summary = _summary((row["raw_residual"] for row in clip_depth), depth_topk)
        depth_summary = _summary((row["residual"] for row in clip_depth), depth_topk)
        output.append(
            {
                "video_id": manifest_row["video_id"],
                "label": manifest_row["label"],
                "label_name": manifest_row["label_name"],
                "split": manifest_row["split"],
                "clip_id": clip.clip_id,
                "num_frames": len(frame_set),
                "valid_rsd_pairs": len(clip_rsd),
                "rsd_log_mean": rsd_summary["mean"],
                "rsd_log_max": rsd_summary["max"],
                "rsd_log_topk_mean": rsd_summary["topk_mean"],
                "valid_depth_transitions": len(clip_depth),
                "depth_cons_raw_mean": raw_summary["mean"],
                "depth_cons_raw_max": raw_summary["max"],
                "depth_cons_raw_p95": raw_summary["p95"],
                "depth_cons_mean": depth_summary["mean"],
                "depth_cons_max": depth_summary["max"],
            }
        )
    return output


def evaluate_video(
    manifest_row: Mapping[str, Any],
    config: Mapping[str, Any],
    output_root: Path,
    resolver: ScalePriorResolver,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    """Run all structural residual stages for one video."""

    video_id = str(manifest_row["video_id"])
    video_output = output_root / "videos" / video_id
    observation_dir = video_output / "observations"
    depth_dir = video_output / "depth_maps"
    _build_observations(manifest_row, config, observation_dir, depth_dir)

    depth_args = _depth_args(config, observation_dir, depth_dir, video_output)
    depth_stats = run_depth_pipeline(depth_args)
    associated_dir = Path(depth_args.associated_observation_dir)

    unique_rsd_rows, rsd_stats = compute_rsd_rows(
        associated_dir,
        resolver=resolver,
        return_stats=True,
    )
    rsd_csv = video_output / "rsd_pairs.csv"
    save_rows_csv(unique_rsd_rows, rsd_csv)
    save_clip_score_plot(unique_rsd_rows, video_output / "visualizations/rsd_scores.png")

    clip_rsd_rows = compute_rsd_rows(observation_dir, resolver=resolver)
    save_rows_csv(clip_rsd_rows, video_output / "rsd_clip_pairs.csv")
    depth_pair_rows = _read_csv(Path(depth_args.output_pair_csv))
    valid_depth_rows = [row for row in depth_pair_rows if _is_true(row.get("valid"))]

    scale_cfg = config["scale_depth"]
    depth_cfg = config["depth_consistency"]
    rsd_topk = int(scale_cfg.get("topk", 3))
    depth_topk = int(depth_cfg.get("topk", 3))
    rsd_summary = _summary((row["R_sd_log"] for row in unique_rsd_rows), rsd_topk)
    raw_summary = _summary((row["raw_residual"] for row in valid_depth_rows), depth_topk)
    residual_summary = _summary((row["residual"] for row in valid_depth_rows), depth_topk)

    coverage_rows, coverage_counts, coverage_total = _coverage_rows(
        video_id, associated_dir, resolver
    )
    reliable_count = coverage_counts["exact"] + coverage_counts["alias"]
    coverage_ratio = (
        float(reliable_count / coverage_total) if coverage_total > 0 else math.nan
    )
    valid_rsd = len(unique_rsd_rows)
    valid_depth = len(valid_depth_rows)
    num_objects = int(depth_stats.get("total_objects", 0))
    quality_reason = _quality_reasons(
        num_objects, valid_rsd, valid_depth, coverage_ratio, depth_stats
    )

    video_row: dict[str, object] = {
        "video_id": video_id,
        "video_path": str(manifest_row["video_path"]),
        "label": manifest_row["label"],
        "label_name": manifest_row["label_name"],
        "split": manifest_row["split"],
        "status": _quality_status(valid_rsd, valid_depth),
        "num_frames": int(depth_stats.get("total_unique_frames", 0)),
        "num_objects": num_objects,
        "num_tracks": int(depth_stats.get("total_tracks", 0)),
        "valid_rsd_pairs": valid_rsd,
        "rsd_log_mean": rsd_summary["mean"],
        "rsd_log_max": rsd_summary["max"],
        "rsd_log_topk_mean": rsd_summary["topk_mean"],
        "valid_depth_transitions": valid_depth,
        "depth_cons_raw_mean": raw_summary["mean"],
        "depth_cons_raw_max": raw_summary["max"],
        "depth_cons_raw_p95": raw_summary["p95"],
        "depth_cons_mean": residual_summary["mean"],
        "depth_cons_max": residual_summary["max"],
        "scale_prior_coverage_ratio": coverage_ratio,
        "exact_prior_objects": coverage_counts["exact"],
        "alias_prior_objects": coverage_counts["alias"],
        "missing_prior_objects": coverage_counts["missing"],
        "unreliable_prior_objects": coverage_counts["unreliable"],
        "duplicate_frame_count": int(depth_stats.get("duplicate_frame_count", 0)),
        "duplicate_track_frame_count": int(depth_stats.get("duplicate_track_frame_count", 0)),
        "one_to_many_assignment_count": int(depth_stats.get("one_to_many_assignment_count", 0)),
        "skipped_invalid_depth": int(depth_stats.get("skipped_invalid_depth", 0)),
        "skipped_missing_reference": int(depth_stats.get("skipped_missing_reference", 0)),
        "quality_reason": quality_reason,
        "error_message": "",
    }
    clip_rows = _clip_feature_rows(
        manifest_row,
        observation_dir,
        clip_rsd_rows,
        depth_pair_rows,
        rsd_topk,
        depth_topk,
    )
    return video_row, clip_rows, coverage_rows


def _failed_video_row(row: Mapping[str, Any], exc: Exception) -> dict[str, object]:
    """Build a complete failure row while preserving missing evidence as NaN."""

    result: dict[str, object] = {field: math.nan for field in PER_VIDEO_FIELDS}
    result.update(
        {
            "video_id": row["video_id"],
            "video_path": str(row["video_path"]),
            "label": row["label"],
            "label_name": row["label_name"],
            "split": row["split"],
            "status": "failed",
            "num_frames": 0,
            "num_objects": 0,
            "num_tracks": 0,
            "valid_rsd_pairs": 0,
            "valid_depth_transitions": 0,
            "quality_reason": "pipeline_error",
            "error_message": f"{type(exc).__name__}: {exc}",
        }
    )
    return result


def run_evaluation(
    manifest_path: Path,
    config_path: Path,
    output_dir: Path,
    overwrite: bool = False,
    allow_test_providers: bool = False,
) -> list[dict[str, object]]:
    """Run every manifest video and save feature and quality tables."""

    config = load_config(config_path)
    validate_config(config, allow_test_providers=allow_test_providers)
    manifest_rows = load_manifest(manifest_path)
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}. Pass --overwrite."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scale_prior_path = _resolve_project_path(config["scale_depth"]["scale_prior_path"])
    resolver = load_scale_prior_resolver(scale_prior_path)
    video_rows: list[dict[str, object]] = []
    clip_rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []

    for index, row in enumerate(manifest_rows, start=1):
        print(f"\n[{index}/{len(manifest_rows)}] Evaluating {row['video_id']} ({row['label_name']})")
        try:
            video_row, current_clips, current_coverage = evaluate_video(
                row, config, output_dir, resolver
            )
            video_rows.append(video_row)
            clip_rows.extend(current_clips)
            coverage_rows.extend(current_coverage)
            print(
                f"  status={video_row['status']}, objects={video_row['num_objects']}, "
                f"R_sd pairs={video_row['valid_rsd_pairs']}, "
                f"depth transitions={video_row['valid_depth_transitions']}"
            )
        except Exception as exc:
            failed = _failed_video_row(row, exc)
            video_rows.append(failed)
            print(f"  ERROR: {failed['error_message']}", file=sys.stderr)

    _write_csv(video_rows, output_dir / "per_video_features.csv", PER_VIDEO_FIELDS)
    _write_csv(clip_rows, output_dir / "per_clip_features.csv", PER_CLIP_FIELDS)
    quality_rows = [{field: row.get(field, "") for field in QUALITY_FIELDS} for row in video_rows]
    _write_csv(quality_rows, output_dir / "quality_report.csv", QUALITY_FIELDS)
    _write_csv(coverage_rows, output_dir / "scale_prior_coverage.csv", COVERAGE_FIELDS)

    success = sum(row["status"] != "failed" for row in video_rows)
    print(f"\nCompleted videos: {success}/{len(video_rows)}; failed={len(video_rows) - success}")
    print(f"Saved evaluation tables under: {output_dir}")
    print("Depth is monocular relative depth, not metric distance in meters.")
    print("Project convention: larger depth values mean farther objects.")
    return video_rows


def main() -> None:
    """Run the manifest-driven batch evaluation."""

    args = parse_args()
    run_evaluation(
        manifest_path=Path(args.manifest),
        config_path=Path(args.config),
        output_dir=Path(args.output_dir),
        overwrite=args.overwrite,
        allow_test_providers=args.allow_test_providers,
    )


if __name__ == "__main__":
    main()
