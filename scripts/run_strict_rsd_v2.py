#!/usr/bin/env python3
"""Run the dimension-aligned strict physical R_sd v2 baseline."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from semantic3d.dimension_aligned_scale_depth import (  # noqa: E402
    DimensionAlignedPriorResolver,
    compute_dimension_aligned_rsd,
    load_dimension_aligned_prior_resolver,
)
from semantic3d.io import load_clip_observation  # noqa: E402
from semantic3d.observations import FrameObservationJSON  # noqa: E402
from semantic3d.projected_measurement import (  # noqa: E402
    load_projected_measurement_rules,
)


PAIR_FIELDS = [
    "video_id", "frame_index", "object_a_id", "object_b_id", "object_a_label", "object_b_label",
    "object_a_canonical_label", "object_b_canonical_label", "object_a_prior_status", "object_b_prior_status",
    "object_a_prior_resolution", "object_b_prior_resolution", "object_a_prior_low", "object_a_prior_high",
    "object_b_prior_low", "object_b_prior_high", "characteristic_dimension_a", "characteristic_dimension_b",
    "projected_measurement_type_a", "projected_measurement_type_b", "compatibility_group_a",
    "compatibility_group_b", "projected_measurement_a", "projected_measurement_b", "measurement_quality_a",
    "measurement_quality_b", "measurement_invalid_reason_a", "measurement_invalid_reason_b", "gate_passed_a",
    "gate_passed_b", "gate_score_a", "gate_score_b", "gate_reasons_a", "gate_reasons_b",
    "failed_gate_reasons_a", "failed_gate_reasons_b", "observed_depth_ratio", "expected_ratio_low",
    "expected_ratio_high", "observed_log_ratio", "expected_log_low", "expected_log_high", "rsd_ratio",
    "rsd_log", "distance_to_interval", "evidence_tier", "combined_available", "prior_version", "prior_source",
    "depth_mode", "valid", "skip_reason", "explanation_level", "explanation_text",
]

VIDEO_FIELDS = [
    "video_id", "label", "label_name", "num_frames", "num_detected_objects", "num_candidate_pairs",
    "num_valid_pairs", "num_skipped_pairs", "valid_pair_ratio", "num_frames_with_valid_rsd",
    "video_rsd_coverage", "strict_high_num_pairs", "strict_high_rsd_log_mean", "strict_high_rsd_log_max",
    "strict_high_rsd_log_topk_mean", "conditional_num_pairs", "conditional_rsd_log_mean",
    "conditional_rsd_log_max", "conditional_rsd_log_topk_mean", "combined_available",
    "combined_rsd_log_mean", "combined_rsd_log_max", "combined_rsd_log_topk_mean", "prior_strict_high_count",
    "prior_conditional_count", "prior_pose_sensitive_count", "prior_unsupported_count", "prior_missing_count",
    "status", "primary_failure_reason",
]

COVERAGE_FIELDS = [
    "video_id", "label", "object_count", "resolved_label", "prior_resolution", "reliability_status",
    "characteristic_dimension", "projected_measurement", "compatibility_group", "source_count",
    "pose_sensitivity", "available_in_v2", "reliability_reason",
]


def sha256_file(path: Path) -> str:
    """Return a stable SHA-256 digest."""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> list[dict[str, object]]:
    """Load a video-level real/fake manifest without treating clips as samples."""

    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    seen: set[str] = set()
    output: list[dict[str, object]] = []
    for row in rows:
        video_id = str(row.get("video_id", ""))
        if not video_id or video_id in seen:
            raise ValueError(f"Empty or duplicate video_id: {video_id!r}")
        seen.add(video_id)
        label = int(row["label"])
        label_name = str(row["label_name"])
        if label not in {0, 1} or label_name != ("real" if label == 0 else "fake"):
            raise ValueError(f"Invalid real/fake label for {video_id}")
        output.append({**row, "label": label, "label_name": label_name})
    return output


def load_unique_frames(observation_dir: Path) -> list[FrameObservationJSON]:
    """Load globally indexed frames and remove overlap from clip windows."""

    frames: dict[int, FrameObservationJSON] = {}
    for path in sorted(observation_dir.rglob("*.json")):
        clip = load_clip_observation(path)
        for frame in clip.frames:
            frames.setdefault(int(frame.frame_index), frame)
    if not frames:
        raise FileNotFoundError(f"No associated observations found: {observation_dir}")
    return [frames[index] for index in sorted(frames)]


def _nan() -> float:
    return float("nan")


def _finite(values: Iterable[object]) -> np.ndarray:
    output: list[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            output.append(number)
    return np.asarray(output, dtype=float)


def _summary(values: Iterable[object], topk: int) -> tuple[float, float, float]:
    array = _finite(values)
    if array.size == 0:
        return _nan(), _nan(), _nan()
    k = min(max(1, topk), int(array.size))
    return float(np.mean(array)), float(np.max(array)), float(np.mean(np.sort(array)[-k:]))


def _serialize(value: object) -> object:
    return "NaN" if isinstance(value, float) and math.isnan(value) else value


def save_csv(rows: Sequence[Mapping[str, object]], path: Path, fields: Sequence[str]) -> None:
    """Write CSV while preserving missing residual evidence as explicit NaN."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _serialize(row.get(field, "")) for field in fields})


def evaluate_frames(
    video_id: str,
    frames: Sequence[FrameObservationJSON],
    resolver: DimensionAlignedPriorResolver,
    rules_path: Path,
    depth_mode: str,
) -> list[dict[str, object]]:
    """Evaluate every unordered same-frame pair with strict v2 checks."""

    rules = load_projected_measurement_rules(rules_path)
    rows: list[dict[str, object]] = []
    for frame in frames:
        if len(frame.objects) < 2:
            rows.append(
                {
                    "video_id": video_id,
                    "frame_index": int(frame.frame_index),
                    "valid": False,
                    "skip_reason": "insufficient_objects",
                    "evidence_tier": "unavailable",
                    "prior_version": resolver.metadata.get("prior_version", "strict_physical_v2"),
                    "prior_source": "physical",
                    "depth_mode": depth_mode,
                    "rsd_ratio": _nan(),
                    "rsd_log": _nan(),
                    "explanation_level": "insufficient_evidence",
                    "explanation_text": "该帧不足两个检测对象，无法形成对象对；这是证据不足而不是零残差。",
                }
            )
            continue
        for index, obj_a in enumerate(frame.objects):
            for obj_b in frame.objects[index + 1 :]:
                row = compute_dimension_aligned_rsd(
                    frame, obj_a, obj_b, resolver, rules, depth_mode=depth_mode
                )
                row["video_id"] = video_id
                rows.append(row)
    return rows


def coverage_rows(
    video_id: str,
    frames: Sequence[FrameObservationJSON],
    resolver: DimensionAlignedPriorResolver,
) -> tuple[list[dict[str, object]], Counter[str]]:
    """Summarize observed v2 prior status without changing any prior."""

    labels = Counter(obj.label for frame in frames for obj in frame.objects)
    counts: Counter[str] = Counter()
    rows: list[dict[str, object]] = []
    for label, count in sorted(labels.items()):
        resolved = resolver.resolve(label)
        entry = resolved.entry
        status = entry.reliability_status if entry else "missing"
        counts[status] += count
        rows.append(
            {
                "video_id": video_id,
                "label": label,
                "object_count": count,
                "resolved_label": resolved.resolved_label,
                "prior_resolution": resolved.resolution,
                "reliability_status": status,
                "characteristic_dimension": entry.characteristic_dimension if entry else "",
                "projected_measurement": entry.projected_measurement if entry else "",
                "compatibility_group": entry.compatibility_group if entry else "",
                "source_count": entry.source_count if entry else 0,
                "pose_sensitivity": entry.pose_sensitivity if entry else "",
                "available_in_v2": entry.available if entry else False,
                "reliability_reason": entry.reliability_reason if entry else "missing prior",
            }
        )
    return rows, counts


def aggregate_video(
    manifest_row: Mapping[str, object],
    frames: Sequence[FrameObservationJSON],
    pair_rows: Sequence[Mapping[str, object]],
    prior_counts: Counter[str],
    topk: int,
) -> dict[str, object]:
    """Aggregate strict_high and conditional evidence separately."""

    candidates = [row for row in pair_rows if row.get("skip_reason") != "insufficient_objects"]
    valid = [row for row in candidates if bool(row.get("valid"))]
    strict = [row for row in valid if row.get("evidence_tier") == "strict_high"]
    conditional = [row for row in valid if row.get("evidence_tier") == "conditional_physical"]
    strict_stats = _summary((row.get("rsd_log") for row in strict), topk)
    conditional_stats = _summary((row.get("rsd_log") for row in conditional), topk)
    combined_stats = _summary((row.get("rsd_log") for row in valid), topk)
    frames_with_valid = {int(row["frame_index"]) for row in valid}
    skip_counts = Counter(str(row.get("skip_reason")) for row in pair_rows if row.get("skip_reason"))
    candidate_count = len(candidates)
    return {
        "video_id": manifest_row["video_id"],
        "label": manifest_row["label"],
        "label_name": manifest_row["label_name"],
        "num_frames": len(frames),
        "num_detected_objects": sum(len(frame.objects) for frame in frames),
        "num_candidate_pairs": candidate_count,
        "num_valid_pairs": len(valid),
        "num_skipped_pairs": candidate_count - len(valid),
        "valid_pair_ratio": float(len(valid) / candidate_count) if candidate_count else 0.0,
        "num_frames_with_valid_rsd": len(frames_with_valid),
        "video_rsd_coverage": float(len(frames_with_valid) / len(frames)) if frames else 0.0,
        "strict_high_num_pairs": len(strict),
        "strict_high_rsd_log_mean": strict_stats[0],
        "strict_high_rsd_log_max": strict_stats[1],
        "strict_high_rsd_log_topk_mean": strict_stats[2],
        "conditional_num_pairs": len(conditional),
        "conditional_rsd_log_mean": conditional_stats[0],
        "conditional_rsd_log_max": conditional_stats[1],
        "conditional_rsd_log_topk_mean": conditional_stats[2],
        "combined_available": bool(valid),
        "combined_rsd_log_mean": combined_stats[0],
        "combined_rsd_log_max": combined_stats[1],
        "combined_rsd_log_topk_mean": combined_stats[2],
        "prior_strict_high_count": prior_counts["strict_high"],
        "prior_conditional_count": prior_counts["conditional_physical"],
        "prior_pose_sensitive_count": prior_counts["pose_sensitive"],
        "prior_unsupported_count": prior_counts["unsupported"] + prior_counts["insufficient_source"],
        "prior_missing_count": prior_counts["missing"],
        "status": "ok" if valid else "insufficient_rsd_evidence",
        "primary_failure_reason": skip_counts.most_common(1)[0][0] if skip_counts else "",
    }


def _plot_v2_scores(video_rows: Sequence[Mapping[str, object]], output_path: Path) -> None:
    """Plot tier-separated finite scores and mark no-evidence videos as N/A."""

    x = np.arange(len(video_rows))
    names = [str(row["video_id"]) for row in video_rows]
    strict = np.asarray([float(row["strict_high_rsd_log_mean"]) for row in video_rows])
    conditional = np.asarray([float(row["conditional_rsd_log_mean"]) for row in video_rows])
    fig, ax = plt.subplots(figsize=(10, 5.5))
    width = 0.36
    if np.isfinite(strict).any():
        ax.bar(x[np.isfinite(strict)] - width / 2, strict[np.isfinite(strict)], width, label="strict_high")
    if np.isfinite(conditional).any():
        ax.bar(x[np.isfinite(conditional)] + width / 2, conditional[np.isfinite(conditional)], width, label="conditional_physical")
    for index in x[~(np.isfinite(strict) | np.isfinite(conditional))]:
        ax.text(index, 0.5, "N/A\nno evidence", ha="center", color="#666666", fontsize=8)
    ax.set_ylim(bottom=0)
    ax.set_xticks(x, names, rotation=35, ha="right")
    ax.set_ylabel("R_sd_log mean")
    ax.set_title("Dimension-Aligned Strict Physical R_sd v2")
    if np.isfinite(strict).any() or np.isfinite(conditional).any():
        ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def run_v2_baseline(
    manifest_path: Path,
    observation_root: Path,
    prior_path: Path,
    output_dir: Path,
    depth_mode: str = "real_depth_invert",
    device: str = "cpu",
    topk: int = 3,
    overwrite: bool = False,
) -> dict[str, object]:
    """Run strict v2 using frozen inputs and no learned/empirical fallback."""

    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output exists: {output_dir}; pass --overwrite.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolver = load_dimension_aligned_prior_resolver(prior_path)
    rules_value = str(resolver.metadata.get("projected_measurement_rules"))
    rules_path = Path(rules_value)
    if not rules_path.is_absolute():
        rules_path = PROJECT_ROOT / rules_path
    manifest = load_manifest(manifest_path)

    all_pairs: list[dict[str, object]] = []
    all_coverage: list[dict[str, object]] = []
    video_rows: list[dict[str, object]] = []
    for manifest_row in manifest:
        video_id = str(manifest_row["video_id"])
        observation_dir = observation_root / "videos" / video_id / "associated_observations"
        frames = load_unique_frames(observation_dir)
        pair_rows = evaluate_frames(video_id, frames, resolver, rules_path, depth_mode)
        coverage, prior_counts = coverage_rows(video_id, frames, resolver)
        video_rows.append(aggregate_video(manifest_row, frames, pair_rows, prior_counts, topk))
        all_pairs.extend(pair_rows)
        all_coverage.extend(coverage)

    skip_counts = Counter(str(row["skip_reason"]) for row in all_pairs if row.get("skip_reason"))
    skip_rows = [{"skip_reason": reason, "count": count} for reason, count in skip_counts.most_common()]
    save_csv(video_rows, output_dir / "per_video_rsd_features.csv", VIDEO_FIELDS)
    save_csv(all_pairs, output_dir / "per_pair_rsd_details.csv", PAIR_FIELDS)
    save_csv(all_coverage, output_dir / "rsd_coverage_report.csv", COVERAGE_FIELDS)
    save_csv(skip_rows, output_dir / "rsd_skip_reason_summary.csv", ["skip_reason", "count"])
    save_csv(
        all_pairs,
        output_dir / "rsd_explanations.csv",
        [
            "video_id", "frame_index", "object_a_label", "object_b_label", "evidence_tier", "valid",
            "skip_reason", "observed_log_ratio", "expected_log_low", "expected_log_high", "rsd_log",
            "explanation_level", "explanation_text",
        ],
    )
    _plot_v2_scores(video_rows, output_dir / "visualizations/rsd_score_by_video.png")

    candidates = [row for row in all_pairs if row.get("skip_reason") != "insufficient_objects"]
    valid = [row for row in candidates if bool(row.get("valid"))]
    summary = {
        "total_videos": len(video_rows),
        "videos_with_valid_rsd": sum(int(row["num_valid_pairs"]) > 0 for row in video_rows),
        "videos_without_valid_rsd": sum(int(row["num_valid_pairs"]) == 0 for row in video_rows),
        "total_candidate_pairs": len(candidates),
        "total_valid_pairs": len(valid),
        "valid_pair_ratio": float(len(valid) / len(candidates)) if candidates else 0.0,
        "strict_high_valid_pairs": sum(row.get("evidence_tier") == "strict_high" for row in valid),
        "conditional_valid_pairs": sum(row.get("evidence_tier") == "conditional_physical" for row in valid),
        "top_skip_reasons": skip_counts.most_common(8),
    }
    metadata = {
        "prior_version": resolver.metadata.get("prior_version"),
        "prior_file_hash": sha256_file(prior_path),
        "prior_created_at": resolver.metadata.get("prior_created_at"),
        "prior_source": "physical",
        "projected_measurement_rules": str(rules_path),
        "projected_measurement_rules_hash": sha256_file(rules_path),
        "manifest": str(manifest_path),
        "observation_root": str(observation_root),
        "depth_mode": depth_mode,
        "depth_convention": "larger_is_farther",
        "metric_depth": False,
        "device": device,
        "empirical_pair_prior_enabled": False,
        "pilot_data_used_to_fit_priors_or_gates": False,
        "nan_is_zero": False,
        "summary": summary,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("Dimension-aligned strict physical R_sd v2 summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print("  depth: monocular relative depth, not metric meters")
    print("  convention: larger depth values mean farther objects")
    print(f"Saved strict v2 outputs under: {output_dir}")
    return metadata

