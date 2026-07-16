#!/usr/bin/env python3
"""Evaluate a frozen strict physical R_sd prior with explicit missing evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
from collections import Counter
from pathlib import Path
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
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from semantic3d.io import load_clip_observation  # noqa: E402
from semantic3d.observations import FrameObservationJSON, ObjectObservationJSON  # noqa: E402
from semantic3d.scale_prior import normalize_label  # noqa: E402
from semantic3d.strict_scale_prior import (  # noqa: E402
    StrictPhysicalScalePriorResolver,
    StrictResolvedPhysicalPrior,
    load_strict_physical_prior_resolver,
)


PAIR_FIELDS = [
    "video_id",
    "frame_index",
    "object_a_id",
    "object_b_id",
    "object_a_label",
    "object_b_label",
    "object_a_canonical_label",
    "object_b_canonical_label",
    "object_a_prior_status",
    "object_b_prior_status",
    "object_a_prior_audit_status",
    "object_b_prior_audit_status",
    "object_a_prior_low",
    "object_a_prior_high",
    "object_b_prior_low",
    "object_b_prior_high",
    "projection_scale_a",
    "projection_scale_b",
    "relative_depth_a",
    "relative_depth_b",
    "observed_depth_ratio",
    "expected_ratio_low",
    "expected_ratio_high",
    "observed_log_ratio",
    "expected_log_low",
    "expected_log_high",
    "rsd_ratio",
    "rsd_log",
    "distance_to_interval",
    "valid",
    "skip_reason",
    "prior_source",
    "pair_prior_class",
    "explanation_level",
    "explanation_text",
]

VIDEO_FIELDS = [
    "video_id",
    "label",
    "label_name",
    "num_frames",
    "num_detected_objects",
    "num_candidate_pairs",
    "num_valid_pairs",
    "num_skipped_pairs",
    "valid_pair_ratio",
    "num_frames_with_valid_rsd",
    "video_rsd_coverage",
    "rsd_ratio_mean",
    "rsd_ratio_max",
    "rsd_ratio_topk_mean",
    "rsd_log_mean",
    "rsd_log_max",
    "rsd_log_topk_mean",
    "prior_exact_count",
    "prior_alias_count",
    "prior_unreliable_count",
    "prior_missing_count",
    "status",
    "primary_failure_reason",
]

COVERAGE_FIELDS = [
    "video_id",
    "label",
    "object_count",
    "resolved_label",
    "prior_status",
    "audit_status",
    "reliable",
    "characteristic_dimension",
    "dimension_definition",
    "unit",
    "prior_low",
    "prior_high",
    "source_count",
    "reliability_reason",
    "allowed_in_strict_rsd",
]

SKIP_FIELDS = ["video_id", "skip_reason", "count"]
EXPLANATION_FIELDS = [
    "video_id",
    "frame_index",
    "object_a_label",
    "object_b_label",
    "observed_log_ratio",
    "expected_log_low",
    "expected_log_high",
    "distance_to_interval",
    "valid",
    "skip_reason",
    "explanation_level",
    "explanation_text",
]

PRIOR_EXPLANATION_FIELDS = [
    "observed_label",
    "resolved_label",
    "characteristic_dimension",
    "dimension_definition",
    "physical_range",
    "unit",
    "source_count",
    "audit_status",
    "allowed_in_strict_rsd",
    "decision_explanation",
]


def parse_args() -> argparse.Namespace:
    """Parse strict baseline arguments."""

    parser = argparse.ArgumentParser(description="Run frozen strict physical R_sd baseline.")
    parser.add_argument(
        "--manifest",
        "--video_manifest",
        dest="manifest",
        default=str(PROJECT_ROOT / "data/manifests/pilot_real_fake.csv"),
    )
    parser.add_argument(
        "--observation_root",
        default=str(PROJECT_ROOT / "outputs/evaluation/pilot_6video"),
        help="Evaluation root containing videos/<id>/associated_observations.",
    )
    parser.add_argument(
        "--scale_prior_config",
        required=True,
        help="Frozen strict prior file; it is never modified by this script.",
    )
    parser.add_argument(
        "--evaluation_config",
        default=str(PROJECT_ROOT / "configs/structural_residual_eval.yaml"),
    )
    parser.add_argument(
        "--output_dir",
        default=str(PROJECT_ROOT / "outputs/evaluation/rsd_strict_baseline"),
    )
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--depth_mode", default="real_depth_invert")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of an input config."""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> list[dict[str, object]]:
    """Read the fixed real/fake manifest and preserve video-level labels."""

    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    output: list[dict[str, object]] = []
    seen: set[str] = set()
    for row in rows:
        video_id = str(row["video_id"])
        if video_id in seen:
            raise ValueError(f"Duplicate video_id: {video_id}")
        seen.add(video_id)
        label = int(row["label"])
        label_name = str(row["label_name"])
        if label not in {0, 1} or label_name != ("real" if label == 0 else "fake"):
            raise ValueError(f"Invalid real/fake label for {video_id}")
        output.append({**row, "label": label, "label_name": label_name})
    return output


def _nan() -> float:
    """Return a readable NaN sentinel for missing evidence."""

    return float("nan")


def _finite_positive(value: object) -> bool:
    """Check whether a scalar is finite and positive."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number > 0.0


def _projection_scale(obj: ObjectObservationJSON) -> float:
    """Compute sqrt(projected area / frame area), or NaN if invalid."""

    if not _finite_positive(obj.mask_area) or not _finite_positive(obj.frame_area):
        return _nan()
    if float(obj.mask_area) > float(obj.frame_area):
        return _nan()
    return math.sqrt(float(obj.mask_area) / float(obj.frame_area))


def _depth_reference(frame: FrameObservationJSON) -> float:
    """Compute a per-frame median used only to report relative object depth."""

    depths = [float(obj.depth) for obj in frame.objects if _finite_positive(obj.depth)]
    return float(np.median(depths)) if depths else _nan()


def _distance(value: float, low: float, high: float) -> float:
    """Return scalar distance to a closed interval."""

    if value < low:
        return low - value
    if value > high:
        return value - high
    return 0.0


def _base_pair(video_id: str, frame_index: int) -> dict[str, object]:
    """Create a pair row whose unavailable numeric evidence starts as NaN."""

    row: dict[str, object] = {field: "" for field in PAIR_FIELDS}
    for field in (
        "object_a_prior_low",
        "object_a_prior_high",
        "object_b_prior_low",
        "object_b_prior_high",
        "projection_scale_a",
        "projection_scale_b",
        "relative_depth_a",
        "relative_depth_b",
        "observed_depth_ratio",
        "expected_ratio_low",
        "expected_ratio_high",
        "observed_log_ratio",
        "expected_log_low",
        "expected_log_high",
        "rsd_ratio",
        "rsd_log",
        "distance_to_interval",
    ):
        row[field] = _nan()
    row.update(
        {
            "video_id": video_id,
            "frame_index": frame_index,
            "valid": False,
            "prior_source": "physical",
        }
    )
    return row


def _canonical_label(
    obj: ObjectObservationJSON, resolved: StrictResolvedPhysicalPrior
) -> str:
    """Choose the strict resolved label, falling back to observation metadata."""

    if resolved.resolution != "missing":
        return resolved.resolved_label
    return normalize_label(obj.canonical_label or obj.label)


def _skip_explanation(
    obj_a: ObjectObservationJSON,
    obj_b: ObjectObservationJSON,
    resolved_a: StrictResolvedPhysicalPrior,
    resolved_b: StrictResolvedPhysicalPrior,
    reason: str,
) -> str:
    """Generate a user-readable explanation for missing strict evidence."""

    if reason == "insufficient_objects":
        return "该帧不足两个检测对象，无法构造对象对；结果表示证据不足，而不是正常残差为零。"
    side = "a" if reason.endswith("_a") else "b"
    obj = obj_a if side == "a" else obj_b
    resolved = resolved_a if side == "a" else resolved_b
    if reason.startswith("missing_prior"):
        return (
            f"{obj.label} 没有经过来源审核的物理尺度先验，本对象对被标记为证据不足，"
            "而非正常残差为零。"
        )
    if reason.startswith("unreliable_prior"):
        audit = resolved.audit_status
        detail = resolved.entry.reliability_reason if resolved.entry else "缺少可靠依据"
        return (
            f"{obj.label} 的严格物理先验状态为 {audit}（{detail}），当前不提供可靠 R_sd；"
            "本对象对被标记为证据不足，而非正常残差为零。"
        )
    if reason.startswith("invalid_depth"):
        return f"{obj.label} 的深度不是有限正值，无法计算尺度—深度比，本对象对证据无效。"
    if reason.startswith("invalid_projection_area"):
        return f"{obj.label} 的投影面积或帧面积无效，无法计算等效投影尺度，本对象对证据无效。"
    return "本对象对没有形成有效的严格物理 R_sd 证据。"


def _pair_prior_class(
    a: StrictResolvedPhysicalPrior, b: StrictResolvedPhysicalPrior
) -> str:
    """Classify a candidate pair for aggregate prior-coverage accounting."""

    if "missing" in {a.resolution, b.resolution}:
        return "missing"
    if "unreliable" in {a.resolution, b.resolution}:
        return "unreliable"
    if "alias" in {a.resolution, b.resolution}:
        return "alias"
    return "exact"


def evaluate_object_pair(
    video_id: str,
    frame_index: int,
    obj_a: ObjectObservationJSON,
    obj_b: ObjectObservationJSON,
    depth_reference: float,
    resolver: StrictPhysicalScalePriorResolver,
) -> dict[str, object]:
    """Compute or explicitly skip one strict physical object-pair residual."""

    resolved_a = resolver.resolve(obj_a.label)
    resolved_b = resolver.resolve(obj_b.label)
    projection_a = _projection_scale(obj_a)
    projection_b = _projection_scale(obj_b)
    relative_a = float(obj_a.depth) / depth_reference if _finite_positive(obj_a.depth) and _finite_positive(depth_reference) else _nan()
    relative_b = float(obj_b.depth) / depth_reference if _finite_positive(obj_b.depth) and _finite_positive(depth_reference) else _nan()
    row = _base_pair(video_id, frame_index)
    row.update(
        {
            "object_a_id": obj_a.object_id,
            "object_b_id": obj_b.object_id,
            "object_a_label": obj_a.label,
            "object_b_label": obj_b.label,
            "object_a_canonical_label": _canonical_label(obj_a, resolved_a),
            "object_b_canonical_label": _canonical_label(obj_b, resolved_b),
            "object_a_prior_status": resolved_a.resolution,
            "object_b_prior_status": resolved_b.resolution,
            "object_a_prior_audit_status": resolved_a.audit_status,
            "object_b_prior_audit_status": resolved_b.audit_status,
            "object_a_prior_low": resolved_a.low,
            "object_a_prior_high": resolved_a.high,
            "object_b_prior_low": resolved_b.low,
            "object_b_prior_high": resolved_b.high,
            "projection_scale_a": projection_a,
            "projection_scale_b": projection_b,
            "relative_depth_a": relative_a,
            "relative_depth_b": relative_b,
            "pair_prior_class": _pair_prior_class(resolved_a, resolved_b),
        }
    )

    # Observed geometry does not depend on a physical prior. Preserve it for
    # diagnostics even when the pair is later skipped as missing/unreliable.
    if _finite_positive(obj_a.depth) and _finite_positive(obj_b.depth):
        row["observed_depth_ratio"] = float(obj_a.depth) / float(obj_b.depth)
        row["observed_log_ratio"] = math.log(float(obj_a.depth)) - math.log(
            float(obj_b.depth)
        )

    checks = [
        (not _finite_positive(obj_a.depth), "invalid_depth_a"),
        (not _finite_positive(obj_b.depth), "invalid_depth_b"),
        (not math.isfinite(projection_a), "invalid_projection_area_a"),
        (not math.isfinite(projection_b), "invalid_projection_area_b"),
        (resolved_a.resolution == "missing", "missing_prior_a"),
        (resolved_b.resolution == "missing", "missing_prior_b"),
        (not resolved_a.reliable, "unreliable_prior_a"),
        (not resolved_b.reliable, "unreliable_prior_b"),
    ]
    skip_reason = next((reason for failed, reason in checks if failed), "")
    if skip_reason:
        row.update(
            {
                "skip_reason": skip_reason,
                "explanation_level": "insufficient_evidence",
                "explanation_text": _skip_explanation(
                    obj_a, obj_b, resolved_a, resolved_b, skip_reason
                ),
            }
        )
        return row

    ratio = float(row["observed_depth_ratio"])
    ratio_low = (resolved_a.low / resolved_b.high) * (projection_b / projection_a)
    ratio_high = (resolved_a.high / resolved_b.low) * (projection_b / projection_a)
    observed_log = float(row["observed_log_ratio"])
    log_low = math.log(resolved_a.low / resolved_b.high) + math.log(projection_b) - math.log(projection_a)
    log_high = math.log(resolved_a.high / resolved_b.low) + math.log(projection_b) - math.log(projection_a)
    rsd_ratio = _distance(ratio, ratio_low, ratio_high)
    rsd_log = _distance(observed_log, log_low, log_high)
    normal = rsd_log == 0.0
    if normal:
        explanation = (
            f"{obj_a.label} 与 {obj_b.label} 的观测尺度—深度比位于可靠物理先验范围内，"
            "当前对象对未发现明显结构异常。"
        )
        level = "normal"
    else:
        boundary = "下界" if observed_log < log_low else "上界"
        explanation = (
            f"{obj_a.label} 与 {obj_b.label} 的观测尺度—深度比超出可靠参考区间{boundary}，"
            "投影大小与相对深度关系存在异常偏差。"
        )
        level = "anomaly"
    row.update(
        {
            "observed_depth_ratio": ratio,
            "expected_ratio_low": ratio_low,
            "expected_ratio_high": ratio_high,
            "observed_log_ratio": observed_log,
            "expected_log_low": log_low,
            "expected_log_high": log_high,
            "rsd_ratio": rsd_ratio,
            "rsd_log": rsd_log,
            "distance_to_interval": rsd_log,
            "valid": True,
            "skip_reason": "",
            "explanation_level": level,
            "explanation_text": explanation,
        }
    )
    return row


def evaluate_frame(
    video_id: str,
    frame: FrameObservationJSON,
    resolver: StrictPhysicalScalePriorResolver,
) -> list[dict[str, object]]:
    """Evaluate all unordered object pairs or record insufficient_objects."""

    if len(frame.objects) < 2:
        row = _base_pair(video_id, int(frame.frame_index))
        row.update(
            {
                "skip_reason": "insufficient_objects",
                "pair_prior_class": "none",
                "explanation_level": "insufficient_evidence",
                "explanation_text": _skip_explanation(
                    frame.objects[0] if frame.objects else _placeholder_object(frame),
                    _placeholder_object(frame),
                    resolver.resolve("__missing__"),
                    resolver.resolve("__missing__"),
                    "insufficient_objects",
                ),
            }
        )
        return [row]
    reference = _depth_reference(frame)
    return [
        evaluate_object_pair(video_id, int(frame.frame_index), obj_a, obj_b, reference, resolver)
        for index, obj_a in enumerate(frame.objects)
        for obj_b in frame.objects[index + 1 :]
    ]


def _placeholder_object(frame: FrameObservationJSON) -> ObjectObservationJSON:
    """Create an internal placeholder used only for frame-level explanations."""

    return ObjectObservationJSON(
        object_id="",
        label="object",
        mask_area=1.0,
        frame_area=max(1.0, float(frame.width * frame.height)),
        depth=1.0,
    )


def _finite(values: Iterable[object]) -> np.ndarray:
    """Convert an iterable to finite floats."""

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
    """Return mean/max/top-k or NaN for no valid evidence."""

    array = _finite(values)
    if array.size == 0:
        return _nan(), _nan(), _nan()
    k = min(max(1, topk), int(array.size))
    return float(np.mean(array)), float(np.max(array)), float(np.mean(np.sort(array)[-k:]))


def load_unique_frames(observation_dir: Path) -> list[FrameObservationJSON]:
    """Load globally deduplicated frames from associated observation JSON."""

    frames: dict[int, FrameObservationJSON] = {}
    for path in sorted(observation_dir.rglob("*.json")):
        clip = load_clip_observation(path)
        for frame in clip.frames:
            frames.setdefault(int(frame.frame_index), frame)
    if not frames:
        raise FileNotFoundError(f"No associated observation frames found in {observation_dir}")
    return [frames[index] for index in sorted(frames)]


def _coverage_for_video(
    video_id: str,
    frames: Sequence[FrameObservationJSON],
    resolver: StrictPhysicalScalePriorResolver,
) -> tuple[list[dict[str, object]], Counter[str]]:
    """Build observed-label prior coverage and object-level status counts."""

    labels = Counter(normalize_label(obj.label) for frame in frames for obj in frame.objects)
    status_counts: Counter[str] = Counter()
    rows: list[dict[str, object]] = []
    for label, count in sorted(labels.items()):
        resolved = resolver.resolve(label)
        status_counts[resolved.resolution] += count
        entry = resolved.entry
        rows.append(
            {
                "video_id": video_id,
                "label": label,
                "object_count": count,
                "resolved_label": resolved.resolved_label,
                "prior_status": resolved.resolution,
                "audit_status": resolved.audit_status,
                "reliable": resolved.reliable,
                "characteristic_dimension": entry.characteristic_dimension if entry else "",
                "dimension_definition": entry.dimension_definition if entry else "",
                "unit": entry.unit if entry else "",
                "prior_low": resolved.low,
                "prior_high": resolved.high,
                "source_count": entry.source_count if entry else 0,
                "reliability_reason": entry.reliability_reason if entry else "missing reviewed prior",
                "allowed_in_strict_rsd": resolved.reliable,
            }
        )
    return rows, status_counts


def aggregate_video(
    manifest_row: Mapping[str, object],
    frames: Sequence[FrameObservationJSON],
    pair_rows: Sequence[Mapping[str, object]],
    status_counts: Counter[str],
    topk: int,
) -> dict[str, object]:
    """Aggregate strict pair evidence, preserving no-evidence values as NaN."""

    candidate_rows = [row for row in pair_rows if row.get("skip_reason") != "insufficient_objects"]
    valid_rows = [row for row in candidate_rows if bool(row.get("valid"))]
    skipped_rows = [row for row in candidate_rows if not bool(row.get("valid"))]
    frames_with_valid = {int(row["frame_index"]) for row in valid_rows}
    ratio_stats = _summary((row["rsd_ratio"] for row in valid_rows), topk)
    log_stats = _summary((row["rsd_log"] for row in valid_rows), topk)
    skip_counts = Counter(str(row["skip_reason"]) for row in pair_rows if row.get("skip_reason"))
    candidate_count = len(candidate_rows)
    valid_count = len(valid_rows)
    return {
        "video_id": manifest_row["video_id"],
        "label": manifest_row["label"],
        "label_name": manifest_row["label_name"],
        "num_frames": len(frames),
        "num_detected_objects": sum(len(frame.objects) for frame in frames),
        "num_candidate_pairs": candidate_count,
        "num_valid_pairs": valid_count,
        "num_skipped_pairs": len(skipped_rows),
        "valid_pair_ratio": float(valid_count / candidate_count) if candidate_count else 0.0,
        "num_frames_with_valid_rsd": len(frames_with_valid),
        "video_rsd_coverage": float(len(frames_with_valid) / len(frames)) if frames else 0.0,
        "rsd_ratio_mean": ratio_stats[0],
        "rsd_ratio_max": ratio_stats[1],
        "rsd_ratio_topk_mean": ratio_stats[2],
        "rsd_log_mean": log_stats[0],
        "rsd_log_max": log_stats[1],
        "rsd_log_topk_mean": log_stats[2],
        "prior_exact_count": status_counts["exact"],
        "prior_alias_count": status_counts["alias"],
        "prior_unreliable_count": status_counts["unreliable"],
        "prior_missing_count": status_counts["missing"],
        "status": "ok" if valid_count else "insufficient_rsd_evidence",
        "primary_failure_reason": skip_counts.most_common(1)[0][0] if skip_counts else "",
    }


def _serialize(value: object) -> object:
    """Serialize NaN explicitly instead of coercing it to zero."""

    return "NaN" if isinstance(value, float) and math.isnan(value) else value


def save_csv(rows: Sequence[Mapping[str, object]], path: Path, fields: list[str]) -> None:
    """Save stable CSV output with explicit NaN values."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _serialize(row.get(field, "")) for field in fields})


def _label_colors(rows: Sequence[Mapping[str, object]]) -> list[str]:
    """Map real/fake videos to stable colors."""

    return ["#4C78A8" if row["label_name"] == "real" else "#E45756" for row in rows]


def plot_score_by_video(rows: Sequence[Mapping[str, object]], output_path: Path) -> list[str]:
    """Plot only finite scores and separately annotate no-evidence videos."""

    names = [str(row["video_id"]) for row in rows]
    values = np.asarray([float(row["rsd_log_topk_mean"]) for row in rows], dtype=float)
    finite = np.isfinite(values)
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    if np.any(finite):
        ax.bar(x[finite], values[finite], color=np.asarray(_label_colors(rows))[finite], edgecolor="black")
    upper = max(1.0, float(np.max(values[finite])) * 1.25) if np.any(finite) else 1.0
    for index in x[~finite]:
        ax.text(index, upper * 0.55, "N/A\nno evidence", ha="center", va="center", color="#666666", fontsize=8)
    ax.set_ylim(0, upper)
    ax.set_xticks(x, names, rotation=35, ha="right")
    ax.set_ylabel("R_sd_log top-k mean")
    ax.set_title("Strict Physical R_sd by Video (NaN is not zero)")
    ax.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return [names[index] for index in x[finite]]


def _plot_valid_ratio(rows: Sequence[Mapping[str, object]], output_path: Path) -> None:
    """Plot pair and frame coverage with pair counts."""

    names = [str(row["video_id"]) for row in rows]
    x = np.arange(len(rows))
    width = 0.36
    pair_ratio = [float(row["valid_pair_ratio"]) for row in rows]
    frame_ratio = [float(row["video_rsd_coverage"]) for row in rows]
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.bar(x - width / 2, pair_ratio, width, label="valid pair ratio", color="#59A14F")
    ax.bar(x + width / 2, frame_ratio, width, label="frames with valid R_sd", color="#76B7B2")
    for index, row in enumerate(rows):
        ax.text(index - width / 2, pair_ratio[index] + 0.02, f"{row['num_valid_pairs']}/{row['num_candidate_pairs']}", ha="center", fontsize=8)
    ax.set_ylim(0, 1.12)
    ax.set_xticks(x, names, rotation=35, ha="right")
    ax.set_ylabel("coverage ratio")
    ax.set_title("Strict R_sd Evidence Coverage")
    ax.legend()
    ax.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_skip_reasons(rows: Sequence[Mapping[str, object]], output_path: Path) -> None:
    """Plot aggregate skip reasons."""

    counts = Counter(str(row["skip_reason"]) for row in rows if row.get("skip_reason"))
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    if counts:
        labels, values = zip(*counts.most_common())
        ax.bar(labels, values, color="#F28E2B", edgecolor="black")
        ax.tick_params(axis="x", labelrotation=40)
    else:
        ax.text(0.5, 0.5, "No skipped pairs", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Strict R_sd Skip Reasons")
    ax.set_ylabel("count")
    ax.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_distribution(rows: Sequence[Mapping[str, object]], output_path: Path) -> None:
    """Plot finite video scores by label; never substitute NaN with zero."""

    grouped = {
        group: [
            float(row["rsd_log_mean"])
            for row in rows
            if row["label_name"] == group and math.isfinite(float(row["rsd_log_mean"]))
        ]
        for group in ("real", "fake")
    }
    fig, ax = plt.subplots(figsize=(8.5, 5.3))
    if any(grouped.values()):
        for index, group in enumerate(("real", "fake")):
            values = grouped[group]
            if values:
                ax.scatter([index] * len(values), values, s=60, label=f"{group} n={len(values)}")
        ax.set_xticks([0, 1], ["real", "fake"])
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No valid strict R_sd evidence in either group", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([0, 1], ["real", "fake"])
    ax.set_title("Exploratory Real/Fake Strict R_sd Distribution")
    ax.set_ylabel("video R_sd_log mean")
    ax.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _prior_explanations(
    coverage_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Build one understandable decision explanation per observed/resolved label."""

    unique: dict[tuple[str, str], Mapping[str, object]] = {}
    for row in coverage_rows:
        unique.setdefault((str(row["label"]), str(row["resolved_label"])), row)
    output: list[dict[str, object]] = []
    for (_, _), row in sorted(unique.items()):
        allowed = bool(row["allowed_in_strict_rsd"])
        low, high = row["prior_low"], row["prior_high"]
        physical_range = (
            "unavailable"
            if not (isinstance(low, (int, float)) and math.isfinite(float(low)))
            else f"[{float(low):.6f}, {float(high):.6f}] {row['unit']}"
        )
        decision = (
            "通过全部自动审核规则，允许作为 strict physical R_sd 证据。"
            if allowed
            else f"未通过 reliable_single 审核（{row['audit_status']}: {row['reliability_reason']}），严格基线跳过。"
        )
        output.append(
            {
                "observed_label": row["label"],
                "resolved_label": row["resolved_label"],
                "characteristic_dimension": row["characteristic_dimension"],
                "dimension_definition": row["dimension_definition"],
                "physical_range": physical_range,
                "unit": row["unit"],
                "source_count": row["source_count"],
                "audit_status": row["audit_status"],
                "allowed_in_strict_rsd": allowed,
                "decision_explanation": decision,
            }
        )
    return output


def _mean_finite(values: Iterable[object]) -> float:
    """Return finite mean or NaN."""

    array = _finite(values)
    return float(np.mean(array)) if array.size else _nan()


def _print_summary(
    video_rows: Sequence[Mapping[str, object]],
    pair_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Print and return the required strict baseline summary."""

    candidates = [row for row in pair_rows if row.get("skip_reason") != "insufficient_objects"]
    valid = [row for row in candidates if bool(row.get("valid"))]
    pair_classes = Counter(str(row.get("pair_prior_class")) for row in candidates)
    skip_counts = Counter(str(row["skip_reason"]) for row in pair_rows if row.get("skip_reason"))
    real_values = [row["rsd_log_mean"] for row in video_rows if row["label_name"] == "real"]
    fake_values = [row["rsd_log_mean"] for row in video_rows if row["label_name"] == "fake"]
    summary = {
        "total_videos": len(video_rows),
        "real_videos": sum(row["label_name"] == "real" for row in video_rows),
        "fake_videos": sum(row["label_name"] == "fake" for row in video_rows),
        "videos_with_valid_rsd": sum(int(row["num_valid_pairs"]) > 0 for row in video_rows),
        "videos_without_valid_rsd": sum(int(row["num_valid_pairs"]) == 0 for row in video_rows),
        "total_candidate_pairs": len(candidates),
        "total_valid_pairs": len(valid),
        "valid_pair_ratio": float(len(valid) / len(candidates)) if candidates else 0.0,
        "exact_prior_pairs": pair_classes["exact"],
        "alias_prior_pairs": pair_classes["alias"],
        "unreliable_prior_pairs": pair_classes["unreliable"],
        "missing_prior_pairs": pair_classes["missing"],
        "top_skip_reasons": skip_counts.most_common(5),
        "real_rsd_log_mean": _mean_finite(real_values),
        "fake_rsd_log_mean": _mean_finite(fake_values),
    }
    print("Strict physical R_sd baseline summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    return summary


def run_baseline(
    manifest_path: Path,
    observation_root: Path,
    prior_path: Path,
    evaluation_config_path: Path,
    output_dir: Path,
    topk: int = 3,
    overwrite: bool = False,
) -> dict[str, object]:
    """Run the strict baseline without modifying or learning physical priors."""

    if topk < 1:
        raise ValueError("topk must be >= 1.")
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output exists: {output_dir}; pass --overwrite.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolver = load_strict_physical_prior_resolver(prior_path)
    manifest = load_manifest(manifest_path)

    all_pairs: list[dict[str, object]] = []
    all_coverage: list[dict[str, object]] = []
    video_rows: list[dict[str, object]] = []
    for manifest_row in manifest:
        video_id = str(manifest_row["video_id"])
        observation_dir = observation_root / "videos" / video_id / "associated_observations"
        frames = load_unique_frames(observation_dir)
        pair_rows = [
            pair
            for frame in frames
            for pair in evaluate_frame(video_id, frame, resolver)
        ]
        coverage_rows, status_counts = _coverage_for_video(video_id, frames, resolver)
        video_row = aggregate_video(
            manifest_row, frames, pair_rows, status_counts, topk=topk
        )
        all_pairs.extend(pair_rows)
        all_coverage.extend(coverage_rows)
        video_rows.append(video_row)

    skip_rows = [
        {"video_id": video_id, "skip_reason": reason, "count": count}
        for (video_id, reason), count in sorted(
            Counter(
                (str(row["video_id"]), str(row["skip_reason"]))
                for row in all_pairs
                if row.get("skip_reason")
            ).items()
        )
    ]
    explanations = [
        {field: row.get(field, "") for field in EXPLANATION_FIELDS}
        for row in all_pairs
    ]
    save_csv(video_rows, output_dir / "per_video_rsd_features.csv", VIDEO_FIELDS)
    save_csv(all_pairs, output_dir / "per_pair_rsd_details.csv", PAIR_FIELDS)
    save_csv(all_coverage, output_dir / "rsd_coverage_report.csv", COVERAGE_FIELDS)
    save_csv(skip_rows, output_dir / "rsd_skip_reason_summary.csv", SKIP_FIELDS)
    save_csv(explanations, output_dir / "rsd_explanations.csv", EXPLANATION_FIELDS)
    save_csv(
        _prior_explanations(all_coverage),
        output_dir / "prior_explanations.csv",
        PRIOR_EXPLANATION_FIELDS,
    )

    visual_dir = output_dir / "visualizations"
    visual_dir.mkdir(parents=True, exist_ok=True)
    plotted = plot_score_by_video(video_rows, visual_dir / "rsd_score_by_video.png")
    _plot_valid_ratio(video_rows, visual_dir / "rsd_valid_pair_ratio.png")
    _plot_skip_reasons(all_pairs, visual_dir / "rsd_skip_reasons.png")
    _plot_distribution(video_rows, visual_dir / "rsd_real_fake_distribution.png")

    summary = _print_summary(video_rows, all_pairs)
    metadata = {
        "prior_version": resolver.metadata["prior_version"],
        "prior_file_hash": sha256_file(prior_path),
        "prior_created_at": resolver.metadata.get("prior_created_at"),
        "source_report_path": resolver.metadata.get("source_report_path"),
        "audit_report_path": resolver.metadata.get("audit_report_path"),
        "prior_source": "physical",
        "scale_prior_config": str(prior_path),
        "config_hash": sha256_file(evaluation_config_path),
        "evaluation_config": str(evaluation_config_path),
        "manifest": str(manifest_path),
        "observation_root": str(observation_root),
        "plotted_score_video_ids": plotted,
        "nan_is_zero": False,
        "empirical_pair_prior_enabled": False,
        "summary": summary,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    print(f"Saved strict baseline outputs under: {output_dir}")
    print("NaN means insufficient physical R_sd evidence; it is never converted to zero.")
    return metadata


def main() -> None:
    """Run the frozen strict physical baseline."""

    args = parse_args()
    import yaml

    with Path(args.scale_prior_config).open("r", encoding="utf-8") as file:
        prior_header = yaml.safe_load(file)
    if int(prior_header.get("metadata", {}).get("schema_version", 1)) == 2:
        from run_strict_rsd_v2 import run_v2_baseline

        run_v2_baseline(
            manifest_path=Path(args.manifest),
            observation_root=Path(args.observation_root),
            prior_path=Path(args.scale_prior_config),
            output_dir=Path(args.output_dir),
            depth_mode=args.depth_mode,
            device=args.device,
            topk=args.topk,
            overwrite=args.overwrite,
        )
        return
    run_baseline(
        manifest_path=Path(args.manifest),
        observation_root=Path(args.observation_root),
        prior_path=Path(args.scale_prior_config),
        evaluation_config_path=Path(args.evaluation_config),
        output_dir=Path(args.output_dir),
        topk=args.topk,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
