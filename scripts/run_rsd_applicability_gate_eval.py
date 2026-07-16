#!/usr/bin/env python3
"""Evaluate strict R_sd v2 before/after physical-prior applicability gates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_PYTHON = PROJECT_ROOT / ".venv/bin/python"


def _ensure_project_environment() -> None:
    if not PROJECT_PYTHON.exists():
        raise RuntimeError(f"Project environment is missing: {PROJECT_PYTHON}")
    if Path(sys.executable).resolve() != PROJECT_PYTHON.resolve():
        os.execv(str(PROJECT_PYTHON), [str(PROJECT_PYTHON), *sys.argv])


_ensure_project_environment()
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
SCRIPT_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import cv2  # noqa: E402
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
from semantic3d.keypoint_provider import (  # noqa: E402
    BaseKeypointProvider,
    KeypointPrediction,
    RealHumanKeypointProvider,
)
from semantic3d.observations import FrameObservationJSON, ObjectObservationJSON  # noqa: E402
from semantic3d.pose_applicability import (  # noqa: E402
    ApplicabilityGateResult,
    CupHeightApplicabilityGate,
    PersonHeightApplicabilityGate,
)
from semantic3d.projected_measurement import (  # noqa: E402
    ProjectedMeasurementRules,
    load_projected_measurement_rules,
)
from semantic3d.rsd_v2_error_audit import compute_depth_strategy  # noqa: E402
from run_strict_rsd_v2 import PAIR_FIELDS, load_manifest  # noqa: E402


APPLICABILITY_FIELDS = [
    "before_valid", "before_skip_reason", "before_rsd_ratio", "before_rsd_log",
    "after_valid", "after_skip_reason", "applicability_required",
    "applicability_status_a", "applicability_status_b", "applicability_score_a",
    "applicability_score_b", "applicability_passed_checks_a",
    "applicability_passed_checks_b", "applicability_failed_checks_a",
    "applicability_failed_checks_b", "pose_provider_a", "pose_provider_b",
    "keypoint_count_a", "keypoint_count_b", "valid_keypoint_count_a",
    "valid_keypoint_count_b", "keypoint_frame_index_a", "keypoint_frame_index_b",
    "person_track_id_a", "person_track_id_b", "keypoints_2d_a", "keypoints_2d_b",
    "keypoint_confidences_a", "keypoint_confidences_b", "cup_depth_iqr_a",
    "cup_depth_iqr_b", "cup_valid_depth_ratio_a", "cup_valid_depth_ratio_b",
]

VIDEO_FIELDS = [
    "video_id", "label", "label_name", "num_frames", "num_candidate_pairs",
    "before_valid_pairs", "after_valid_pairs", "before_valid_video", "after_valid_video",
    "before_valid_pair_ratio", "after_valid_pair_ratio", "person_pose_rejection_count",
    "cup_quality_rejection_count", "after_nan_count", "after_rsd_log_mean",
    "after_rsd_log_max", "status", "primary_skip_reason",
    "primary_applicability_rejection_reason",
]

COMPARISON_FIELDS = [
    "video_id", "label", "label_name", "candidate_pairs", "v2_before_valid_pairs",
    "v2_after_valid_pairs", "valid_pair_change", "v2_before_valid_video",
    "v2_after_valid_video", "person_pose_rejections", "cup_quality_rejections",
    "before_rsd_log_mean", "after_rsd_log_mean", "before_rsd_log_max",
    "after_rsd_log_max", "interpretation",
]

SKIP_FIELDS = ["video_id", "skip_reason", "count"]

SKELETON = (
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply person/cup physical-prior applicability gates to strict R_sd v2."
    )
    parser.add_argument("--input_dir", default="outputs/evaluation/rsd_strict_v2")
    parser.add_argument("--manifest", default="data/manifests/pilot_real_fake.csv")
    parser.add_argument("--scale_prior_config", default="configs/scale_priors_strict_v2.yaml")
    parser.add_argument("--pose_model_path", default="checkpoints/yolov8n-pose.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output_dir",
        default="outputs/evaluation/rsd_strict_v2_applicability_gate",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _truth(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _number(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def _finite(values: Sequence[object]) -> np.ndarray:
    return np.asarray([number for value in values if math.isfinite(number := _number(value))])


def _serialize(value: object) -> object:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, allow_nan=False)
    return "NaN" if isinstance(value, float) and math.isnan(value) else value


def _json_safe(value: object) -> object:
    """Replace non-finite diagnostic numbers with JSON null recursively."""

    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def save_csv(
    rows: Sequence[Mapping[str, object]],
    path: Path,
    fields: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(dict.fromkeys(fields)))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {field: _serialize(row.get(field, "")) for field in writer.fieldnames}
            )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _load_frames(observation_root: Path, video_id: str) -> list[FrameObservationJSON]:
    directory = observation_root / "videos" / video_id / "associated_observations"
    paths = sorted(directory.rglob("*.json"))
    if not paths:
        raise FileNotFoundError(f"Associated observations not found: {directory}")
    frames: dict[int, FrameObservationJSON] = {}
    for path in paths:
        clip = load_clip_observation(path)
        for frame in clip.frames:
            if int(frame.frame_index) in frames:
                raise ValueError(
                    f"Duplicate associated global frame {frame.frame_index} for {video_id}."
                )
            frames[int(frame.frame_index)] = frame
    return [frames[index] for index in sorted(frames)]


def _object(frame: FrameObservationJSON, object_id: str) -> ObjectObservationJSON:
    matches = [obj for obj in frame.objects if obj.object_id == object_id]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one object {object_id!r} in frame {frame.frame_index}, got {len(matches)}."
        )
    return matches[0]


def _track_key(obj: ObjectObservationJSON) -> str:
    return str(obj.track_id or f"fallback:{obj.label}:{obj.object_id}")


def _projection_histories(
    frames: Sequence[FrameObservationJSON],
) -> dict[tuple[str, str], list[float]]:
    histories: dict[tuple[str, str], list[float]] = defaultdict(list)
    for frame in frames:
        for obj in frame.objects:
            if obj.bbox is None or len(obj.bbox) != 4 or frame.height <= 0:
                continue
            height = float(obj.bbox[3]) - float(obj.bbox[1])
            if math.isfinite(height) and height > 0:
                histories[(obj.label, _track_key(obj))].append(height / frame.height)
    return histories


def _depth_map(frame: FrameObservationJSON) -> Optional[np.ndarray]:
    if not frame.depth_map_path:
        return None
    path = Path(frame.depth_map_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        return None
    depth = np.load(path)
    return np.asarray(depth) if depth.ndim == 2 else None


class ApplicabilityEvaluator:
    """Cache pose inference and applicability decisions across object pairs."""

    def __init__(self, provider: BaseKeypointProvider, keypoint_output_dir: Path) -> None:
        self.provider = provider
        self.keypoint_output_dir = keypoint_output_dir
        self.person_gate = PersonHeightApplicabilityGate()
        self.cup_gate = CupHeightApplicabilityGate()
        self.person_cache: dict[tuple[str, int, str], tuple[KeypointPrediction, ApplicabilityGateResult]] = {}
        self.cup_cache: dict[tuple[str, int, str], ApplicabilityGateResult] = {}

    def person(
        self,
        video_id: str,
        frame: FrameObservationJSON,
        obj: ObjectObservationJSON,
    ) -> tuple[KeypointPrediction, ApplicabilityGateResult]:
        key = (video_id, int(frame.frame_index), obj.object_id)
        if key in self.person_cache:
            return self.person_cache[key]
        if obj.bbox is None or not frame.image_path:
            prediction = KeypointPrediction(
                obj.label, (), "missing_image_or_bbox", "unavailable"
            )
        else:
            prediction = self.provider.predict(frame.image_path, obj.bbox, obj.label)
        result = self.person_gate.evaluate(
            obj.bbox or (),
            (frame.width, frame.height),
            prediction,
            obj.confidence,
        )
        self.person_cache[key] = (prediction, result)
        self._save_keypoints(video_id, frame, obj, prediction, result)
        return prediction, result

    def cup(
        self,
        video_id: str,
        frame: FrameObservationJSON,
        obj: ObjectObservationJSON,
        projection_history: Sequence[float],
    ) -> ApplicabilityGateResult:
        key = (video_id, int(frame.frame_index), obj.object_id)
        if key in self.cup_cache:
            return self.cup_cache[key]
        depth = _depth_map(frame)
        if depth is None:
            depth_iqr = valid_ratio = representative = math.nan
        else:
            statistic = compute_depth_strategy(depth, obj, "full_bbox_median")
            depth_iqr = statistic.depth_iqr
            valid_ratio = statistic.valid_depth_ratio
            representative = statistic.depth
        result = self.cup_gate.evaluate(
            obj.bbox or (),
            (frame.width, frame.height),
            obj.confidence,
            projection_history=projection_history,
            depth_iqr=depth_iqr,
            representative_depth=representative,
            valid_depth_ratio=valid_ratio,
        )
        self.cup_cache[key] = result
        return result

    def _save_keypoints(
        self,
        video_id: str,
        frame: FrameObservationJSON,
        obj: ObjectObservationJSON,
        prediction: KeypointPrediction,
        result: ApplicabilityGateResult,
    ) -> None:
        self.keypoint_output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "video_id": video_id,
            "object_id": obj.object_id,
            "label": obj.label,
            "keypoints_2d": [point.to_dict() for point in prediction.keypoints],
            "keypoint_confidences": [point.confidence for point in prediction.keypoints],
            "pose_provider": prediction.provider_name,
            "keypoint_frame_index": int(frame.frame_index),
            "person_track_id": _track_key(obj),
            "applicability": {
                "applicable": result.applicable,
                "applicability_score": result.applicability_score,
                "applicability_status": result.applicability_status,
                "passed_checks": list(result.passed_checks),
                "failed_checks": list(result.failed_checks),
                "diagnostic_details": dict(result.diagnostic_details),
            },
            "scope_note": "2D keypoints only; no 3D projection, R_track, or R_reproj.",
        }
        path = self.keypoint_output_dir / (
            f"{video_id}_frame_{int(frame.frame_index):06d}_{obj.object_id}.json"
        )
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )


def _application_for_object(
    evaluator: ApplicabilityEvaluator,
    video_id: str,
    frame: FrameObservationJSON,
    obj: ObjectObservationJSON,
    histories: Mapping[tuple[str, str], Sequence[float]],
) -> tuple[Optional[KeypointPrediction], Optional[ApplicabilityGateResult]]:
    if obj.label == "person":
        return evaluator.person(video_id, frame, obj)
    if obj.label == "cup":
        return None, evaluator.cup(
            video_id,
            frame,
            obj,
            histories.get((obj.label, _track_key(obj)), ()),
        )
    return None, None


def _application_fields(
    side: str,
    obj: ObjectObservationJSON,
    prediction: Optional[KeypointPrediction],
    result: Optional[ApplicabilityGateResult],
    frame_index: int,
) -> dict[str, object]:
    keypoints = list(prediction.keypoints) if prediction else []
    output: dict[str, object] = {
        f"pose_provider_{side}": prediction.provider_name if prediction else "not_required",
        f"keypoint_count_{side}": len(keypoints),
        f"valid_keypoint_count_{side}": sum(point.valid for point in keypoints),
        f"keypoint_frame_index_{side}": frame_index if prediction else "",
        f"person_track_id_{side}": _track_key(obj) if obj.label == "person" else "",
        f"keypoints_2d_{side}": [point.to_dict() for point in keypoints],
        f"keypoint_confidences_{side}": [point.confidence for point in keypoints],
    }
    if obj.label == "cup" and result is not None:
        output[f"cup_depth_iqr_{side}"] = result.diagnostic_details.get(
            "depth_iqr", math.nan
        )
        output[f"cup_valid_depth_ratio_{side}"] = result.diagnostic_details.get(
            "valid_depth_ratio", math.nan
        )
    return output


def evaluate_video(
    video_id: str,
    frames: Sequence[FrameObservationJSON],
    original_rows: Sequence[Mapping[str, str]],
    resolver: DimensionAlignedPriorResolver,
    rules: ProjectedMeasurementRules,
    evaluator: ApplicabilityEvaluator,
) -> list[dict[str, object]]:
    """Apply extra gates only to pairs that were valid in frozen strict v2."""

    frame_map = {int(frame.frame_index): frame for frame in frames}
    histories = _projection_histories(frames)
    output: list[dict[str, object]] = []
    for original in original_rows:
        row: dict[str, object] = dict(original)
        before_valid = _truth(original.get("valid", False))
        row.update(
            {
                "before_valid": before_valid,
                "before_skip_reason": original.get("skip_reason", ""),
                "before_rsd_ratio": _number(original.get("rsd_ratio")),
                "before_rsd_log": _number(original.get("rsd_log")),
            }
        )
        if not before_valid:
            row.update(
                {
                    "after_valid": False,
                    "after_skip_reason": original.get("skip_reason", ""),
                    "valid": False,
                    "rsd_ratio": math.nan,
                    "rsd_log": math.nan,
                    "applicability_required": True,
                }
            )
            output.append(row)
            continue

        frame_index = int(original["frame_index"])
        frame = frame_map[frame_index]
        obj_a = _object(frame, original["object_a_id"])
        obj_b = _object(frame, original["object_b_id"])
        prediction_a, applicability_a = _application_for_object(
            evaluator, video_id, frame, obj_a, histories
        )
        prediction_b, applicability_b = _application_for_object(
            evaluator, video_id, frame, obj_b, histories
        )
        result = compute_dimension_aligned_rsd(
            frame,
            obj_a,
            obj_b,
            resolver,
            rules,
            depth_mode="real_depth_invert",
            applicability_a=applicability_a,
            applicability_b=applicability_b,
            require_applicability=True,
        )
        result["video_id"] = video_id
        row = {
            **row,
            **result,
            **_application_fields("a", obj_a, prediction_a, applicability_a, frame_index),
            **_application_fields("b", obj_b, prediction_b, applicability_b, frame_index),
            "after_valid": bool(result["valid"]),
            "after_skip_reason": result["skip_reason"],
        }
        output.append(row)
    return output


def _stats(values: Sequence[object]) -> tuple[float, float]:
    finite = _finite(values)
    if finite.size == 0:
        return math.nan, math.nan
    return float(np.mean(finite)), float(np.max(finite))


def summarize_video(
    manifest_row: Mapping[str, object],
    frames: Sequence[FrameObservationJSON],
    rows: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], dict[str, object]]:
    candidates = [row for row in rows if row.get("skip_reason") != "insufficient_objects"]
    before = [row for row in candidates if _truth(row.get("before_valid", False))]
    after = [row for row in candidates if _truth(row.get("after_valid", False))]
    skips = Counter(
        str(row.get("after_skip_reason", ""))
        for row in candidates
        if row.get("after_skip_reason")
    )
    person_rejections = sum(
        str(row.get("after_skip_reason", "")).startswith("person_") for row in candidates
    )
    cup_rejections = sum(
        str(row.get("after_skip_reason", "")).startswith("cup_") for row in candidates
    )
    applicability_rejections = Counter(
        str(row.get("after_skip_reason", ""))
        for row in candidates
        if _truth(row.get("before_valid", False))
        and not _truth(row.get("after_valid", False))
        and row.get("after_skip_reason")
    )
    before_mean, before_max = _stats([row.get("before_rsd_log") for row in before])
    after_mean, after_max = _stats([row.get("rsd_log") for row in after])
    candidate_count = len(candidates)
    summary = {
        "video_id": manifest_row["video_id"],
        "label": manifest_row["label"],
        "label_name": manifest_row["label_name"],
        "num_frames": len(frames),
        "num_candidate_pairs": candidate_count,
        "before_valid_pairs": len(before),
        "after_valid_pairs": len(after),
        "before_valid_video": bool(before),
        "after_valid_video": bool(after),
        "before_valid_pair_ratio": len(before) / candidate_count if candidate_count else 0.0,
        "after_valid_pair_ratio": len(after) / candidate_count if candidate_count else 0.0,
        "person_pose_rejection_count": person_rejections,
        "cup_quality_rejection_count": cup_rejections,
        "after_nan_count": sum(not math.isfinite(_number(row.get("rsd_log"))) for row in candidates),
        "after_rsd_log_mean": after_mean,
        "after_rsd_log_max": after_max,
        "status": "ok" if after else "insufficient_applicable_rsd_evidence",
        "primary_skip_reason": skips.most_common(1)[0][0] if skips else "",
        "primary_applicability_rejection_reason": (
            applicability_rejections.most_common(1)[0][0]
            if applicability_rejections
            else ""
        ),
    }
    comparison = {
        "video_id": manifest_row["video_id"],
        "label": manifest_row["label"],
        "label_name": manifest_row["label_name"],
        "candidate_pairs": candidate_count,
        "v2_before_valid_pairs": len(before),
        "v2_after_valid_pairs": len(after),
        "valid_pair_change": len(after) - len(before),
        "v2_before_valid_video": bool(before),
        "v2_after_valid_video": bool(after),
        "person_pose_rejections": person_rejections,
        "cup_quality_rejections": cup_rejections,
        "before_rsd_log_mean": before_mean,
        "after_rsd_log_mean": after_mean,
        "before_rsd_log_max": before_max,
        "after_rsd_log_max": after_max,
        "interpretation": (
            "physical_evidence_rejected_not_zero_residual"
            if before and not after
            else "applicable_evidence_retained" if after else "no_v2_evidence_before_gate"
        ),
    }
    return summary, comparison


def save_pose_overlays(
    evaluator: ApplicabilityEvaluator,
    frames: Sequence[FrameObservationJSON],
    output_dir: Path,
    video_id: str = "real_3",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_map = {int(frame.frame_index): frame for frame in frames}
    for (cached_video, frame_index, object_id), (prediction, gate) in evaluator.person_cache.items():
        if cached_video != video_id:
            continue
        frame = frame_map[frame_index]
        obj = _object(frame, object_id)
        image = cv2.imread(str(frame.image_path)) if frame.image_path else None
        if image is None:
            image = np.full((frame.height, frame.width, 3), 245, np.uint8)
        if obj.bbox:
            x1, y1, x2, y2 = (int(round(value)) for value in obj.bbox)
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 180, 0), 3)
        points = {point.keypoint_name: point for point in prediction.keypoints}
        for start, end in SKELETON:
            if start in points and end in points and points[start].valid and points[end].valid:
                p1 = (int(points[start].x), int(points[start].y))
                p2 = (int(points[end].x), int(points[end].y))
                cv2.line(image, p1, p2, (20, 210, 90), 2)
        for point in prediction.keypoints:
            color = (20, 210, 90) if point.valid else (90, 90, 220)
            center = (int(point.x), int(point.y))
            if 0 <= center[0] < frame.width and 0 <= center[1] < frame.height:
                cv2.circle(image, center, 5, color, -1)
                cv2.putText(
                    image,
                    f"{point.keypoint_name.replace('left_', 'L_').replace('right_', 'R_')}:{point.confidence:.2f}",
                    (center[0] + 5, center[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.36,
                    color,
                    1,
                    cv2.LINE_AA,
                )
        details = gate.diagnostic_details
        lines = [
            f"track={_track_key(obj)} frame={frame_index}",
            f"status={gate.applicability_status}",
            f"heuristic_score={gate.applicability_score:.3f} (not probability)",
            f"torso_angle={details.get('torso_tilt_degrees', 'N/A')}",
            f"failed={','.join(gate.failed_checks) or 'none'}",
        ]
        panel_height = 28 * len(lines) + 14
        overlay = image.copy()
        cv2.rectangle(overlay, (8, 8), (frame.width - 8, panel_height), (255, 255, 255), -1)
        image = cv2.addWeighted(overlay, 0.80, image, 0.20, 0)
        for index, line in enumerate(lines):
            cv2.putText(
                image,
                line,
                (18, 34 + 26 * index),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (25, 25, 25),
                2,
                cv2.LINE_AA,
            )
        cv2.imwrite(str(output_dir / f"{video_id}_frame_{frame_index:06d}_pose_gate.png"), image)


def save_summary_plots(
    video_rows: Sequence[Mapping[str, object]],
    pair_rows: Sequence[Mapping[str, object]],
    evaluator: ApplicabilityEvaluator,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    real3 = sorted(
        (
            (frame_index, result)
            for (video_id, frame_index, _), (_, result) in evaluator.person_cache.items()
            if video_id == "real_3"
        ),
        key=lambda item: item[0],
    )
    fig, ax = plt.subplots(figsize=(11, 5.5))
    if real3:
        statuses = list(dict.fromkeys(result.applicability_status for _, result in real3))
        status_index = {status: index for index, status in enumerate(statuses)}
        ax.scatter(
            [frame for frame, _ in real3],
            [status_index[result.applicability_status] for _, result in real3],
            c=["#2A9D8F" if result.applicable else "#D1495B" for _, result in real3],
            s=55,
        )
        ax.set_yticks(range(len(statuses)), statuses)
    else:
        ax.text(0.5, 0.5, "No real_3 person keypoints", transform=ax.transAxes, ha="center")
    ax.set(title="real_3 Person Height Applicability by Frame", xlabel="global frame")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "applicability_status_by_frame.png", dpi=180)
    plt.close(fig)

    x = np.arange(len(video_rows))
    before = np.asarray([float(row["before_valid_pairs"]) for row in video_rows])
    after = np.asarray([float(row["after_valid_pairs"]) for row in video_rows])
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(x - 0.2, before, 0.4, label="strict v2 before gate", color="#4C78A8")
    ax.bar(x + 0.2, after, 0.4, label="after applicability gate", color="#E45756")
    ax.set_xticks(x, [str(row["video_id"]) for row in video_rows], rotation=30, ha="right")
    ax.set_ylabel("valid pair count")
    ax.set_title("Strict v2 Pair Coverage Before/After Applicability Gate")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "v2_pair_coverage_before_after_gate.png", dpi=180)
    plt.close(fig)

    reasons = Counter(
        str(row.get("after_skip_reason", ""))
        for row in pair_rows
        if row.get("after_skip_reason")
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    labels, counts = zip(*reasons.most_common()) if reasons else (("none",), (0,))
    ax.bar(np.arange(len(labels)), counts, color="#8C6BB1")
    ax.set_xticks(np.arange(len(labels)), labels, rotation=50, ha="right")
    ax.set_ylabel("pair count")
    ax.set_title("Applicability Evaluation Skip Reasons")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "applicability_skip_reasons.png", dpi=180)
    plt.close(fig)


def run_evaluation(
    input_dir: Path,
    manifest_path: Path,
    prior_path: Path,
    pose_model_path: Path,
    output_dir: Path,
    device: str = "cpu",
) -> dict[str, object]:
    """Run the complete read-only applicability-gated six-video evaluation."""

    v1_path = PROJECT_ROOT / "configs/scale_priors_strict_v1.yaml"
    v2_path = PROJECT_ROOT / "configs/scale_priors_strict_v2.yaml"
    before_hashes = {"v1": sha256_file(v1_path), "v2": sha256_file(v2_path)}
    metadata = json.loads((input_dir / "run_metadata.json").read_text(encoding="utf-8"))
    observation_root = Path(str(metadata["observation_root"]))
    if not observation_root.is_absolute():
        observation_root = PROJECT_ROOT / observation_root
    resolver = load_dimension_aligned_prior_resolver(prior_path)
    rules_path = Path(str(resolver.metadata["projected_measurement_rules"]))
    if not rules_path.is_absolute():
        rules_path = PROJECT_ROOT / rules_path
    rules = load_projected_measurement_rules(rules_path)
    manifest = load_manifest(manifest_path)
    original_rows = _read_csv(input_dir / "per_pair_rsd_details.csv")
    provider = RealHumanKeypointProvider(pose_model_path, device=device)
    evaluator = ApplicabilityEvaluator(provider, output_dir / "keypoints")

    all_rows: list[dict[str, object]] = []
    video_rows: list[dict[str, object]] = []
    comparison_rows: list[dict[str, object]] = []
    frames_by_video: dict[str, list[FrameObservationJSON]] = {}
    for manifest_row in manifest:
        video_id = str(manifest_row["video_id"])
        frames = _load_frames(observation_root, video_id)
        frames_by_video[video_id] = frames
        video_original = [row for row in original_rows if row["video_id"] == video_id]
        rows = evaluate_video(
            video_id, frames, video_original, resolver, rules, evaluator
        )
        summary, comparison = summarize_video(manifest_row, frames, rows)
        all_rows.extend(rows)
        video_rows.append(summary)
        comparison_rows.append(comparison)

    output_dir.mkdir(parents=True, exist_ok=True)
    pair_fields = list(dict.fromkeys([*PAIR_FIELDS, *APPLICABILITY_FIELDS]))
    save_csv(all_rows, output_dir / "per_pair_applicability_details.csv", pair_fields)
    save_csv(video_rows, output_dir / "per_video_applicability_summary.csv", VIDEO_FIELDS)
    save_csv(
        comparison_rows,
        output_dir / "v2_before_after_gate_comparison.csv",
        COMPARISON_FIELDS,
    )
    skip_rows: list[dict[str, object]] = []
    for video_id in [str(row["video_id"]) for row in manifest]:
        counts = Counter(
            str(row.get("after_skip_reason", ""))
            for row in all_rows
            if row.get("video_id") == video_id and row.get("after_skip_reason")
        )
        skip_rows.extend(
            {"video_id": video_id, "skip_reason": reason, "count": count}
            for reason, count in sorted(counts.items())
        )
    save_csv(
        skip_rows,
        output_dir / "applicability_skip_reason_summary.csv",
        SKIP_FIELDS,
    )
    if "real_3" in frames_by_video:
        save_pose_overlays(
            evaluator,
            frames_by_video["real_3"],
            output_dir / "visualizations/real_3_pose_gate_frames",
        )
    save_summary_plots(video_rows, all_rows, evaluator, output_dir / "visualizations")

    real3 = next((row for row in video_rows if row["video_id"] == "real_3"), None)
    real3_person_statuses = Counter(
        result.applicability_status
        for (video_id, _, _), (_, result) in evaluator.person_cache.items()
        if video_id == "real_3"
    )
    real3_cup_statuses = Counter(
        result.applicability_status
        for (video_id, _, _), result in evaluator.cup_cache.items()
        if video_id == "real_3"
    )
    after_hashes = {"v1": sha256_file(v1_path), "v2": sha256_file(v2_path)}
    if before_hashes != after_hashes:
        raise AssertionError("Frozen strict v1/v2 prior files changed during gate evaluation.")
    summary = {
        "total_candidate_pairs": sum(int(row["num_candidate_pairs"]) for row in video_rows),
        "before_valid_pairs": sum(int(row["before_valid_pairs"]) for row in video_rows),
        "after_valid_pairs": sum(int(row["after_valid_pairs"]) for row in video_rows),
        "before_valid_videos": sum(bool(row["before_valid_video"]) for row in video_rows),
        "after_valid_videos": sum(bool(row["after_valid_video"]) for row in video_rows),
        "person_pose_rejections": sum(int(row["person_pose_rejection_count"]) for row in video_rows),
        "cup_quality_rejections": sum(int(row["cup_quality_rejection_count"]) for row in video_rows),
        "nan_count": sum(int(row["after_nan_count"]) for row in video_rows),
        "real_3": real3,
        "real_3_person_status_counts": dict(real3_person_statuses),
        "real_3_cup_status_counts": dict(real3_cup_statuses),
        "pose_provider": provider.provider_name,
        "pose_model_path": str(pose_model_path),
        "pose_model_hash": sha256_file(pose_model_path),
        "strict_v1_hash": before_hashes["v1"],
        "strict_v2_hash": before_hashes["v2"],
        "score_semantics": "heuristic_quality_not_probability",
        "note": "Gate rejection means insufficient physical evidence; rejected R_sd is NaN, not zero.",
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(_json_safe(summary), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print("R_sd physical-prior applicability gate evaluation:")
    for key in (
        "total_candidate_pairs", "before_valid_pairs", "after_valid_pairs",
        "before_valid_videos", "after_valid_videos", "person_pose_rejections",
        "cup_quality_rejections", "nan_count",
    ):
        print(f"  {key}: {summary[key]}")
    if real3:
        print(f"  real_3 before/after: {real3['before_valid_pairs']}/{real3['after_valid_pairs']}")
        print(f"  real_3 primary skip: {real3['primary_skip_reason']}")
        print(
            "  real_3 applicability rejection: "
            f"{real3['primary_applicability_rejection_reason']}"
        )
    return summary


def main() -> None:
    args = parse_args()
    run_evaluation(
        Path(args.input_dir),
        Path(args.manifest),
        Path(args.scale_prior_config),
        Path(args.pose_model_path),
        Path(args.output_dir),
        args.device,
    )


if __name__ == "__main__":
    main()
