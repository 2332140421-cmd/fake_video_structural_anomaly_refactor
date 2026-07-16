#!/usr/bin/env python3
"""Attribute persistent strict v2 person-cup residual errors without refitting priors."""

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
from typing import Iterable, Mapping, Optional, Sequence


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

import cv2  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from semantic3d.io import load_clip_observation  # noqa: E402
from semantic3d.observations import FrameObservationJSON, ObjectObservationJSON  # noqa: E402
from semantic3d.rsd_v2_error_audit import (  # noqa: E402
    DEPTH_STRATEGIES,
    boundary_contacts,
    coefficient_of_variation,
    compute_depth_strategy,
    deterministic_track_id,
    diagnostic_labels,
    make_frame_pair_id,
    make_track_pair_id,
    recompute_scale_depth_formula,
    swapped_log_residual_consistent,
)


PAIR_AUDIT_FIELDS = [
    "video_id", "global_frame_index", "object_a_track_id", "object_b_track_id", "label_pair",
    "track_pair_id", "frame_pair_id", "track_id_fallback_used", "frame_width", "frame_height",
    "person_track_id", "cup_track_id", "person_bbox_x1", "person_bbox_y1", "person_bbox_x2",
    "person_bbox_y2", "cup_bbox_x1", "cup_bbox_y1", "cup_bbox_x2", "cup_bbox_y2",
    "person_detection_confidence", "cup_detection_confidence", "person_bbox_width", "person_bbox_height",
    "cup_bbox_width", "cup_bbox_height", "person_bbox_height_norm", "cup_bbox_height_norm",
    "person_aspect_ratio", "cup_aspect_ratio", "person_projected_measurement", "cup_projected_measurement",
    "person_measurement_quality", "cup_measurement_quality", "person_current_depth", "cup_current_depth",
    "observed_depth_ratio", "depth_direction_convention", "depth_mode", "person_characteristic_dimension",
    "cup_characteristic_dimension", "person_prior_min", "person_prior_max", "cup_prior_min", "cup_prior_max",
    "person_reliability_status", "cup_reliability_status", "evidence_tier", "expected_ratio_low",
    "expected_ratio_high", "observed_log_ratio", "expected_log_low", "expected_log_high",
    "distance_below_interval", "distance_above_interval", "rsd_ratio", "rsd_log", "person_gate_passed",
    "cup_gate_passed", "person_gate_score", "cup_gate_score", "person_gate_reasons", "cup_gate_reasons",
    "person_failed_gate_reasons", "cup_failed_gate_reasons", "person_touches_top", "person_touches_bottom",
    "person_touches_left", "person_touches_right", "cup_touches_top", "cup_touches_bottom",
    "cup_touches_left", "cup_touches_right", "person_bbox_area_ratio", "cup_bbox_area_ratio",
    "person_height_temporal_cv", "cup_height_temporal_cv", "person_aspect_ratio_temporal_cv",
    "cup_aspect_ratio_temporal_cv", "depth_ratio_temporal_cv", "person_full_depth_iqr",
    "cup_full_depth_iqr", "diagnostic_labels",
]

DEPTH_COMPARISON_FIELDS = [
    "video_id", "track_pair_id", "frame_index", "depth_strategy", "person_depth", "cup_depth",
    "person_valid_depth_ratio", "cup_valid_depth_ratio", "person_depth_iqr", "cup_depth_iqr",
    "person_method_detail", "cup_method_detail", "observed_depth_ratio", "expected_ratio_low",
    "expected_ratio_high", "rsd_log",
]

DEPTH_SUMMARY_FIELDS = [
    "depth_strategy", "valid_frames", "mean_rsd_log", "median_rsd_log", "max_rsd_log", "p95_rsd_log",
    "temporal_cv", "mean_valid_depth_ratio",
]

TRACK_FIELDS = [
    "video_id", "track_pair_id", "label_pair", "num_frames", "start_frame", "end_frame", "mean_rsd_log",
    "median_rsd_log", "max_rsd_log", "p95_rsd_log", "persistent_high_residual_ratio",
    "person_projected_height_cv", "cup_projected_height_cv", "depth_ratio_cv", "person_gate_pass_ratio",
    "cup_gate_pass_ratio", "primary_diagnostic_reason",
]

PROJECTION_FIELDS = [
    "video_id", "frame_index", "track_pair_id", "projection_strategy", "diagnostic_only",
    "person_projected_measurement", "cup_projected_measurement", "expected_ratio_low", "expected_ratio_high",
    "observed_depth_ratio", "rsd_log",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit strict v2 person-cup high residuals.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--video_dir", required=True)
    parser.add_argument("--video_id", default="real_3")
    parser.add_argument("--label_a", default="person")
    parser.add_argument("--label_b", default="cup")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--debug_threshold", type=float, default=0.1)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _serialize(value: object) -> object:
    return "NaN" if isinstance(value, float) and math.isnan(value) else value


def save_csv(rows: Sequence[Mapping[str, object]], path: Path, fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _serialize(row.get(field, "")) for field in fields})


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


def _load_frames(observation_root: Path, video_id: str) -> tuple[dict[int, FrameObservationJSON], Path]:
    directory = observation_root / "videos" / video_id / "associated_observations"
    paths = sorted(directory.rglob("*.json"))
    if not paths:
        raise FileNotFoundError(f"Associated observation JSON not found: {directory}")
    clip = load_clip_observation(paths[0])
    frames = {int(frame.frame_index): frame for frame in clip.frames}
    if len(frames) != len(clip.frames):
        raise ValueError("Associated observations contain duplicate global frame indices.")
    return frames, observation_root / "videos" / video_id


def _object(frame: FrameObservationJSON, object_id: str) -> ObjectObservationJSON:
    matches = [obj for obj in frame.objects if obj.object_id == object_id]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one object {object_id!r} in frame {frame.frame_index}.")
    return matches[0]


def _bbox_values(obj: ObjectObservationJSON) -> tuple[float, float, float, float]:
    if obj.bbox is None or len(obj.bbox) != 4:
        return math.nan, math.nan, math.nan, math.nan
    return tuple(float(value) for value in obj.bbox)  # type: ignore[return-value]


def _resolve_paths(input_dir: Path) -> tuple[Path, Path]:
    metadata = json.loads((input_dir / "run_metadata.json").read_text(encoding="utf-8"))
    observation_root = Path(str(metadata["observation_root"]))
    if not observation_root.is_absolute():
        observation_root = PROJECT_ROOT / observation_root
    return observation_root, input_dir / "per_pair_rsd_details.csv"


def _valid_pair_rows(pair_path: Path, video_id: str, label_a: str, label_b: str) -> list[dict[str, str]]:
    wanted = {label_a, label_b}
    rows = [
        row for row in _read_csv(pair_path)
        if row["video_id"] == video_id
        and row["valid"].lower() == "true"
        and {row["object_a_label"], row["object_b_label"]} == wanted
    ]
    if not rows:
        raise ValueError(f"No valid {label_a}-{label_b} strict v2 pairs for {video_id}.")
    return rows


def _ordered_objects(
    frame: FrameObservationJSON, pair: Mapping[str, str], person_label: str, cup_label: str
) -> tuple[ObjectObservationJSON, ObjectObservationJSON, bool]:
    a = _object(frame, pair["object_a_id"])
    b = _object(frame, pair["object_b_id"])
    if a.label == person_label and b.label == cup_label:
        return a, b, False
    if b.label == person_label and a.label == cup_label:
        return b, a, True
    raise ValueError(f"Unexpected labels in frame {frame.frame_index}: {a.label}, {b.label}")


def _formal_values(pair: Mapping[str, str], swapped: bool) -> dict[str, float]:
    suffix_person, suffix_cup = ("b", "a") if swapped else ("a", "b")
    return {
        "person_projected": float(pair[f"projected_measurement_{suffix_person}"]),
        "cup_projected": float(pair[f"projected_measurement_{suffix_cup}"]),
        "person_prior_min": float(pair[f"object_{suffix_person}_prior_low"]),
        "person_prior_max": float(pair[f"object_{suffix_person}_prior_high"]),
        "cup_prior_min": float(pair[f"object_{suffix_cup}_prior_low"]),
        "cup_prior_max": float(pair[f"object_{suffix_cup}_prior_high"]),
    }


def _mask_for(obj: ObjectObservationJSON, shape: tuple[int, int]) -> Optional[np.ndarray]:
    if not obj.mask_path:
        return None
    path = Path(obj.mask_path)
    if not path.exists():
        return None
    if path.suffix == ".npy":
        mask = np.load(path)
    else:
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return None if mask is None or mask.shape[:2] != shape else np.asarray(mask) > 0


def _projection_value(obj: ObjectObservationJSON, frame: FrameObservationJSON, strategy: str) -> float:
    x1, y1, x2, y2 = _bbox_values(obj)
    width, height = x2 - x1, y2 - y1
    if not all(math.isfinite(value) and value > 0 for value in (width, height, frame.width, frame.height)):
        return math.nan
    if strategy == "bbox_height_norm":
        return height / frame.height
    if strategy == "sqrt_bbox_area_norm":
        return math.sqrt(width * height / (frame.width * frame.height))
    if strategy == "bbox_diagonal_norm":
        return math.hypot(width / frame.width, height / frame.height)
    raise ValueError(strategy)


def _repeated_clip_count(
    video_root: Path,
    unique_frame_ids: set[str],
    label_a: str,
    label_b: str,
) -> int:
    """Count duplicate raw-clip occurrences of globally unique frame pairs."""

    associated = load_clip_observation(next((video_root / "associated_observations").rglob("*.json")))
    track_map = {
        (int(frame.frame_index), obj.object_id): deterministic_track_id(obj)[0]
        for frame in associated.frames for obj in frame.objects
    }
    occurrences: Counter[str] = Counter()
    for path in sorted((video_root / "observations").glob("*.json")):
        clip = load_clip_observation(path)
        for frame in clip.frames:
            objects_a = [obj for obj in frame.objects if obj.label == label_a]
            objects_b = [obj for obj in frame.objects if obj.label == label_b]
            for obj_a in objects_a:
                for obj_b in objects_b:
                    track_a = track_map.get(
                        (int(frame.frame_index), obj_a.object_id),
                        f"fallback:{label_a}:{obj_a.object_id}",
                    )
                    track_b = track_map.get(
                        (int(frame.frame_index), obj_b.object_id),
                        f"fallback:{label_b}:{obj_b.object_id}",
                    )
                    frame_id = make_frame_pair_id(
                        clip.video_id, int(frame.frame_index), track_a, track_b
                    )
                    if frame_id in unique_frame_ids:
                        occurrences[frame_id] += 1
    return sum(max(0, count - 1) for count in occurrences.values())


def build_audit_data(
    input_dir: Path,
    video_id: str,
    label_a: str,
    label_b: str,
) -> tuple[
    list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], dict[str, int], Path
]:
    """Build formula, depth, and projection audit rows without writing inputs."""

    observation_root, pair_path = _resolve_paths(input_dir)
    frames, video_root = _load_frames(observation_root, video_id)
    pairs = _valid_pair_rows(pair_path, video_id, label_a, label_b)
    preliminary: list[dict[str, object]] = []
    depth_rows: list[dict[str, object]] = []
    projection_rows: list[dict[str, object]] = []
    formula_errors = interval_errors = swap_errors = depth_mode_errors = 0
    seen_frame_pair_ids: set[str] = set()
    input_duplicate_count = 0

    for pair in pairs:
        frame_index = int(pair["frame_index"])
        frame = frames[frame_index]
        person, cup, swapped = _ordered_objects(frame, pair, label_a, label_b)
        person_track, person_fallback = deterministic_track_id(person)
        cup_track, cup_fallback = deterministic_track_id(cup)
        track_pair_id = make_track_pair_id(video_id, person_track, cup_track)
        frame_pair_id = make_frame_pair_id(video_id, frame_index, person_track, cup_track)
        if frame_pair_id in seen_frame_pair_ids:
            input_duplicate_count += 1
            continue
        seen_frame_pair_ids.add(frame_pair_id)
        formal = _formal_values(pair, swapped)
        formula = recompute_scale_depth_formula(
            person.depth, cup.depth, formal["person_prior_min"], formal["person_prior_max"],
            formal["cup_prior_min"], formal["cup_prior_max"], formal["person_projected"], formal["cup_projected"],
        )
        if formula["expected_ratio_low"] > formula["expected_ratio_high"]:
            interval_errors += 1
        ratio_matches = math.isclose(
            formula["rsd_ratio"], float(pair["rsd_ratio"]), rel_tol=1e-8, abs_tol=1e-8
        )
        log_matches = math.isclose(
            formula["rsd_log"], float(pair["rsd_log"]), rel_tol=1e-8, abs_tol=1e-8
        )
        if not (ratio_matches and log_matches):
            formula_errors += 1
        swap_arguments = {
            "depth_a": person.depth, "depth_b": cup.depth,
            "prior_a_min": formal["person_prior_min"], "prior_a_max": formal["person_prior_max"],
            "prior_b_min": formal["cup_prior_min"], "prior_b_max": formal["cup_prior_max"],
            "projected_a": formal["person_projected"], "projected_b": formal["cup_projected"],
        }
        if not swapped_log_residual_consistent(swap_arguments):
            swap_errors += 1
        if pair.get("depth_mode") != "real_depth_invert":
            depth_mode_errors += 1

        px1, py1, px2, py2 = _bbox_values(person)
        cx1, cy1, cx2, cy2 = _bbox_values(cup)
        person_contact = boundary_contacts(person.bbox, frame.width, frame.height)
        cup_contact = boundary_contacts(cup.bbox, frame.width, frame.height)
        depth_path = Path(str(frame.depth_map_path))
        if not depth_path.is_absolute():
            depth_path = PROJECT_ROOT / depth_path
        depth_map = np.load(depth_path)
        person_full = compute_depth_strategy(depth_map, person, "full_bbox_median")
        cup_full = compute_depth_strategy(depth_map, cup, "full_bbox_median")
        row: dict[str, object] = {
            "video_id": video_id, "global_frame_index": frame_index,
            "object_a_track_id": deterministic_track_id(_object(frame, pair["object_a_id"]))[0],
            "object_b_track_id": deterministic_track_id(_object(frame, pair["object_b_id"]))[0],
            "label_pair": f"{label_a}-{label_b}", "track_pair_id": track_pair_id,
            "frame_pair_id": frame_pair_id, "track_id_fallback_used": person_fallback or cup_fallback,
            "person_track_id": person_track, "cup_track_id": cup_track, "frame_width": frame.width,
            "frame_height": frame.height, "person_bbox_x1": px1, "person_bbox_y1": py1,
            "person_bbox_x2": px2, "person_bbox_y2": py2, "cup_bbox_x1": cx1, "cup_bbox_y1": cy1,
            "cup_bbox_x2": cx2, "cup_bbox_y2": cy2, "person_detection_confidence": person.confidence,
            "cup_detection_confidence": cup.confidence, "person_bbox_width": px2 - px1,
            "person_bbox_height": py2 - py1, "cup_bbox_width": cx2 - cx1, "cup_bbox_height": cy2 - cy1,
            "person_bbox_height_norm": (py2 - py1) / frame.height, "cup_bbox_height_norm": (cy2 - cy1) / frame.height,
            "person_aspect_ratio": (px2 - px1) / (py2 - py1), "cup_aspect_ratio": (cx2 - cx1) / (cy2 - cy1),
            "person_projected_measurement": formal["person_projected"],
            "cup_projected_measurement": formal["cup_projected"],
            "person_measurement_quality": pair["measurement_quality_b" if swapped else "measurement_quality_a"],
            "cup_measurement_quality": pair["measurement_quality_a" if swapped else "measurement_quality_b"],
            "person_current_depth": person.depth, "cup_current_depth": cup.depth,
            "depth_direction_convention": "larger_is_farther", "depth_mode": pair.get("depth_mode", ""),
            "person_characteristic_dimension": pair["characteristic_dimension_b" if swapped else "characteristic_dimension_a"],
            "cup_characteristic_dimension": pair["characteristic_dimension_a" if swapped else "characteristic_dimension_b"],
            "person_prior_min": formal["person_prior_min"], "person_prior_max": formal["person_prior_max"],
            "cup_prior_min": formal["cup_prior_min"], "cup_prior_max": formal["cup_prior_max"],
            "person_reliability_status": pair["object_b_prior_status" if swapped else "object_a_prior_status"],
            "cup_reliability_status": pair["object_a_prior_status" if swapped else "object_b_prior_status"],
            "evidence_tier": pair["evidence_tier"],
            "person_gate_passed": pair["gate_passed_b" if swapped else "gate_passed_a"],
            "cup_gate_passed": pair["gate_passed_a" if swapped else "gate_passed_b"],
            "person_gate_score": float(pair["gate_score_b" if swapped else "gate_score_a"]),
            "cup_gate_score": float(pair["gate_score_a" if swapped else "gate_score_b"]),
            "person_gate_reasons": pair["gate_reasons_b" if swapped else "gate_reasons_a"],
            "cup_gate_reasons": pair["gate_reasons_a" if swapped else "gate_reasons_b"],
            "person_failed_gate_reasons": pair["failed_gate_reasons_b" if swapped else "failed_gate_reasons_a"],
            "cup_failed_gate_reasons": pair["failed_gate_reasons_a" if swapped else "failed_gate_reasons_b"],
            **{f"person_touches_{side}": value for side, value in person_contact.items()},
            **{f"cup_touches_{side}": value for side, value in cup_contact.items()},
            "person_bbox_area_ratio": (px2 - px1) * (py2 - py1) / (frame.width * frame.height),
            "cup_bbox_area_ratio": (cx2 - cx1) * (cy2 - cy1) / (frame.width * frame.height),
            "person_full_depth_iqr": person_full.depth_iqr, "cup_full_depth_iqr": cup_full.depth_iqr,
            **formula,
        }
        preliminary.append(row)

        person_mask = _mask_for(person, depth_map.shape)
        cup_mask = _mask_for(cup, depth_map.shape)
        for strategy in DEPTH_STRATEGIES:
            person_stat = compute_depth_strategy(depth_map, person, strategy, person_mask)
            cup_stat = compute_depth_strategy(depth_map, cup, strategy, cup_mask)
            result = recompute_scale_depth_formula(
                person_stat.depth, cup_stat.depth, formal["person_prior_min"], formal["person_prior_max"],
                formal["cup_prior_min"], formal["cup_prior_max"], formal["person_projected"], formal["cup_projected"],
            )
            depth_rows.append(
                {
                    "video_id": video_id, "track_pair_id": track_pair_id, "frame_index": frame_index,
                    "depth_strategy": strategy, "person_depth": person_stat.depth, "cup_depth": cup_stat.depth,
                    "person_valid_depth_ratio": person_stat.valid_depth_ratio,
                    "cup_valid_depth_ratio": cup_stat.valid_depth_ratio, "person_depth_iqr": person_stat.depth_iqr,
                    "cup_depth_iqr": cup_stat.depth_iqr, "person_method_detail": person_stat.method_detail,
                    "cup_method_detail": cup_stat.method_detail, **{
                        key: result[key] for key in ("observed_depth_ratio", "expected_ratio_low", "expected_ratio_high", "rsd_log")
                    },
                }
            )
        for strategy in ("bbox_height_norm", "sqrt_bbox_area_norm", "bbox_diagonal_norm"):
            p_person = _projection_value(person, frame, strategy)
            p_cup = _projection_value(cup, frame, strategy)
            result = recompute_scale_depth_formula(
                person.depth, cup.depth, formal["person_prior_min"], formal["person_prior_max"],
                formal["cup_prior_min"], formal["cup_prior_max"], p_person, p_cup,
            )
            projection_rows.append(
                {
                    "video_id": video_id, "frame_index": frame_index, "track_pair_id": track_pair_id,
                    "projection_strategy": strategy, "diagnostic_only": True,
                    "person_projected_measurement": p_person, "cup_projected_measurement": p_cup,
                    "expected_ratio_low": result["expected_ratio_low"], "expected_ratio_high": result["expected_ratio_high"],
                    "observed_depth_ratio": result["observed_depth_ratio"], "rsd_log": result["rsd_log"],
                }
            )

    person_height_cv = coefficient_of_variation(row["person_bbox_height_norm"] for row in preliminary)
    cup_height_cv = coefficient_of_variation(row["cup_bbox_height_norm"] for row in preliminary)
    person_aspect_cv = coefficient_of_variation(row["person_aspect_ratio"] for row in preliminary)
    cup_aspect_cv = coefficient_of_variation(row["cup_aspect_ratio"] for row in preliminary)
    depth_ratio_cv = coefficient_of_variation(row["observed_depth_ratio"] for row in preliminary)
    for row in preliminary:
        row.update(
            {
                "person_height_temporal_cv": person_height_cv, "cup_height_temporal_cv": cup_height_cv,
                "person_aspect_ratio_temporal_cv": person_aspect_cv,
                "cup_aspect_ratio_temporal_cv": cup_aspect_cv, "depth_ratio_temporal_cv": depth_ratio_cv,
            }
        )
        row["diagnostic_labels"] = "|".join(diagnostic_labels(row))

    frame_ids = [str(row["frame_pair_id"]) for row in preliminary]
    preliminary.sort(key=lambda row: int(row["global_frame_index"]))
    final_duplicate_count = len(frame_ids) - len(set(frame_ids))
    stats = {
        "formula_error_count": formula_errors,
        "interval_order_error_count": interval_errors,
        "swapped_pair_inconsistency_count": swap_errors,
        "depth_mode_error_count": depth_mode_errors,
        "input_duplicate_frame_pair_count": input_duplicate_count,
        "duplicate_frame_pair_count": final_duplicate_count,
        "repeated_clip_pair_count": _repeated_clip_count(
            video_root, set(frame_ids), label_a, label_b
        ),
    }
    return preliminary, depth_rows, projection_rows, stats, video_root


def depth_strategy_summary(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for strategy in DEPTH_STRATEGIES:
        group = [row for row in rows if row["depth_strategy"] == strategy]
        residuals = _finite(row["rsd_log"] for row in group)
        ratios = _finite(row["observed_depth_ratio"] for row in group)
        valid_ratios = _finite(
            (float(row["person_valid_depth_ratio"]) + float(row["cup_valid_depth_ratio"])) / 2 for row in group
        )
        output.append(
            {
                "depth_strategy": strategy, "valid_frames": int(residuals.size),
                "mean_rsd_log": float(np.mean(residuals)) if residuals.size else math.nan,
                "median_rsd_log": float(np.median(residuals)) if residuals.size else math.nan,
                "max_rsd_log": float(np.max(residuals)) if residuals.size else math.nan,
                "p95_rsd_log": float(np.percentile(residuals, 95)) if residuals.size else math.nan,
                "temporal_cv": coefficient_of_variation(ratios),
                "mean_valid_depth_ratio": float(np.mean(valid_ratios)) if valid_ratios.size else math.nan,
            }
        )
    return output


def per_track_pair(rows: Sequence[Mapping[str, object]], debug_threshold: float) -> list[dict[str, object]]:
    groups: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(row["track_pair_id"])].append(row)
    output: list[dict[str, object]] = []
    for track_pair_id, group in sorted(groups.items()):
        residuals = _finite(row["rsd_log"] for row in group)
        reasons = Counter(
            reason for row in group for reason in str(row["diagnostic_labels"]).split("|") if reason
        )
        output.append(
            {
                "video_id": group[0]["video_id"], "track_pair_id": track_pair_id,
                "label_pair": group[0]["label_pair"], "num_frames": len(group),
                "start_frame": min(int(row["global_frame_index"]) for row in group),
                "end_frame": max(int(row["global_frame_index"]) for row in group),
                "mean_rsd_log": float(np.mean(residuals)), "median_rsd_log": float(np.median(residuals)),
                "max_rsd_log": float(np.max(residuals)), "p95_rsd_log": float(np.percentile(residuals, 95)),
                "persistent_high_residual_ratio": float(np.mean(residuals > debug_threshold)),
                "person_projected_height_cv": coefficient_of_variation(row["person_projected_measurement"] for row in group),
                "cup_projected_height_cv": coefficient_of_variation(row["cup_projected_measurement"] for row in group),
                "depth_ratio_cv": coefficient_of_variation(row["observed_depth_ratio"] for row in group),
                "person_gate_pass_ratio": float(np.mean([str(row["person_gate_passed"]).lower() == "true" for row in group])),
                "cup_gate_pass_ratio": float(np.mean([str(row["cup_gate_passed"]).lower() == "true" for row in group])),
                "primary_diagnostic_reason": reasons.most_common(1)[0][0] if reasons else "no_obvious_issue",
            }
        )
    return output


def _read_image(frame: FrameObservationJSON, video_path: Path) -> Optional[np.ndarray]:
    if frame.image_path and Path(frame.image_path).exists():
        return cv2.imread(frame.image_path)
    capture = cv2.VideoCapture(str(video_path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame.frame_index))
    ok, image = capture.read()
    capture.release()
    return image if ok else None


def save_frame_plots(
    rows: Sequence[Mapping[str, object]],
    depth_rows: Sequence[Mapping[str, object]],
    frames: Mapping[int, FrameObservationJSON],
    video_path: Path,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    by_frame_strategy = {(int(row["frame_index"]), str(row["depth_strategy"])): row for row in depth_rows}
    for row in rows:
        index = int(row["global_frame_index"])
        image = _read_image(frames[index], video_path)
        if image is None:
            image = np.full((int(row["frame_height"]), int(row["frame_width"]), 3), 245, np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        fig, ax = plt.subplots(figsize=(11, 8))
        ax.imshow(image)
        for prefix, color in (("person", "#00A6D6"), ("cup", "#E45756")):
            x1, y1, x2, y2 = (float(row[f"{prefix}_bbox_{name}"]) for name in ("x1", "y1", "x2", "y2"))
            ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color=color, linewidth=2))
            ax.text(x1, max(4, y1 - 5), f"{prefix} {row[f'{prefix}_track_id']} conf={float(row[f'{prefix}_detection_confidence']):.2f}", color=color, fontsize=8, bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"})
        center50 = by_frame_strategy[(index, "center_50_bbox_median")]
        cluster = by_frame_strategy[(index, "foreground_depth_cluster")]
        text = (
            f"frame={index}  formal R_sd_log={float(row['rsd_log']):.3f}\n"
            f"p_h person/cup={float(row['person_projected_measurement']):.4f}/{float(row['cup_projected_measurement']):.4f}\n"
            f"full depth={float(row['person_current_depth']):.3f}/{float(row['cup_current_depth']):.3f}\n"
            f"center50 depth={float(center50['person_depth']):.3f}/{float(center50['cup_depth']):.3f}, R={float(center50['rsd_log']):.3f}\n"
            f"cluster depth={float(cluster['person_depth']):.3f}/{float(cluster['cup_depth']):.3f}, R={float(cluster['rsd_log']):.3f}\n"
            f"observed ratio={float(row['observed_depth_ratio']):.3f}, expected=[{float(row['expected_ratio_low']):.3f}, {float(row['expected_ratio_high']):.3f}]\n"
            f"gate={float(row['person_gate_score']):.2f}/{float(row['cup_gate_score']):.2f}; boundary person/cup="
            f"{any(bool(row[f'person_touches_{s}']) for s in ('top','bottom','left','right'))}/"
            f"{any(bool(row[f'cup_touches_{s}']) for s in ('top','bottom','left','right'))}\n"
            f"diagnostics={row['diagnostic_labels']}"
        )
        ax.text(0.01, 0.01, text, transform=ax.transAxes, va="bottom", fontsize=8, bbox={"facecolor": "white", "alpha": 0.86, "edgecolor": "#555555"})
        ax.set_title(f"Strict v2 person-cup error audit: frame {index}")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(output_dir / f"frame_{index:06d}_person_cup_audit.png", dpi=160)
        plt.close(fig)


def save_summary_plots(
    audit_rows: Sequence[Mapping[str, object]],
    depth_rows: Sequence[Mapping[str, object]],
    projection_rows: Sequence[Mapping[str, object]],
    frames: Mapping[int, FrameObservationJSON],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    x = np.asarray([int(row["global_frame_index"]) for row in audit_rows])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, [row["person_projected_measurement"] for row in audit_rows], marker="o", label="person bbox_height_norm")
    ax.plot(x, [row["cup_projected_measurement"] for row in audit_rows], marker="o", label="cup bbox_height_norm")
    ax.set(title="Person-Cup Projected Height Over Time", xlabel="global frame", ylabel="normalized height")
    ax.legend(); ax.grid(alpha=0.2); fig.tight_layout(); fig.savefig(output_dir / "person_cup_projected_height_over_time.png", dpi=180); plt.close(fig)

    selected = ("full_bbox_median", "center_50_bbox_median", "foreground_depth_cluster")
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for strategy in selected:
        group = [row for row in depth_rows if row["depth_strategy"] == strategy]
        axes[0].plot([row["frame_index"] for row in group], [row["person_depth"] for row in group], label=strategy)
        axes[1].plot([row["frame_index"] for row in group], [row["cup_depth"] for row in group], label=strategy)
    axes[0].set_ylabel("person relative depth"); axes[1].set_ylabel("cup relative depth"); axes[1].set_xlabel("global frame")
    axes[0].legend(fontsize=8); axes[1].legend(fontsize=8); axes[0].grid(alpha=0.2); axes[1].grid(alpha=0.2)
    fig.suptitle("Person-Cup Depth by Diagnostic Strategy"); fig.tight_layout(); fig.savefig(output_dir / "person_cup_depth_by_strategy_over_time.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    observed = np.asarray([row["observed_depth_ratio"] for row in audit_rows], float)
    low = np.asarray([row["expected_ratio_low"] for row in audit_rows], float)
    high = np.asarray([row["expected_ratio_high"] for row in audit_rows], float)
    ax.fill_between(x, low, high, alpha=0.25, label="expected interval")
    ax.plot(x, observed, marker="o", label="observed person/cup depth ratio")
    ax.set(title="Observed vs Expected Depth Ratio", xlabel="global frame", ylabel="depth ratio")
    ax.legend(); ax.grid(alpha=0.2); fig.tight_layout(); fig.savefig(output_dir / "person_cup_observed_vs_expected_ratio.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6))
    for strategy in DEPTH_STRATEGIES:
        group = [row for row in depth_rows if row["depth_strategy"] == strategy]
        values = np.asarray([float(row["rsd_log"]) for row in group], float)
        if np.isfinite(values).any():
            ax.plot([row["frame_index"] for row in group], values, label=strategy)
    ax.set(title="Person-Cup R_sd by Depth Strategy (Diagnostic Only)", xlabel="global frame", ylabel="R_sd_log")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.2); fig.tight_layout(); fig.savefig(output_dir / "person_cup_rsd_by_depth_strategy.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, [row["person_gate_score"] for row in audit_rows], marker="o", label="person gate score")
    ax.plot(x, [row["cup_gate_score"] for row in audit_rows], marker="o", label="cup gate score")
    ax.plot(x, [row["cup_bbox_area_ratio"] for row in audit_rows], marker=".", label="cup bbox area ratio")
    ax.set(title="Gate and Geometry Quality Over Time (Heuristic)", xlabel="global frame", ylabel="score / ratio")
    ax.legend(); ax.grid(alpha=0.2); fig.tight_layout(); fig.savefig(output_dir / "person_cup_gate_and_quality_over_time.png", dpi=180); plt.close(fig)

    examples = [audit_rows[0], audit_rows[len(audit_rows) // 2], audit_rows[-1]]
    fig, axes = plt.subplots(len(examples), 2, figsize=(10, 8))
    for row_index, row in enumerate(examples):
        frame = frames[int(row["global_frame_index"])]
        depth_path = Path(str(frame.depth_map_path)); depth_path = depth_path if depth_path.is_absolute() else PROJECT_ROOT / depth_path
        depth = np.load(depth_path)
        for column, prefix in enumerate(("person", "cup")):
            x1, y1, x2, y2 = (int(float(row[f"{prefix}_bbox_{name}"])) for name in ("x1", "y1", "x2", "y2"))
            crop = depth[max(0,y1):min(depth.shape[0],y2), max(0,x1):min(depth.shape[1],x2)]
            values = crop[np.isfinite(crop) & (crop > 0)]
            axes[row_index, column].hist(values, bins=40, color="#4C78A8" if prefix == "person" else "#E45756", alpha=0.8)
            axes[row_index, column].set_title(f"frame {row['global_frame_index']} {prefix}")
    fig.suptitle("Object Bbox Depth Pixel Distributions"); fig.tight_layout(); fig.savefig(output_dir / "person_cup_depth_pixel_distribution_examples.png", dpi=180); plt.close(fig)


def attribution_report(
    audit_rows: Sequence[Mapping[str, object]],
    summaries: Sequence[Mapping[str, object]],
    track_rows: Sequence[Mapping[str, object]],
    projection_rows: Sequence[Mapping[str, object]],
    stats: Mapping[str, int],
) -> dict[str, object]:
    formal = _finite(row["rsd_log"] for row in audit_rows)
    candidates = [row for row in summaries if math.isfinite(float(row["mean_rsd_log"]))]
    best = min(candidates, key=lambda row: float(row["mean_rsd_log"]))
    labels = Counter(reason for row in audit_rows for reason in str(row["diagnostic_labels"]).split("|") if reason)
    primary = labels.most_common(1)[0][0] if labels else "no_obvious_issue"
    persistent = float(track_rows[0]["persistent_high_residual_ratio"]) if track_rows else math.nan
    complete = float(np.mean([
        not any(bool(row[f"person_touches_{side}"]) for side in ("top", "bottom", "left", "right"))
        for row in audit_rows
    ]))
    quality = Counter(str(row["cup_measurement_quality"]) for row in audit_rows)
    formal_mean = float(np.mean(formal))
    best_mean = float(best["mean_rsd_log"])
    improvement = formal_mean - best_mean
    improvement_ratio = improvement / formal_mean if formal_mean > 0 else 0.0
    projection_means = {
        strategy: float(np.mean(_finite(
            row["rsd_log"]
            for row in projection_rows
            if row["projection_strategy"] == strategy
        )))
        for strategy in ("bbox_height_norm", "sqrt_bbox_area_norm", "bbox_diagonal_norm")
    }
    best_projection = min(projection_means, key=projection_means.get)  # type: ignore[arg-type]
    all_gates_passed = all(
        str(row["person_gate_passed"]).lower() == "true"
        and str(row["cup_gate_passed"]).lower() == "true"
        for row in audit_rows
    )
    if persistent > 0.8 and "likely_person_pose_mismatch" in labels:
        primary = "likely_person_pose_mismatch"
    elif persistent > 0.8 and best_mean > 0.1 and improvement_ratio < 0.25 and all_gates_passed:
        primary = "likely_prior_domain_mismatch"
    secondary = [reason for reason, _ in labels.most_common() if reason != primary][:4]
    if primary != "likely_prior_domain_mismatch" and persistent > 0.8 and best_mean > 0.1:
        secondary.append("likely_prior_domain_mismatch")
    secondary = list(dict.fromkeys(secondary))[:4]
    return {
        "video_id": audit_rows[0]["video_id"], "label_pair": audit_rows[0]["label_pair"],
        "num_valid_frame_pairs": len(audit_rows),
        "num_unique_track_pairs": len({row["track_pair_id"] for row in audit_rows}),
        "input_duplicate_frame_pair_count": stats["input_duplicate_frame_pair_count"],
        "duplicate_frame_pair_count": stats["duplicate_frame_pair_count"],
        "repeated_clip_pair_count": stats["repeated_clip_pair_count"],
        "frames_per_track_pair": dict(Counter(str(row["track_pair_id"]) for row in audit_rows)),
        "formal_mean_rsd_log": formal_mean, "best_depth_strategy": best["depth_strategy"],
        "best_depth_strategy_mean_rsd_log": best_mean,
        "depth_strategy_improvement_ratio": improvement_ratio,
        "best_projection_strategy": best_projection,
        "projection_strategy_mean_rsd_log": projection_means,
        "all_formal_gates_passed": all_gates_passed,
        "persistent_high_residual_ratio": persistent,
        "person_complete_visibility_ratio": complete,
        "cup_measurement_quality_summary": dict(quality), "likely_primary_error_source": primary,
        "secondary_error_sources": secondary,
        "instance_segmentation_expected_benefit": (
            "high" if "likely_bbox_background_contamination" in labels and improvement_ratio > 0.25
            else "medium" if "likely_bbox_background_contamination" in labels else "low"
        ),
        "gate_refinement_expected_benefit": "high" if any("pose_mismatch" in key for key in labels) else "low",
        "depth_strategy_refinement_expected_benefit": "high" if improvement > 0.2 else "medium" if improvement > 0.05 else "low",
        "prior_domain_issue": True if any(
            reason in {"likely_prior_domain_mismatch", "likely_person_pose_mismatch", "likely_cup_pose_mismatch"}
            for reason in [primary, *secondary]
        ) else "uncertain",
        "formula_checks": {
            "formula_error_count": stats["formula_error_count"],
            "interval_order_error_count": stats["interval_order_error_count"],
            "swapped_pair_inconsistency_count": stats["swapped_pair_inconsistency_count"],
            "depth_mode_error_count": stats["depth_mode_error_count"],
        },
        "recommended_next_step": (
            "Add an independent upright-person/object-subtype applicability check and evaluate instance-mask depth "
            "on additional videos; do not change frozen physical min/max from this clip."
        ),
        "note": "Diagnostic result only. Strict v1/v2 priors and formal R_sd observations were not modified.",
    }


def run_audit(
    input_dir: Path,
    video_dir: Path,
    video_id: str,
    label_a: str,
    label_b: str,
    output_dir: Path,
    debug_threshold: float = 0.1,
) -> dict[str, object]:
    """Run the complete read-only strict v2 pair error audit."""

    v1_path = PROJECT_ROOT / "configs/scale_priors_strict_v1.yaml"
    v2_path = PROJECT_ROOT / "configs/scale_priors_strict_v2.yaml"
    before_hashes = {"v1": sha256_file(v1_path), "v2": sha256_file(v2_path)}
    audit_rows, depth_rows, projection_rows, stats, video_root = build_audit_data(
        input_dir, video_id, label_a, label_b
    )
    if stats["duplicate_frame_pair_count"] != 0:
        raise AssertionError("Deduplication failed: duplicate frame_pair_id remains.")
    summaries = depth_strategy_summary(depth_rows)
    track_rows = per_track_pair(audit_rows, debug_threshold)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_csv(audit_rows, output_dir / "person_cup_pair_audit.csv", PAIR_AUDIT_FIELDS)
    save_csv(depth_rows, output_dir / "person_cup_depth_strategy_comparison.csv", DEPTH_COMPARISON_FIELDS)
    save_csv(summaries, output_dir / "person_cup_depth_strategy_summary.csv", DEPTH_SUMMARY_FIELDS)
    save_csv(track_rows, output_dir / "per_track_pair_rsd_audit.csv", TRACK_FIELDS)
    save_csv(projection_rows, output_dir / "person_cup_projection_strategy_comparison.csv", PROJECTION_FIELDS)

    observation_root, _ = _resolve_paths(input_dir)
    frames, _ = _load_frames(observation_root, video_id)
    video_path = video_dir / f"{video_id}.mp4"
    save_frame_plots(audit_rows, depth_rows, frames, video_path, output_dir / "frames")
    save_summary_plots(audit_rows, depth_rows, projection_rows, frames, output_dir)
    report = attribution_report(audit_rows, summaries, track_rows, projection_rows, stats)
    report.update(
        {
            "num_unique_track_pairs": len(track_rows),
            "strict_v1_hash": before_hashes["v1"], "strict_v2_hash": before_hashes["v2"],
            "depth_direction_convention": "larger_is_farther", "depth_mode": "real_depth_invert",
        }
    )
    after_hashes = {"v1": sha256_file(v1_path), "v2": sha256_file(v2_path)}
    if after_hashes != before_hashes:
        raise AssertionError("Strict prior config changed during read-only audit.")
    (output_dir / "error_attribution_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print("Strict R_sd v2 person-cup error audit:")
    for key in (
        "num_valid_frame_pairs", "num_unique_track_pairs", "duplicate_frame_pair_count",
        "repeated_clip_pair_count", "formal_mean_rsd_log", "best_depth_strategy",
        "best_depth_strategy_mean_rsd_log", "likely_primary_error_source",
    ):
        print(f"  {key}: {report[key]}")
    for key, value in report["formula_checks"].items():
        print(f"  {key}: {value}")
    print("Diagnostic only: no strict prior, observation, or formal R_sd was modified.")
    return report


def main() -> None:
    args = parse_args()
    run_audit(
        Path(args.input_dir), Path(args.video_dir), args.video_id, args.label_a, args.label_b,
        Path(args.output_dir), args.debug_threshold,
    )


if __name__ == "__main__":
    main()
