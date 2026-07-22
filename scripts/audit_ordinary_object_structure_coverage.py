#!/usr/bin/env python3
"""Audit every stable formal-mask track through the ordinary-structure funnel."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
for candidate in (PROJECT_ROOT, SRC_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from semantic3d.occlusion import (  # noqa: E402
    InstanceMaskObservation,
    adaptive_erosion_pixels,
    eroded_mask_interior,
    select_formal_mask_internal_points,
    track_formal_mask_internal_points,
)


COLUMNS = (
    "video_id", "object_track_id", "semantic_label", "frame_count",
    "formal_mask_valid_ratio", "geometry_mode", "depth_valid_ratio",
    "eroded_mask_area_ratio", "candidate_internal_points",
    "stable_point_count", "point_track_consistency", "valid_3d_point_ratio",
    "structure_edge_count", "valid_transition_count", "final_status",
    "primary_failure_reason", "all_failure_reasons", "point_audit_frame_count",
    "adaptive_erosion", "truth_labels_used",
)


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _dynamic_modes(dynamic_root: Path) -> tuple[dict[str, str], dict[tuple[str, str], set[int]]]:
    modes: dict[str, str] = {}
    shared_frames: dict[tuple[str, str], set[int]] = defaultdict(set)
    for report_path in dynamic_root.glob("*/smoke_report.json"):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        video_id = str(report.get("video_id", ""))
        modes[video_id] = str(report.get("geometry_mode", "unavailable"))
        for row in _read_csv(report_path.parent / "object_point_bindings.csv"):
            if not _bool(row.get("valid", False)):
                continue
            for value in row.get("frame_indices", "").split(";"):
                if value:
                    shared_frames[(video_id, row["object_track_id"])].add(int(value))
    return modes, shared_frames


def _load_observations(
    rows: Sequence[Mapping[str, str]],
    *,
    max_frames: int,
    frames_root: Path,
) -> tuple[list[InstanceMaskObservation], dict[int, np.ndarray], list[float], int]:
    valid_rows = [row for row in rows if _bool(row["valid"]) and row["visible_mask_path"]]
    selected = valid_rows[:max_frames]
    observations = []
    images = {}
    erosion_ratios = []
    candidate_count = 0
    for position, row in enumerate(selected):
        mask = np.load(row["visible_mask_path"], allow_pickle=False).astype(bool)
        frame_index = int(row["frame_index"])
        observation = InstanceMaskObservation.from_visible_mask(
            video_id=row["video_id"], frame_index=frame_index,
            object_track_id=row["object_track_id"], semantic_label=row["class_name"],
            mask=mask, confidence=float(row["confidence"]),
            source_provider=row["source_provider"],
            metadata={"formal_mask_evidence": True, "legacy_bbox_fallback": False},
        )
        observations.append(observation)
        erosion = adaptive_erosion_pixels(mask)
        interior = eroded_mask_interior(mask, erosion)
        erosion_ratios.append(float(np.count_nonzero(interior) / np.count_nonzero(mask)))
        image_path = frames_root / row["video_id"] / f"frame_{frame_index:06d}.jpg"
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is not None:
            images[frame_index] = image
            if position == 0:
                candidate_count = len(select_formal_mask_internal_points(
                    image, observation, erosion_pixels=None,
                ))
    return observations, images, erosion_ratios, candidate_count


def audit_ordinary_object_structure_coverage(
    *,
    coverage_root: Path,
    dynamic_root: Path,
    output_dir: Path,
    stable_quality_threshold: float = 0.5,
    max_point_audit_frames: int = 64,
) -> dict[str, Any]:
    """Write one complete funnel row for every stable mask track."""

    tracking = _read_csv(coverage_root / "mask_tracking_quality.csv")
    masks = _read_csv(coverage_root / "mask_coverage.csv")
    structures = _read_csv(coverage_root / "structure_graph_coverage.csv")
    residuals = _read_csv(coverage_root / "structure_residual_coverage.csv")
    stable_keys = {
        (row["video_id"], row["object_track_id"])
        for row in tracking
        if _bool(row["valid"]) and float(row["track_quality"]) >= stable_quality_threshold
    }
    masks_by_track: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in masks:
        masks_by_track[(row["video_id"], row["object_track_id"])].append(row)
    modes, shared_frames = _dynamic_modes(dynamic_root)
    graph_lookup = {(row["video_id"], row["object_track_id"]): row for row in structures if _bool(row["valid"])}
    residual_lookup = {(row["video_id"], row["object_track_id"]): row for row in residuals if _bool(row["valid"])}
    rows = []
    for key in sorted(stable_keys):
        track_rows = sorted(masks_by_track[key], key=lambda row: int(row["frame_index"]))
        valid_rows = [row for row in track_rows if _bool(row["valid"])]
        label = valid_rows[0]["class_name"] if valid_rows else track_rows[0]["class_name"]
        observations, images, erosion_ratios, candidate_count = _load_observations(
            valid_rows, max_frames=max_point_audit_frames,
            frames_root=coverage_root / "frames",
        )
        usable_observations = [item for item in observations if item.frame_index in images]
        points = track_formal_mask_internal_points(
            images, usable_observations, max_points=24, erosion_pixels=None,
        ) if usable_observations else ()
        counts = Counter(item.point_id for item in points if item.valid)
        stable_point_count = sum(count >= 3 for count in counts.values())
        expected = len(counts) * len(usable_observations)
        consistency = sum(counts.values()) / expected if expected else float("nan")
        geometry_mode = modes.get(key[0], "unsupported_mode")
        shared_count = len(shared_frames.get(key, set()))
        depth_ratio = shared_count / len(track_rows) if track_rows else float("nan")
        graph = graph_lookup.get(key)
        residual = residual_lookup.get(key)
        edge_count = int(graph["edge_count"]) if graph else 0
        transition_count = int(residual["valid_residual_count"]) if residual else 0
        failures = []
        if label == "person":
            failures.append("non_ordinary_person_track")
        if len(track_rows) < 3:
            failures.append("track_too_short")
        if len(valid_rows) / len(track_rows) < 0.5:
            failures.append("insufficient_formal_mask_coverage")
        if geometry_mode not in {"static_camera_3d", "full_se3_3d"}:
            failures.append("unsupported_geometry_mode")
        if not math.isfinite(depth_ratio) or depth_ratio < 0.8:
            failures.append("insufficient_shared_depth_coverage")
        if candidate_count == 0:
            failures.append("no_mask_internal_candidates")
        if stable_point_count == 0:
            failures.append("no_stable_mask_internal_point_ids")
        if edge_count == 0:
            failures.append("no_formal_3d_structure_graph")
        if transition_count == 0:
            failures.append("no_formal_structure_transition")
        final_status = "formal_structure_residual_available" if not failures else "stopped_in_funnel"
        rows.append({
            "video_id": key[0], "object_track_id": key[1], "semantic_label": label,
            "frame_count": len(track_rows),
            "formal_mask_valid_ratio": len(valid_rows) / len(track_rows) if track_rows else float("nan"),
            "geometry_mode": geometry_mode, "depth_valid_ratio": depth_ratio,
            "eroded_mask_area_ratio": float(np.mean(erosion_ratios)) if erosion_ratios else float("nan"),
            "candidate_internal_points": candidate_count,
            "stable_point_count": stable_point_count,
            "point_track_consistency": consistency,
            "valid_3d_point_ratio": depth_ratio if stable_point_count else 0.0,
            "structure_edge_count": edge_count,
            "valid_transition_count": transition_count,
            "final_status": final_status,
            "primary_failure_reason": failures[0] if failures else "",
            "all_failure_reasons": ";".join(failures),
            "point_audit_frame_count": len(usable_observations),
            "adaptive_erosion": True, "truth_labels_used": False,
        })
    funnel = {
        "formal_mask_tracks": len(rows),
        "tracks_long_enough": sum(row["frame_count"] >= 3 for row in rows),
        "tracks_geometry_valid": sum(row["geometry_mode"] in {"static_camera_3d", "full_se3_3d"} for row in rows),
        "tracks_depth_valid": sum(math.isfinite(row["depth_valid_ratio"]) and row["depth_valid_ratio"] >= 0.8 for row in rows),
        "tracks_with_internal_points": sum(row["candidate_internal_points"] > 0 for row in rows),
        "tracks_with_stable_ids": sum(row["stable_point_count"] > 0 for row in rows),
        "tracks_with_structure_graph": sum(row["structure_edge_count"] > 0 and row["semantic_label"] != "person" for row in rows),
        "tracks_with_structure_residuals": sum(row["valid_transition_count"] > 0 and row["semantic_label"] != "person" for row in rows),
        "audited_track_count": len(rows),
        "expected_stable_track_count": len(stable_keys),
        "tracks_silently_dropped": len(stable_keys) - len(rows),
        "truth_labels_used": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "ordinary_structure_track_audit.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "ordinary_structure_funnel.json").write_text(json.dumps(funnel, indent=2), encoding="utf-8")
    with (output_dir / "ordinary_structure_funnel.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("stage", "track_count"))
        writer.writerows((key, value) for key, value in funnel.items() if isinstance(value, int))
    stages = [key for key in funnel if key.startswith("tracks_") or key == "formal_mask_tracks"]
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.bar(range(len(stages)), [funnel[key] for key in stages], color="#4e79a7")
    axis.set_xticks(range(len(stages)), stages, rotation=45, ha="right")
    axis.set_ylabel("track count")
    axis.set_title("Ordinary Object Structure Coverage Funnel")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "ordinary_structure_funnel.png", dpi=180, bbox_inches="tight")
    plt.close(figure)
    return {"rows": rows, "funnel": funnel}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage_root", type=Path, default=PROJECT_ROOT / "outputs/real_3d_evidence_coverage_v2")
    parser.add_argument("--dynamic_root", type=Path, default=PROJECT_ROOT / "outputs/real_object_dynamic_3d_smoke")
    parser.add_argument("--output_dir", type=Path, default=PROJECT_ROOT / "outputs/ordinary_object_structure_audit")
    parser.add_argument("--max_point_audit_frames", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = audit_ordinary_object_structure_coverage(
        coverage_root=args.coverage_root, dynamic_root=args.dynamic_root,
        output_dir=args.output_dir, max_point_audit_frames=args.max_point_audit_frames,
    )
    print(json.dumps(result["funnel"], indent=2))


if __name__ == "__main__":
    main()
