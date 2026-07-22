#!/usr/bin/env python3
"""Build formal-mask occlusion signals and audit multi-window candidates."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
for candidate in (PROJECT_ROOT, SRC_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from scripts.find_real_3d_evidence_clips import (  # noqa: E402
    scan_occlusion_event_windows,
    write_occlusion_window_diagnostics,
)


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _mask_contact(first: np.ndarray, second: np.ndarray) -> tuple[bool, bool]:
    overlap = bool(np.any(first & second))
    kernel = np.ones((5, 5), dtype=np.uint8)
    contact = bool(np.any(cv2.dilate(first.astype(np.uint8), kernel, iterations=1).astype(bool) & second))
    return overlap, contact


def build_occlusion_frame_signals(coverage_root: Path) -> list[dict[str, Any]]:
    """Derive observation-only event signals from formal visible masks."""

    masks = [row for row in _read_csv(coverage_root / "mask_coverage.csv") if _bool(row["valid"])]
    visibility = _read_csv(coverage_root / "visibility_event_coverage.csv")
    by_frame: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in masks:
        by_frame[(row["video_id"], int(row["frame_index"]))].append(row)
    visibility_by_frame: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in visibility:
        visibility_by_frame[(row["video_id"], int(row["frame_index"]))].append(row)
    videos = sorted({key[0] for key in by_frame} | {key[0] for key in visibility_by_frame})
    previous_area: dict[tuple[str, str], float] = {}
    previous_bbox: dict[tuple[str, str], tuple[float, ...]] = {}
    previously_missing: set[tuple[str, str]] = set()
    output = []
    for video_id in videos:
        indices = sorted({key[1] for key in by_frame if key[0] == video_id} | {key[1] for key in visibility_by_frame if key[0] == video_id})
        for frame_index in indices:
            rows = by_frame.get((video_id, frame_index), [])
            loaded = []
            for row in rows:
                path = Path(row["visible_mask_path"])
                if path.exists():
                    loaded.append((row, np.load(path, allow_pickle=False).astype(bool)))
            overlap = False
            contact = False
            for first_index, (_, first) in enumerate(loaded):
                for _, second in loaded[first_index + 1:]:
                    pair_overlap, pair_contact = _mask_contact(first, second)
                    overlap = overlap or pair_overlap
                    contact = contact or pair_contact
            area_drop = False
            reappearance = False
            for row, _ in loaded:
                track_key = (video_id, row["object_track_id"])
                area = float(row["mask_area"])
                if track_key in previous_area and area < 0.70 * previous_area[track_key]:
                    area_drop = True
                if track_key in previously_missing:
                    reappearance = True
                    previously_missing.discard(track_key)
                previous_area[track_key] = area
                try:
                    previous_bbox[track_key] = tuple(json.loads(row["mask_bbox"]))
                except (TypeError, json.JSONDecodeError):
                    pass
            visibility_rows = visibility_by_frame.get((video_id, frame_index), [])
            missing_tracks = [row["object_track_id"] for row in visibility_rows if row["event_type"] == "detector_missing"]
            out_of_frame = False
            for track_id in missing_tracks:
                track_key = (video_id, track_id)
                previously_missing.add(track_key)
                bbox = previous_bbox.get(track_key)
                if bbox and loaded:
                    height, width = loaded[0][1].shape
                    x1, y1, x2, y2 = bbox
                    out_of_frame = out_of_frame or x1 <= 2 or y1 <= 2 or x2 >= width - 2 or y2 >= height - 2
            output.append({
                "video_id": video_id, "frame_index": frame_index,
                "multi_object_mask_overlap": overlap,
                "sustained_mask_contact": contact,
                "visible_area_drop": area_drop,
                "possible_occluder": contact and len(loaded) >= 2,
                "depth_order_available": False,
                "disappearance": bool(missing_tracks),
                "reappearance": reappearance,
                "out_of_frame": out_of_frame,
                "detector_missing_control": bool(missing_tracks),
                "formal_mask_object_count": len(loaded),
                "object_track_ids": ";".join(sorted(row["object_track_id"] for row, _ in loaded)),
                "scene_cut": any(_bool(row.get("scene_cut", False)) for row in visibility_rows),
            })
    return output


def audit_occlusion_event_coverage(
    *, coverage_root: Path, output_dir: Path,
    window_sizes: Sequence[int] = (8, 16, 24, 32),
) -> dict[str, Any]:
    """Write frame signals, every tested window, and an explicit empty-event result."""

    frame_rows = build_occlusion_frame_signals(coverage_root)
    windows = scan_occlusion_event_windows(frame_rows, window_sizes=window_sizes)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_columns = (
        "video_id", "frame_index", "multi_object_mask_overlap",
        "sustained_mask_contact", "visible_area_drop", "possible_occluder",
        "depth_order_available", "disappearance", "reappearance",
        "out_of_frame", "detector_missing_control", "formal_mask_object_count",
        "object_track_ids", "scene_cut",
    )
    with (output_dir / "occlusion_frame_signals.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=frame_columns)
        writer.writeheader()
        writer.writerows(frame_rows)
    write_occlusion_window_diagnostics(windows, output_dir / "occlusion_window_diagnostics.csv")
    accepted = [row for row in windows if row["accepted"]]
    summary = {
        "video_count": len({row["video_id"] for row in frame_rows}),
        "frame_signal_count": len(frame_rows),
        "tested_window_count": len(windows),
        "formal_occlusion_candidate_count": len(accepted),
        "status": "formal_candidates_available" if accepted else "no_observable_occlusion_event",
        "window_sizes": list(window_sizes),
        "truth_labels_used": False,
        "anomaly_scores_used": False,
        "depth_order_available": any(row["depth_order_available"] for row in frame_rows),
    }
    (output_dir / "occlusion_window_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage_root", type=Path, default=PROJECT_ROOT / "outputs/real_3d_evidence_coverage_v2")
    parser.add_argument("--output_dir", type=Path, default=PROJECT_ROOT / "outputs/occlusion_event_coverage_audit")
    parser.add_argument("--window_sizes", type=int, nargs="+", default=(8, 16, 24, 32))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = audit_occlusion_event_coverage(
        coverage_root=args.coverage_root, output_dir=args.output_dir,
        window_sizes=tuple(args.window_sizes),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
