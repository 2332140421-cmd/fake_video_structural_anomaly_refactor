#!/usr/bin/env python3
"""Run cross-frame association and R_depth_cons on observation JSON files."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _ensure_project_environment() -> None:
    """Re-execute with the project .venv Python when available."""

    project_python = PROJECT_ROOT / ".venv" / "bin" / "python"
    current_python = Path(sys.executable).resolve()
    if project_python.exists() and current_python != project_python.resolve():
        os.execv(str(project_python), [str(project_python), *sys.argv])


_ensure_project_environment()

SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np  # noqa: E402

from semantic3d.depth_temporal_consistency import (  # noqa: E402
    DepthTemporalResidualResult,
    aggregate_clip_depth_residuals,
    aggregate_track_depth_residuals,
    compute_frame_depth_reference,
    compute_track_transitions,
    save_depth_consistency_tracks_plot_from_csv,
)
from semantic3d.io import load_clip_observation, save_clip_observation  # noqa: E402
from semantic3d.object_association import ObjectAssociator  # noqa: E402
from semantic3d.observations import ClipObservationJSON, FrameObservationJSON  # noqa: E402


PAIR_FIELDS = [
    "video_id",
    "depth_mode",
    *[field.name for field in fields(DepthTemporalResidualResult)],
]
TRACK_FIELDS = [
    "video_id",
    "depth_mode",
    "track_id",
    "label",
    "num_transitions",
    "mean_residual",
    "max_residual",
    "topk_mean_residual",
    "mean_weighted_residual",
    "valid_ratio",
]
CLIP_FIELDS = [
    "video_id",
    "depth_mode",
    "clip_id",
    "num_tracks",
    "num_valid_transitions",
    "mean_R_depth_cons",
    "max_R_depth_cons",
    "topk_mean_R_depth_cons",
    "weighted_clip_score",
]

DEPTH_MODES = {"no_depth", "real_depth_no_invert", "real_depth_invert"}


@dataclass(frozen=True)
class LoadedClip:
    """Clip loaded with its source path and inferred depth mode."""

    path: Path
    clip: ClipObservationJSON
    depth_mode: str


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Associate objects across frames and compute R_depth_cons. "
            "This residual is kept separate from object-pair R_sd."
        )
    )
    parser.add_argument(
        "--observation_dir",
        default=str(PROJECT_ROOT / "outputs" / "depth_direction_batch"),
    )
    parser.add_argument(
        "--depth_map_dir",
        default=str(PROJECT_ROOT / "outputs" / "depth_direction_batch"),
    )
    parser.add_argument(
        "--output_pair_csv",
        default=str(PROJECT_ROOT / "outputs" / "results" / "depth_consistency_pairs.csv"),
    )
    parser.add_argument(
        "--output_track_csv",
        default=str(PROJECT_ROOT / "outputs" / "results" / "depth_consistency_tracks.csv"),
    )
    parser.add_argument(
        "--output_clip_csv",
        default=str(PROJECT_ROOT / "outputs" / "results" / "depth_consistency_clips.csv"),
    )
    parser.add_argument(
        "--associated_observation_dir",
        default=str(PROJECT_ROOT / "outputs" / "observations_with_tracks"),
    )
    parser.add_argument(
        "--visualization_path",
        default=str(
            PROJECT_ROOT / "outputs" / "visualizations" / "depth_consistency_tracks.png"
        ),
    )
    parser.add_argument(
        "--raw_residual_visualization_path",
        default=str(
            PROJECT_ROOT
            / "outputs"
            / "visualizations"
            / "depth_consistency_raw_residual.png"
        ),
    )
    parser.add_argument(
        "--thresholded_residual_visualization_path",
        default=str(
            PROJECT_ROOT
            / "outputs"
            / "visualizations"
            / "depth_consistency_thresholded_residual.png"
        ),
    )
    parser.add_argument(
        "--combined_visualization_path",
        default=str(
            PROJECT_ROOT
            / "outputs"
            / "visualizations"
            / "depth_consistency_combined_analysis.png"
        ),
    )
    parser.add_argument(
        "--depth_mode",
        choices=["auto", "no_depth", "real_depth_no_invert", "real_depth_invert"],
        default="auto",
        help=(
            "Depth mode to use. Use real_depth_invert for the main experiment. "
            "auto is allowed only when the input contains exactly one mode."
        ),
    )
    parser.add_argument("--iou_threshold", type=float, default=0.1)
    parser.add_argument("--center_distance_threshold", type=float, default=0.25)
    parser.add_argument("--max_area_ratio", type=float, default=3.0)
    parser.add_argument("--max_frame_gap", type=int, default=1)
    parser.add_argument("--tolerance", type=float, default=0.10)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--max_files", type=int, default=None)
    return parser.parse_args()


def load_candidate_clips(
    observation_dir: Path,
    max_files: Optional[int] = None,
) -> list[LoadedClip]:
    """Load clip observation JSON files, skipping non-observation JSON safely."""

    json_paths = sorted(observation_dir.rglob("*.json"))
    if max_files is not None:
        json_paths = json_paths[: max(0, int(max_files))]
    clips: list[LoadedClip] = []
    for path in json_paths:
        try:
            clip = load_clip_observation(path)
            clips.append(LoadedClip(path=path, clip=clip, depth_mode=infer_depth_mode(path, clip)))
        except Exception as exc:
            print(f"Skipping non-clip JSON {path}: {type(exc).__name__}: {exc}")
    if not clips:
        print(f"No clip observation JSON files found in {observation_dir}.")
    return clips


def infer_depth_mode(path: Path, clip: ClipObservationJSON) -> str:
    """Infer depth mode from metadata first, then path names."""

    metadata = clip.metadata
    if isinstance(metadata.get("depth_mode"), str):
        value = str(metadata["depth_mode"])
        if value in DEPTH_MODES:
            return value

    depth_provider = str(metadata.get("depth_provider", "")).lower()
    invert_depth = bool(metadata.get("invert_depth", False))
    if depth_provider == "none":
        return "no_depth"
    if depth_provider in {"real_depth", "mock_depth"}:
        return "real_depth_invert" if invert_depth else "real_depth_no_invert"

    text = str(path).lower()
    if "real_depth_invert_observations" in text:
        return "real_depth_invert"
    if "real_depth_no_invert_observations" in text:
        return "real_depth_no_invert"
    if "no_depth_observations" in text:
        return "no_depth"
    return "unknown"


def resolve_requested_depth_mode(
    clips: list[LoadedClip],
    requested_depth_mode: str,
) -> str:
    """Resolve and validate the depth mode requested for one run."""

    modes = sorted({clip.depth_mode for clip in clips})
    known_modes = [mode for mode in modes if mode in DEPTH_MODES]
    if requested_depth_mode != "auto":
        if requested_depth_mode not in modes:
            raise ValueError(
                f"Requested depth_mode={requested_depth_mode!r}, but input contains "
                f"modes {modes}. Pass an observation_dir for that mode or rerun "
                "observation generation."
            )
        return requested_depth_mode
    if len(known_modes) == 1 and len(modes) == 1:
        return known_modes[0]
    if len(known_modes) == 1 and set(modes) <= {known_modes[0], "unknown"}:
        return known_modes[0]
    if modes == ["unknown"]:
        return "unknown"
    raise ValueError(
        "Input observation_dir contains multiple depth modes "
        f"{modes}. Please rerun with --depth_mode real_depth_invert, "
        "--depth_mode real_depth_no_invert, or --depth_mode no_depth. "
        "The pipeline will not silently mix modes by frame_index."
    )


def _source_priority(path: Path) -> int:
    """Prefer inverted real-depth observations over other comparison outputs."""

    text = str(path).lower()
    if "real_depth_invert_observations" in text:
        return 40
    if "invert_observations" in text:
        return 35
    if "real_depth_no_invert_observations" in text:
        return 30
    if "no_depth_observations" in text:
        return 10
    return 20


def select_unique_frames_and_clips(
    clips: list[LoadedClip],
    depth_mode: str,
) -> tuple[dict[str, list[FrameObservationJSON]], list[ClipObservationJSON], int]:
    """Select unique frames per video and preferred clip records."""

    best_frames: dict[tuple[str, int], tuple[int, FrameObservationJSON]] = {}
    best_clips: dict[tuple[str, str], tuple[int, ClipObservationJSON]] = {}
    seen_frame_keys: set[tuple[str, int]] = set()
    duplicate_frame_count = 0

    for loaded in clips:
        if loaded.depth_mode != depth_mode:
            continue
        path = loaded.path
        clip = loaded.clip
        priority = _source_priority(path)
        clip_key = (clip.video_id, clip.clip_id)
        if clip_key not in best_clips or priority > best_clips[clip_key][0]:
            best_clips[clip_key] = (priority, clip)
        for frame in clip.frames:
            frame_key = (clip.video_id, int(frame.frame_index))
            if frame_key in seen_frame_keys:
                duplicate_frame_count += 1
            else:
                seen_frame_keys.add(frame_key)
            if frame_key not in best_frames or priority > best_frames[frame_key][0]:
                best_frames[frame_key] = (priority, frame)

    frames_by_video: dict[str, list[FrameObservationJSON]] = {}
    for (video_id, _), (_, frame) in best_frames.items():
        frames_by_video.setdefault(video_id, []).append(frame)
    for video_id, frames in frames_by_video.items():
        frames_by_video[video_id] = sorted(frames, key=lambda item: int(item.frame_index))

    selected_clips = [
        item[1] for _, item in sorted(best_clips.items(), key=lambda pair: pair[0])
    ]
    return frames_by_video, selected_clips, duplicate_frame_count


def _load_depth_map(frame: FrameObservationJSON, depth_map_dir: Path) -> Optional[np.ndarray]:
    """Load a depth map for a frame if available."""

    candidates: list[Path] = []
    if frame.depth_map_path:
        raw = Path(frame.depth_map_path)
        candidates.append(raw)
        if not raw.is_absolute():
            candidates.append(PROJECT_ROOT / raw)
        candidates.extend(depth_map_dir.rglob(raw.name))
    if frame.image_path:
        image_stem = Path(frame.image_path).stem
        candidates.extend(depth_map_dir.rglob(f"{image_stem}_depth.npy"))

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists() and candidate.suffix.lower() == ".npy":
            return np.load(candidate)
    return None


def compute_depth_references(
    frames: list[FrameObservationJSON],
    depth_map_dir: Path,
) -> dict[int, Optional[float]]:
    """Compute frame-level depth references for all frames."""

    references: dict[int, Optional[float]] = {}
    for frame in frames:
        depth_map = _load_depth_map(frame, depth_map_dir)
        references[int(frame.frame_index)] = compute_frame_depth_reference(
            frame,
            depth_map=depth_map,
        )
    return references


def save_associated_observations(
    selected_clips: list[ClipObservationJSON],
    associated_frames_by_video: dict[str, list[FrameObservationJSON]],
    output_dir: Path,
    depth_mode: str,
) -> None:
    """Save one deduplicated video-level observation JSON per video."""

    output_dir.mkdir(parents=True, exist_ok=True)
    for old_json in output_dir.rglob("*.json"):
        old_json.unlink()
    metadata_by_video: dict[str, dict[str, Any]] = {}
    for clip in selected_clips:
        metadata_by_video.setdefault(clip.video_id, {}).update(clip.metadata)

    for video_id, frames in sorted(associated_frames_by_video.items()):
        frame_indices = [int(frame.frame_index) for frame in frames]
        updated_clip = ClipObservationJSON(
            clip_id=f"{video_id}_associated_tracks",
            video_id=video_id,
            frame_indices=frame_indices,
            frames=frames,
            metadata={
                **metadata_by_video.get(video_id, {}),
                "source": "global_deduplicated_associated_frames",
                "depth_mode": depth_mode,
            },
        )
        video_dir = output_dir / video_id
        save_clip_observation(
            updated_clip,
            video_dir / f"{video_id}_associated_tracks.json",
        )


def save_csv(rows: list[dict[str, Any]], output_path: Path, fieldnames: list[str]) -> None:
    """Save rows to CSV with a stable header, even when rows are empty."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def count_duplicate_track_frames(
    video_id: str,
    frames: list[FrameObservationJSON],
) -> int:
    """Count duplicate (video_id, frame_index, track_id) observations."""

    seen: set[tuple[str, int, str]] = set()
    duplicates = 0
    for frame in frames:
        for obj in frame.objects:
            if not obj.track_id:
                continue
            key = (video_id, int(frame.frame_index), str(obj.track_id))
            if key in seen:
                duplicates += 1
            else:
                seen.add(key)
    return duplicates


def assert_residual_formula(results: list[DepthTemporalResidualResult]) -> None:
    """Assert residual=max(0, raw_residual-tolerance) for valid transitions."""

    for result in results:
        if not result.valid:
            continue
        expected = max(0.0, float(result.raw_residual) - float(result.tolerance))
        if not np.isclose(float(result.residual), expected, rtol=1e-7, atol=1e-9):
            raise AssertionError(
                "Invalid R_depth_cons formula for "
                f"{result.track_id}: residual={result.residual}, expected={expected}."
            )


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    """Run association, residual computation, aggregation, and output writing."""

    clips = load_candidate_clips(Path(args.observation_dir), max_files=args.max_files)
    depth_mode = resolve_requested_depth_mode(clips, args.depth_mode)
    frames_by_video, selected_clips, duplicate_frame_count = select_unique_frames_and_clips(
        clips,
        depth_mode=depth_mode,
    )
    associator = ObjectAssociator(
        iou_threshold=args.iou_threshold,
        center_distance_threshold=args.center_distance_threshold,
        max_area_ratio=args.max_area_ratio,
        max_frame_gap=args.max_frame_gap,
    )

    associated_frames_by_video: dict[str, list[FrameObservationJSON]] = {}
    all_results: dict[str, list[DepthTemporalResidualResult]] = {}
    aggregate_stats = {
        "total_unique_frames": 0,
        "total_objects": 0,
        "total_tracks": 0,
        "duplicate_frame_count": duplicate_frame_count,
        "duplicate_track_frame_count": 0,
        "one_to_many_assignment_count": 0,
        "valid_transitions": 0,
        "skipped_invalid_depth": 0,
        "skipped_missing_reference": 0,
        "skipped_frame_gap": 0,
        "depth_mode": depth_mode,
    }

    for video_id, frames in sorted(frames_by_video.items()):
        associated_frames = associator.associate(frames)
        diagnostics = associator.last_diagnostics
        associated_frames_by_video[video_id] = associated_frames
        references = compute_depth_references(associated_frames, Path(args.depth_map_dir))
        results, stats = compute_track_transitions(
            associated_frames,
            references,
            max_frame_gap=args.max_frame_gap,
            tolerance=args.tolerance,
        )
        all_results[video_id] = results

        duplicate_track_frame_count = count_duplicate_track_frames(
            video_id,
            associated_frames,
        )
        aggregate_stats["total_unique_frames"] += len(associated_frames)
        aggregate_stats["total_objects"] += sum(len(frame.objects) for frame in associated_frames)
        aggregate_stats["total_tracks"] += len(
            {
                obj.track_id
                for frame in associated_frames
                for obj in frame.objects
                if obj.track_id
            }
        )
        aggregate_stats["duplicate_track_frame_count"] += max(
            duplicate_track_frame_count,
            diagnostics.duplicate_track_frame_count,
        )
        aggregate_stats["one_to_many_assignment_count"] += (
            diagnostics.one_to_many_assignment_count
        )
        for key in {
            "valid_transitions",
            "skipped_invalid_depth",
            "skipped_missing_reference",
            "skipped_frame_gap",
        }:
            aggregate_stats[key] += stats[key]

    save_associated_observations(
        selected_clips,
        associated_frames_by_video,
        Path(args.associated_observation_dir),
        depth_mode=depth_mode,
    )

    pair_rows: list[dict[str, Any]] = []
    track_rows: list[dict[str, Any]] = []
    clip_rows: list[dict[str, Any]] = []
    flat_results: list[DepthTemporalResidualResult] = []

    for video_id, results in sorted(all_results.items()):
        flat_results.extend(results)
        for result in results:
            pair_rows.append(
                {"video_id": video_id, "depth_mode": depth_mode, **result.to_dict()}
            )
        for summary in aggregate_track_depth_residuals(results, topk=args.topk):
            track_rows.append({"video_id": video_id, "depth_mode": depth_mode, **summary})

    for clip in selected_clips:
        clip_results = all_results.get(clip.video_id, [])
        clip_rows.append(
            {
                "video_id": clip.video_id,
                "depth_mode": depth_mode,
                **aggregate_clip_depth_residuals(
                    clip_results,
                    frame_indices=clip.frame_indices,
                    clip_id=clip.clip_id,
                    topk=args.topk,
                ),
            }
        )

    save_csv(pair_rows, Path(args.output_pair_csv), PAIR_FIELDS)
    save_csv(track_rows, Path(args.output_track_csv), TRACK_FIELDS)
    save_csv(clip_rows, Path(args.output_clip_csv), CLIP_FIELDS)
    save_depth_consistency_tracks_plot_from_csv(
        args.output_pair_csv,
        args.combined_visualization_path,
    )
    save_depth_consistency_tracks_plot_from_csv(
        args.output_pair_csv,
        args.visualization_path,
    )
    from semantic3d.depth_temporal_consistency import (  # noqa: WPS433
        save_raw_and_thresholded_residual_plots_from_csv,
    )

    save_raw_and_thresholded_residual_plots_from_csv(
        args.output_pair_csv,
        raw_output_path=args.raw_residual_visualization_path,
        thresholded_output_path=args.thresholded_residual_visualization_path,
        tolerance=args.tolerance,
    )

    valid_residuals = [result.residual for result in flat_results if result.valid]
    valid_raw_residuals = [result.raw_residual for result in flat_results if result.valid]
    assert_residual_formula(flat_results)
    aggregate_stats["mean_raw_residual"] = (
        float(np.mean(valid_raw_residuals)) if valid_raw_residuals else 0.0
    )
    aggregate_stats["mean_R_depth_cons"] = (
        float(np.mean(valid_residuals)) if valid_residuals else 0.0
    )
    aggregate_stats["max_R_depth_cons"] = max(valid_residuals) if valid_residuals else 0.0
    return aggregate_stats


def main() -> None:
    """Run the depth temporal consistency pipeline."""

    args = parse_args()
    stats = run_pipeline(args)
    print("Depth temporal consistency stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    if stats["valid_transitions"] == 0:
        print(
            "No valid R_depth_cons transitions were produced. Possible reasons: "
            "no repeated tracks, missing depth references, invalid object depths, "
            "or association thresholds too strict."
        )
    print(f"Saved pair CSV: {args.output_pair_csv}")
    print(f"Saved track CSV: {args.output_track_csv}")
    print(f"Saved clip CSV: {args.output_clip_csv}")
    print(f"Saved associated observations: {args.associated_observation_dir}")
    print(f"Saved visualization: {args.visualization_path}")
    print("Depth convention: larger relative depth means farther; not metric meters.")


if __name__ == "__main__":
    main()
