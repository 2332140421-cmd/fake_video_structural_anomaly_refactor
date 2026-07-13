#!/usr/bin/env python3
"""Generate controlled perturbations for R_depth_cons validation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _ensure_project_environment() -> None:
    project_python = PROJECT_ROOT / ".venv" / "bin" / "python"
    current_python = Path(sys.executable).resolve()
    if project_python.exists() and current_python != project_python.resolve():
        os.execv(str(project_python), [str(project_python), *sys.argv])


_ensure_project_environment()

SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from semantic3d.io import save_clip_observation  # noqa: E402
from semantic3d.object_association import ObjectAssociator  # noqa: E402
from semantic3d.observations import (  # noqa: E402
    ClipObservationJSON,
    FrameObservationJSON,
    ObjectObservationJSON,
)
from scripts.run_depth_consistency_pipeline import (  # noqa: E402
    load_candidate_clips,
    resolve_requested_depth_mode,
    select_unique_frames_and_clips,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate perturbed observation JSON.")
    parser.add_argument("--input_observation_dir", required=True)
    parser.add_argument("--output_observation_dir", required=True)
    parser.add_argument("--video_id", default=None)
    parser.add_argument("--track_id", default=None)
    parser.add_argument("--frame_index", type=int, default=None)
    parser.add_argument(
        "--perturbation_type",
        choices=[
            "depth_scale",
            "depth_inverse_scale",
            "mask_area_scale",
            "bbox_scale",
            "combined_inconsistent",
        ],
        required=True,
    )
    parser.add_argument("--factor", type=float, default=1.5)
    parser.add_argument("--metadata_output", default=None)
    parser.add_argument("--auto_select_track", action="store_true")
    parser.add_argument(
        "--depth_mode",
        choices=["auto", "no_depth", "real_depth_no_invert", "real_depth_invert"],
        default="real_depth_invert",
    )
    return parser.parse_args()


def load_associated_frames(
    input_observation_dir: Path,
    depth_mode: str,
) -> tuple[str, dict[str, list[FrameObservationJSON]]]:
    """Load, filter, deduplicate, and associate input frames."""

    clips = load_candidate_clips(input_observation_dir)
    resolved_mode = resolve_requested_depth_mode(clips, depth_mode)
    frames_by_video, _, _ = select_unique_frames_and_clips(clips, resolved_mode)
    associator = ObjectAssociator()
    associated = {
        video_id: associator.associate(frames)
        for video_id, frames in sorted(frames_by_video.items())
    }
    return resolved_mode, associated


def auto_select_target(
    frames_by_video: dict[str, list[FrameObservationJSON]],
) -> tuple[str, str, int]:
    """Select a track with at least three detections, targeting its middle frame."""

    for video_id, frames in sorted(frames_by_video.items()):
        by_track: dict[str, list[int]] = {}
        for frame in frames:
            for obj in frame.objects:
                if obj.track_id:
                    by_track.setdefault(obj.track_id, []).append(int(frame.frame_index))
        for track_id, indices in sorted(by_track.items()):
            unique_indices = sorted(set(indices))
            if len(unique_indices) >= 3:
                return video_id, track_id, unique_indices[len(unique_indices) // 2]
    raise ValueError("Could not auto-select a track with at least 3 observations.")


def perturb_frames(
    frames_by_video: dict[str, list[FrameObservationJSON]],
    video_id: str,
    track_id: str,
    frame_index: int,
    perturbation_type: str,
    factor: float,
) -> tuple[dict[str, list[FrameObservationJSON]], dict[str, Any]]:
    """Return perturbed frames and metadata for the modified object."""

    if factor <= 0:
        raise ValueError(f"factor must be > 0, got {factor}.")
    found = False
    original_depth = perturbed_depth = 0.0
    original_mask_area = perturbed_mask_area = 0.0
    output: dict[str, list[FrameObservationJSON]] = {}

    for current_video_id, frames in frames_by_video.items():
        new_frames: list[FrameObservationJSON] = []
        for frame in frames:
            new_objects: list[ObjectObservationJSON] = []
            for obj in frame.objects:
                if (
                    current_video_id == video_id
                    and obj.track_id == track_id
                    and int(frame.frame_index) == int(frame_index)
                ):
                    found = True
                    original_depth = float(obj.depth)
                    original_mask_area = float(obj.mask_area)
                    new_obj = _perturb_object(obj, frame, perturbation_type, factor)
                    perturbed_depth = float(new_obj.depth)
                    perturbed_mask_area = float(new_obj.mask_area)
                    new_objects.append(new_obj)
                else:
                    new_objects.append(obj)
            new_frames.append(replace(frame, objects=new_objects))
        output[current_video_id] = new_frames

    if not found:
        raise ValueError(
            f"Could not find target video_id={video_id}, track_id={track_id}, "
            f"frame_index={frame_index}."
        )
    metadata = {
        "enabled": True,
        "type": perturbation_type,
        "factor": factor,
        "target_video_id": video_id,
        "target_track_id": track_id,
        "target_frame_index": frame_index,
        "original_depth": original_depth,
        "perturbed_depth": perturbed_depth,
        "original_mask_area": original_mask_area,
        "perturbed_mask_area": perturbed_mask_area,
    }
    return output, metadata


def _perturb_object(
    obj: ObjectObservationJSON,
    frame: FrameObservationJSON,
    perturbation_type: str,
    factor: float,
) -> ObjectObservationJSON:
    depth = float(obj.depth)
    mask_area = float(obj.mask_area)
    bbox = list(obj.bbox) if obj.bbox is not None else None

    if perturbation_type == "depth_scale":
        depth *= factor
    elif perturbation_type == "depth_inverse_scale":
        depth /= factor
    elif perturbation_type == "mask_area_scale":
        mask_area *= factor
    elif perturbation_type == "bbox_scale":
        bbox = _scale_bbox(bbox, factor, frame.width, frame.height)
        mask_area = _bbox_area(bbox)
    elif perturbation_type == "combined_inconsistent":
        depth *= factor
        mask_area *= factor
        if bbox is not None:
            bbox = _scale_bbox(bbox, factor**0.5, frame.width, frame.height)
            mask_area = max(mask_area, _bbox_area(bbox))
    else:
        raise ValueError(f"Unknown perturbation_type: {perturbation_type}")

    if depth <= 0 or mask_area <= 0:
        raise ValueError("Perturbation produced invalid depth or mask_area.")
    return replace(obj, depth=depth, mask_area=mask_area, bbox=bbox)


def _scale_bbox(
    bbox: Optional[list[float]],
    factor: float,
    width: int,
    height: int,
) -> Optional[list[float]]:
    if bbox is None:
        return None
    x1, y1, x2, y2 = [float(value) for value in bbox]
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    half_w = max(1.0, (x2 - x1) * factor / 2.0)
    half_h = max(1.0, (y2 - y1) * factor / 2.0)
    nx1 = max(0.0, cx - half_w)
    ny1 = max(0.0, cy - half_h)
    nx2 = min(float(width), cx + half_w)
    ny2 = min(float(height), cy + half_h)
    if nx2 <= nx1 or ny2 <= ny1:
        raise ValueError("Perturbation produced invalid bbox.")
    return [nx1, ny1, nx2, ny2]


def _bbox_area(bbox: Optional[list[float]]) -> float:
    if bbox is None:
        return 0.0
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def save_perturbed_observations(
    frames_by_video: dict[str, list[FrameObservationJSON]],
    output_dir: Path,
    depth_mode: str,
    perturbation_metadata: dict[str, Any],
) -> None:
    """Save perturbed associated observations without modifying originals."""

    output_dir.mkdir(parents=True, exist_ok=True)
    for old_json in output_dir.rglob("*.json"):
        old_json.unlink()
    for video_id, frames in sorted(frames_by_video.items()):
        clip = ClipObservationJSON(
            clip_id=f"{video_id}_perturbed_depth_consistency",
            video_id=video_id,
            frame_indices=[int(frame.frame_index) for frame in frames],
            frames=frames,
            metadata={
                "depth_mode": depth_mode,
                "source": "depth_consistency_perturbation",
                "perturbation": perturbation_metadata,
            },
        )
        video_dir = output_dir / video_id
        save_clip_observation(
            clip,
            video_dir / f"{video_id}_perturbed_depth_consistency.json",
        )


def main() -> None:
    args = parse_args()
    depth_mode, frames_by_video = load_associated_frames(
        Path(args.input_observation_dir),
        args.depth_mode,
    )
    if args.auto_select_track:
        video_id, track_id, frame_index = auto_select_target(frames_by_video)
    else:
        if args.video_id is None or args.track_id is None or args.frame_index is None:
            raise ValueError(
                "Provide --video_id, --track_id, and --frame_index, or use "
                "--auto_select_track."
            )
        video_id, track_id, frame_index = args.video_id, args.track_id, args.frame_index

    perturbed, perturbation_metadata = perturb_frames(
        frames_by_video,
        video_id=video_id,
        track_id=track_id,
        frame_index=frame_index,
        perturbation_type=args.perturbation_type,
        factor=args.factor,
    )
    save_perturbed_observations(
        perturbed,
        Path(args.output_observation_dir),
        depth_mode=depth_mode,
        perturbation_metadata=perturbation_metadata,
    )
    metadata_output = Path(args.metadata_output) if args.metadata_output else (
        Path(args.output_observation_dir) / "perturbation_metadata.json"
    )
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    with metadata_output.open("w", encoding="utf-8") as file:
        json.dump({"perturbation": perturbation_metadata}, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(f"Saved perturbed observations: {args.output_observation_dir}")
    print(f"Saved perturbation metadata: {metadata_output}")
    print(
        f"target: video={video_id}, track={track_id}, frame={frame_index}, "
        f"type={args.perturbation_type}, factor={args.factor}"
    )


if __name__ == "__main__":
    main()
