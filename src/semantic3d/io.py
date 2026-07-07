"""JSON input/output helpers for semantic3d observations and residual results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Union

from .observations import (
    ClipObservationJSON,
    ClipResidualResultJSON,
    FrameObservationJSON,
)

PathLike = Union[str, Path]


def _write_json(data: Mapping[str, Any], output_path: PathLike) -> None:
    """Write a JSON mapping with stable formatting, creating parents first."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")


def _read_json(json_path: PathLike) -> Mapping[str, Any]:
    """Read a JSON object from disk and validate that it is a mapping."""

    path = Path(json_path)
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}, got {type(data).__name__}.")
    return data


def save_frame_observation(
    frame_obs: FrameObservationJSON, output_path: PathLike
) -> None:
    """Save a single frame observation to a JSON file."""

    _write_json(frame_obs.to_dict(), output_path)


def load_frame_observation(json_path: PathLike) -> FrameObservationJSON:
    """Load a single frame observation from a JSON file."""

    return FrameObservationJSON.from_dict(_read_json(json_path))


def save_clip_observation(clip_obs: ClipObservationJSON, output_path: PathLike) -> None:
    """Save a clip observation to a JSON file."""

    _write_json(clip_obs.to_dict(), output_path)


def load_clip_observation(json_path: PathLike) -> ClipObservationJSON:
    """Load a clip observation from a JSON file."""

    return ClipObservationJSON.from_dict(_read_json(json_path))


def save_clip_residual_result(
    result: ClipResidualResultJSON, output_path: PathLike
) -> None:
    """Save a clip residual result to a JSON file."""

    _write_json(result.to_dict(), output_path)


def load_clip_residual_result(json_path: PathLike) -> ClipResidualResultJSON:
    """Load a clip residual result from a JSON file."""

    return ClipResidualResultJSON.from_dict(_read_json(json_path))
