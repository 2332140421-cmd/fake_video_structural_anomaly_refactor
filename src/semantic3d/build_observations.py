"""Builders that convert extracted video frames into observation JSON objects."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

import cv2
import numpy as np

from .depth_provider import (
    BaseDepthProvider,
    compute_object_depth_from_bbox,
    save_depth_visualization,
)
from .observations import ClipObservationJSON, FrameObservationJSON
from .providers import BaseObjectProvider

PathLike = Union[str, Path]


def build_frame_observation(
    frame_path: PathLike,
    frame_index: int,
    object_provider: BaseObjectProvider,
    depth_provider: Optional[BaseDepthProvider] = None,
    depth_output_dir: Optional[PathLike] = None,
    save_depth_map: bool = False,
    default_depth: float = 5.0,
) -> FrameObservationJSON:
    """Build a FrameObservationJSON by reading image size, objects, and depth."""

    path = Path(frame_path)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read frame image with OpenCV: {path}")

    height, width = image.shape[:2]
    objects = object_provider.predict(path, frame_index, width, height)
    depth_map_path = None
    if depth_provider is not None:
        depth_map = depth_provider.predict_depth(path)
        if depth_map.shape != (height, width):
            raise ValueError(
                f"depth_map shape {depth_map.shape} does not match frame "
                f"shape {(height, width)} for {path}."
            )
        objects = [
            _replace_object_depth(
                obj,
                compute_object_depth_from_bbox(
                    depth_map,
                    obj.bbox,
                    method="median",
                    default_depth=default_depth,
                ),
            )
            for obj in objects
        ]

        if save_depth_map:
            if depth_output_dir is None:
                raise ValueError("depth_output_dir is required when save_depth_map=True.")
            output_dir = Path(depth_output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            depth_path = output_dir / f"{path.stem}_depth.npy"
            np.save(depth_path, np.asarray(depth_map, dtype=np.float32))
            save_depth_visualization(depth_map, output_dir / f"{path.stem}_depth.png")
            depth_map_path = str(depth_path)

    return FrameObservationJSON(
        frame_index=frame_index,
        frame_id=path.stem,
        width=width,
        height=height,
        objects=objects,
        image_path=str(path),
        depth_map_path=depth_map_path,
    )


def _replace_object_depth(obj: Any, depth: float) -> Any:
    """Return an ObjectObservationJSON-like record with a replaced depth."""

    from .observations import ObjectObservationJSON

    return ObjectObservationJSON(
        object_id=obj.object_id,
        label=obj.label,
        mask_area=obj.mask_area,
        frame_area=obj.frame_area,
        depth=float(depth),
        confidence=obj.confidence,
        bbox=obj.bbox,
        mask_path=obj.mask_path,
    )


def build_clip_observation(
    video_id: str,
    clip_id: str,
    frames: Iterable[FrameObservationJSON],
    metadata: Optional[Dict[str, Any]] = None,
) -> ClipObservationJSON:
    """Group frame observations into a ClipObservationJSON."""

    frame_list = list(frames)
    return ClipObservationJSON(
        clip_id=clip_id,
        video_id=video_id,
        frame_indices=[frame.frame_index for frame in frame_list],
        frames=frame_list,
        metadata=dict(metadata or {}),
    )
