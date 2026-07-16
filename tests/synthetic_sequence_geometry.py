"""Synthetic P3-0 sequences with known camera, object, cut, and depth drift."""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import numpy as np

from semantic3d.shared_3d_observation import Shared3DFrameObservation

from synthetic_geometry import synthetic_object_3d, synthetic_shared_3d_frame


def camera_pose(position: Sequence[float]) -> np.ndarray:
    """Return T_world_from_camera for identity rotation and known camera centre."""

    transform = np.eye(4, dtype=float)
    transform[:3, 3] = np.asarray(position, dtype=float)
    return transform


def make_world_consistent_sequence(
    camera_positions: Sequence[Sequence[float]],
    *,
    moving_object_offsets: Sequence[Sequence[float]] | None = None,
    metric: bool = True,
) -> tuple[list[Shared3DFrameObservation], dict[int, np.ndarray]]:
    """Generate camera-frame objects whose static world centre is known."""

    world_center = np.asarray([0.0, 0.0, 8.0], dtype=float)
    offsets = moving_object_offsets or [(0.0, 0.0, 0.0)] * len(camera_positions)
    frames: list[Shared3DFrameObservation] = []
    poses: dict[int, np.ndarray] = {}
    for frame_index, (position, object_offset) in enumerate(
        zip(camera_positions, offsets, strict=True)
    ):
        pose = camera_pose(position)
        poses[frame_index] = pose
        object_world = world_center + np.asarray(object_offset, dtype=float)
        object_camera = object_world - np.asarray(position, dtype=float)
        obj = synthetic_object_3d(
            f"object_{frame_index}",
            label="synthetic_object",
            center=object_camera,
            size=(1.0, 1.0, 1.0),
            metric=metric,
        )
        obj = replace(
            obj,
            frame_index=frame_index,
            track_id="synthetic_track",
            metadata={
                **dict(obj.metadata),
                "source_bbox": [80.0, 60.0, 120.0, 120.0],
            },
        )
        frame = synthetic_shared_3d_frame((obj,), metric=metric, width=320, height=240)
        frames.append(
            replace(
                frame,
                video_id="synthetic_sequence",
                frame_index=frame_index,
                source_frame_id=f"synthetic_{frame_index:03d}",
                objects=(obj,),
            )
        )
    return frames, poses


def depth_drift_pair(
    *,
    scale: float,
    shift: float = 0.0,
    inverse_domain: bool = False,
    count: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Return source/target samples with a known alignment transform."""

    source = np.linspace(1.0, 10.0, count, dtype=float)
    if inverse_domain:
        target_inverse = scale / source + shift
        target = 1.0 / target_inverse
    else:
        target = scale * source + shift
    return source, target


def dotted_background_scene(
    *,
    moving_box_x: int,
    width: int = 320,
    height: int = 240,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a deterministic static background and its moving foreground mask."""

    generator = np.random.default_rng(20260716)
    image = generator.integers(0, 210, size=(height, width, 3), dtype=np.uint8)
    mask = np.zeros((height, width), dtype=bool)
    x1, y1, x2, y2 = moving_box_x, 80, moving_box_x + 60, 160
    image[y1:y2, x1:x2] = (255, 255, 255)
    mask[y1:y2, x1:x2] = True
    return image, mask
