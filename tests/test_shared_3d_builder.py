from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from semantic3d.observations import FrameObservationJSON, ObjectObservationJSON
from semantic3d.reconstruction.shared_3d_builder import Shared3DFrameBuilder

from synthetic_geometry import (
    cuboid_xyz,
    projected_point_observations,
    rasterize_sparse_depth,
    synthetic_camera,
    synthetic_depth_observation,
)

ROOT = Path(__file__).resolve().parents[1]


def _object(object_id: str, bbox: list[float] | None) -> ObjectObservationJSON:
    return ObjectObservationJSON(
        object_id=object_id,
        label="synthetic_cuboid",
        mask_area=1000.0,
        frame_area=640.0 * 480.0,
        depth=8.0,
        confidence=1.0,
        bbox=bbox,
        metadata={"preserve": True},
    )


def test_builder_shares_exact_camera_and_depth_instances() -> None:
    camera = synthetic_camera(with_pose=False)
    assert camera.K is not None
    xyz = cuboid_xyz()
    points = projected_point_observations(xyz, camera.K)
    pixels = np.asarray([[point.x, point.y] for point in points], dtype=float)
    valid_object = _object(
        "valid",
        [float(pixels[:, 0].min()), float(pixels[:, 1].min()), float(pixels[:, 0].max()), float(pixels[:, 1].max())],
    )
    invalid_object = _object("invalid", None)
    frame = FrameObservationJSON(0, "frame_000", 640, 480, [valid_object, invalid_object])
    depth = synthetic_depth_observation(rasterize_sparse_depth(xyz, camera.K))
    shared = Shared3DFrameBuilder().build(
        video_id="synthetic_video",
        frame=frame,
        depth=depth,
        camera=camera,
        boundary_points_by_object={"valid": points},
    )
    assert shared.valid
    assert shared.camera is camera
    assert shared.depth is depth
    assert len(shared.objects) == 2
    assert shared.objects[0].valid
    assert not shared.objects[1].valid
    assert shared.metadata["static_dynamic_shared_contract"] == "Shared3DFrameObservation"


def test_no_valid_object_makes_frame_invalid_without_crashing() -> None:
    camera = synthetic_camera(with_pose=False)
    depth_map = np.full((480, 640), 5.0, dtype=np.float32)
    depth = synthetic_depth_observation(depth_map)
    frame = FrameObservationJSON(0, "frame_000", 640, 480, [_object("bad", None)])
    shared = Shared3DFrameBuilder().build(
        video_id="synthetic_video", frame=frame, depth=depth, camera=camera
    )
    assert not shared.valid
    assert shared.missing_reason == "no_valid_object_reconstruction"
    assert len(shared.objects) == 1
    assert not shared.objects[0].valid


def test_strict_scale_prior_hashes_remain_frozen() -> None:
    expected = {
        "scale_priors_strict_v1.yaml": "e86466e19fe3e1663fa855fdf73843cf7ab8b5b6c8fa8771aa46172f6b726a6b",
        "scale_priors_strict_v2.yaml": "3c55c8eb42e19d2447b00794085d8e6fa233c37a2071ef51f23b447d23ef268b",
    }
    for name, digest in expected.items():
        assert hashlib.sha256((ROOT / "configs" / name).read_bytes()).hexdigest() == digest
