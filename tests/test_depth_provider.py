from __future__ import annotations

from _bootstrap import ensure_project_test_environment

ensure_project_test_environment(__file__)

from pathlib import Path

import cv2
import numpy as np
import pytest

from semantic3d.build_observations import build_frame_observation
from semantic3d.depth_provider import (
    MockDepthProvider,
    RealDepthProvider,
    compute_object_depth_from_bbox,
)
from semantic3d.real_object_provider import RealObjectProvider


def _write_image(path: Path, width: int = 100, height: int = 80) -> None:
    """Write a simple test image."""

    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((height, width, 3), 240, dtype=np.uint8)
    cv2.imwrite(str(path), image)


def test_mock_depth_provider_shape(tmp_path: Path) -> None:
    frame_path = tmp_path / "frame.png"
    _write_image(frame_path, width=96, height=64)

    depth = MockDepthProvider().predict_depth(frame_path)

    assert depth.shape == (64, 96)
    assert np.isfinite(depth).all()


def test_compute_object_depth_from_bbox_median() -> None:
    depth_map = np.arange(25, dtype=float).reshape(5, 5)

    depth = compute_object_depth_from_bbox(depth_map, [1, 1, 4, 4])

    assert depth == pytest.approx(np.median(depth_map[1:4, 1:4]))


def test_invalid_bbox_returns_default_depth() -> None:
    depth_map = np.ones((5, 5), dtype=float)

    assert compute_object_depth_from_bbox(depth_map, [4, 4, 4, 5], default_depth=7.0) == 7.0
    assert compute_object_depth_from_bbox(depth_map, None, default_depth=7.0) == 7.0


def test_real_depth_provider_import() -> None:
    pytest.importorskip("transformers")

    provider = RealDepthProvider(pipeline_instance=lambda _image: {"depth": np.ones((8, 8))})

    assert provider.model_name == "depth-anything/Depth-Anything-V2-Small"


def test_real_depth_provider_smoke(tmp_path: Path) -> None:
    pytest.importorskip("transformers")
    frame_path = tmp_path / "frame.png"
    _write_image(frame_path, width=96, height=64)

    try:
        provider = RealDepthProvider(
            model_name="depth-anything/Depth-Anything-V2-Small",
            device="cpu",
        )
    except Exception as exc:
        pytest.skip(f"real depth model is not available in this environment: {exc}")

    try:
        depth = provider.predict_depth(frame_path)
    except Exception as exc:
        pytest.skip(f"real depth inference is not available in this environment: {exc}")

    assert depth.shape == (64, 96)
    assert depth.dtype == np.float32
    assert np.isfinite(depth).all()
    assert float(depth.min()) > 0


def test_build_observation_with_real_depth_or_mock_depth(tmp_path: Path) -> None:
    frame_path = tmp_path / "frame.png"
    _write_image(frame_path, width=120, height=90)
    provider = RealObjectProvider(
        detector=lambda _path: [
            {
                "label": "person",
                "bbox": [5, 5, 30, 30],
                "confidence": 0.9,
            },
            {
                "label": "car",
                "bbox": [80, 60, 115, 88],
                "confidence": 0.9,
            },
        ],
        default_depth=5.0,
        confidence_threshold=0.3,
    )

    frame = build_frame_observation(
        frame_path,
        0,
        provider,
        depth_provider=MockDepthProvider(min_depth=1.0, max_depth=9.0),
        depth_output_dir=tmp_path / "depth_maps",
        save_depth_map=True,
        default_depth=5.0,
    )

    depths = [obj.depth for obj in frame.objects]
    assert len(depths) == 2
    assert depths[0] != pytest.approx(5.0)
    assert depths[1] != pytest.approx(5.0)
    assert depths[0] != pytest.approx(depths[1])
    assert frame.depth_map_path is not None
    assert Path(frame.depth_map_path).exists()
    assert Path(frame.depth_map_path).with_suffix(".png").exists()


def test_depth_provider_none_keeps_default_depth(tmp_path: Path) -> None:
    frame_path = tmp_path / "frame.png"
    _write_image(frame_path, width=120, height=90)
    provider = RealObjectProvider(
        detector=lambda _path: [
            {
                "label": "person",
                "bbox": [5, 5, 30, 30],
                "confidence": 0.9,
            }
        ],
        default_depth=5.0,
        confidence_threshold=0.3,
    )

    frame = build_frame_observation(frame_path, 0, provider, default_depth=5.0)

    assert frame.objects[0].depth == pytest.approx(5.0)
    assert frame.depth_map_path is None
