from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import cv2
import numpy as np

from semantic3d.depth_provider import (
    DepthObservation,
    DepthRepresentation,
    DepthScaleStatus,
    LargerValueMeans,
)
from semantic3d.depth_provider import MockDepthProvider
from semantic3d.providers import MockObjectProvider
from semantic3d.sequence_geometry import stabilize_sequence_geometry
from scripts.run_sequence_geometry_stabilization import (
    run_sequence_geometry_stabilization,
)


ROOT = Path(__file__).resolve().parents[1]


def _depth(image: np.ndarray, frame_index: int) -> DepthObservation:
    height, width = image.shape[:2]
    base = np.repeat(np.linspace(2.0, 8.0, width)[None, :], height, axis=0)
    drifted = (1.0 + 0.03 * frame_index) * base + 0.01 * frame_index
    valid = np.isfinite(drifted) & (drifted > 0.0)
    return DepthObservation(
        depth_map=drifted,
        raw_model_output=1.0 / drifted,
        visualization_depth=np.zeros_like(drifted),
        depth_representation=DepthRepresentation.RELATIVE_DEPTH,
        scale_status=DepthScaleStatus.RELATIVE_PER_FRAME,
        larger_value_means=LargerValueMeans.FARTHER,
        valid_mask=valid,
        confidence_map=None,
        provider_name="test_relative_depth",
        frame_index=frame_index,
        valid=True,
        quality=1.0,
    )


def test_real_video_frames_run_stabilization_without_semantic_scale_prior() -> None:
    video = ROOT / "data/tests_videos/tests_real_videos/real_1.mp4"
    capture = cv2.VideoCapture(str(video))
    images = {}
    try:
        for index in range(4):
            success, image = capture.read()
            assert success and image is not None
            images[index] = image
    finally:
        capture.release()
    height, width = images[0].shape[:2]
    K = np.asarray(
        [[max(width, height), 0.0, (width - 1) / 2], [0.0, max(width, height), (height - 1) / 2], [0.0, 0.0, 1.0]]
    )
    depths = {index: _depth(image, index) for index, image in images.items()}
    result = stabilize_sequence_geometry(
        images,
        depths,
        K,
        frame_indices=range(4),
        foreground_masks={index: np.zeros(image.shape[:2], dtype=bool) for index, image in images.items()},
    )
    assert result.valid
    assert result.pose_pairs
    assert result.metadata["semantic_scale_prior_used"] is False
    assert result.metadata["scale_prior_config_read"] is False
    for gate in result.frame_validity.values():
        if not gate.dynamic_3d_valid:
            assert np.isnan(gate.value)


def test_stabilization_source_does_not_import_scale_prior_or_anomaly_residuals() -> None:
    import semantic3d.sequence_geometry.depth_alignment as alignment
    import semantic3d.sequence_geometry.pose_estimation as pose
    import semantic3d.sequence_geometry.stabilization as stabilization

    source = "\n".join(
        inspect.getsource(module) for module in (alignment, pose, stabilization)
    ).lower()
    assert "scale_priors.yaml" not in source
    assert "rsd" not in source
    assert "r_semantic_size_3d" not in source


def test_strict_prior_hashes_remain_frozen_after_p305() -> None:
    expected = {
        "scale_priors_strict_v1.yaml": "e86466e19fe3e1663fa855fdf73843cf7ab8b5b6c8fa8771aa46172f6b726a6b",
        "scale_priors_strict_v2.yaml": "3c55c8eb42e19d2447b00794085d8e6fa233c37a2071ef51f23b447d23ef268b",
    }
    for filename, digest in expected.items():
        assert hashlib.sha256((ROOT / "configs" / filename).read_bytes()).hexdigest() == digest


def test_stabilization_smoke_writes_required_outputs(tmp_path: Path) -> None:
    video_path = tmp_path / "static.mp4"
    generator = np.random.default_rng(20)
    frame = generator.integers(0, 220, size=(120, 160, 3), dtype=np.uint8)
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (160, 120)
    )
    assert writer.isOpened()
    for _ in range(4):
        writer.write(frame)
    writer.release()
    output_dir = tmp_path / "output"
    summary = run_sequence_geometry_stabilization(
        clip_id="test_static",
        video_path=video_path,
        start_frame=0,
        num_frames=4,
        output_dir=output_dir,
        object_provider=MockObjectProvider(mode="reasonable"),
        depth_provider=MockDepthProvider(),
    )
    expected = {
        "motion_regimes.csv",
        "pose_candidates.csv",
        "pose_graph.json",
        "camera_trajectory.csv",
        "depth_alignment_candidates.csv",
        "depth_alignment_global.csv",
        "background_tracks.csv",
        "geometry_quality.json",
        "diagnostics.png",
    }
    assert expected.issubset(path.name for path in output_dir.iterdir())
    assert all((output_dir / filename).stat().st_size > 0 for filename in expected)
    assert summary["semantic_scale_prior_used"] is False
    assert summary["formal_dynamic_anomaly_residuals_computed"] is False


def test_unverified_sequence_does_not_upgrade_relative_per_frame_depth() -> None:
    images = {index: np.zeros((80, 100, 3), dtype=np.uint8) for index in range(3)}
    depths = {index: _depth(image, index) for index, image in images.items()}
    K = np.asarray([[100.0, 0.0, 49.5], [0.0, 100.0, 39.5], [0.0, 0.0, 1.0]])
    result = stabilize_sequence_geometry(
        images,
        depths,
        K,
        frame_indices=range(3),
    )
    assert result.sequence_scale_status.value == "relative_per_frame"
    assert not result.dynamic_3d_valid
    assert any(np.isnan(item.value) for item in result.frame_validity.values())
