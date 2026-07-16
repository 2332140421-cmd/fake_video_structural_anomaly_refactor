from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import cv2
import numpy as np

from semantic3d.depth_provider import MockDepthProvider
from semantic3d.dynamic_3d import Dynamic3DReadinessThresholds, load_shared_geometry_cache
from semantic3d.providers import MockObjectProvider
from scripts.run_real_dynamic_3d_smoke import run_real_dynamic_3d_smoke
from scripts.run_sequence_geometry_stabilization import run_sequence_geometry_stabilization


ROOT = Path(__file__).resolve().parents[1]


def _write_static_video(path: Path, frame_count: int = 5) -> None:
    generator = np.random.default_rng(91)
    frame = generator.integers(0, 220, size=(120, 160, 3), dtype=np.uint8)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (160, 120)
    )
    assert writer.isOpened()
    for _ in range(frame_count):
        writer.write(frame)
    writer.release()


def test_dynamic_smoke_consumes_cache_and_writes_required_outputs(tmp_path: Path) -> None:
    video = tmp_path / "static.mp4"
    _write_static_video(video)
    geometry_dir = tmp_path / "geometry"
    run_sequence_geometry_stabilization(
        clip_id="cached_static",
        video_path=video,
        start_frame=0,
        num_frames=5,
        output_dir=geometry_dir,
        object_provider=MockObjectProvider(mode="reasonable"),
        depth_provider=MockDepthProvider(),
    )
    manifest = geometry_dir / "shared_geometry_cache" / "shared_3d_clip_manifest.json"
    assert manifest.exists()
    cache = load_shared_geometry_cache(manifest)
    assert cache.clip.metadata["depth_reestimated"] is False
    output = tmp_path / "dynamic"
    report = run_real_dynamic_3d_smoke(
        geometry_cache_manifest=manifest,
        output_dir=output,
        thresholds=Dynamic3DReadinessThresholds(
            minimum_reprojection_improvement=0.0,
            minimum_depth_stability_improvement=0.0,
            minimum_background_3d_stability_improvement=0.0,
        ),
    )
    expected = {
        "dynamic_readiness.json",
        "point_tracks_2d.csv",
        "point_tracks_3d.csv",
        "track_residuals.csv",
        "reprojection_residuals.csv",
        "track_diagnostics.png",
        "reprojection_diagnostics.png",
        "smoke_report.json",
    }
    assert expected.issubset(path.name for path in output.iterdir())
    assert all((output / name).stat().st_size > 0 for name in expected)
    assert report["shared_clip_reused"] is True
    assert report["depth_reestimated"] is False
    assert report["intrinsics_reestimated"] is False
    assert report["pose_reestimated"] is False
    assert report["formal_real_anomaly_evidence"] is False


def test_dynamic_smoke_source_does_not_instantiate_geometry_models() -> None:
    import scripts.run_real_dynamic_3d_smoke as smoke

    source = inspect.getsource(smoke)
    assert "RealDepthProvider" not in source
    assert "RealObjectProvider" not in source
    assert "stabilize_sequence_geometry(" not in source


def test_current_real2_cache_is_rejected_when_available() -> None:
    readiness_path = (
        ROOT
        / "outputs/real_dynamic_3d_smoke/slowly_moving_camera/dynamic_readiness.json"
    )
    if not readiness_path.exists():
        return
    payload = json.loads(readiness_path.read_text(encoding="utf-8"))
    assert payload["mode"] == "unavailable"
    assert payload["dynamic_3d_ready"] is False
    assert "depth_stability_improved" in payload["missing_reason"]


def test_strict_prior_hashes_remain_frozen_after_p3a() -> None:
    expected = {
        "scale_priors_strict_v1.yaml": "e86466e19fe3e1663fa855fdf73843cf7ab8b5b6c8fa8771aa46172f6b726a6b",
        "scale_priors_strict_v2.yaml": "3c55c8eb42e19d2447b00794085d8e6fa233c37a2071ef51f23b447d23ef268b",
    }
    for filename, digest in expected.items():
        assert hashlib.sha256((ROOT / "configs" / filename).read_bytes()).hexdigest() == digest
