from __future__ import annotations

import inspect
from pathlib import Path

import cv2
import numpy as np

from semantic3d.depth_provider import MockDepthProvider
from semantic3d.dynamic_3d import Dynamic3DReadinessThresholds
from semantic3d.io import save_clip_observation
from semantic3d.observations import ClipObservationJSON, FrameObservationJSON, ObjectObservationJSON
from semantic3d.providers import MockObjectProvider
from scripts.run_real_dynamic_3d_smoke import run_real_dynamic_3d_smoke
from scripts.run_real_object_dynamic_3d_smoke import run_real_object_dynamic_3d_smoke
from scripts.run_sequence_geometry_stabilization import run_sequence_geometry_stabilization


def _write_static_video(path: Path, frame_count: int = 5) -> None:
    generator = np.random.default_rng(321)
    frame = generator.integers(0, 255, size=(120, 160, 3), dtype=np.uint8)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (160, 120))
    assert writer.isOpened()
    for _ in range(frame_count):
        writer.write(frame)
    writer.release()


def _associated_clip(frame_count: int) -> ClipObservationJSON:
    frames = []
    for frame in range(frame_count):
        frames.append(FrameObservationJSON(
            frame_index=frame, frame_id=f"f{frame}", width=160, height=120,
            objects=[ObjectObservationJSON(
                object_id=f"object_{frame}", label="car", mask_area=12000.0,
                frame_area=19200.0, depth=5.0, confidence=1.0,
                bbox=[5.0, 5.0, 155.0, 115.0], track_id="trk_car",
            )],
        ))
    return ClipObservationJSON("associated", "static", list(range(frame_count)), frames)


def test_real_object_smoke_reuses_cache_and_writes_outputs(tmp_path: Path) -> None:
    video = tmp_path / "static.mp4"
    _write_static_video(video)
    geometry = tmp_path / "geometry"
    run_sequence_geometry_stabilization(
        clip_id="cached_static", video_path=video, start_frame=0, num_frames=5,
        output_dir=geometry, object_provider=MockObjectProvider(mode="reasonable"),
        depth_provider=MockDepthProvider(),
    )
    manifest = geometry / "shared_geometry_cache/shared_3d_clip_manifest.json"
    readiness_dir = tmp_path / "readiness"
    run_real_dynamic_3d_smoke(
        geometry_cache_manifest=manifest, output_dir=readiness_dir,
        thresholds=Dynamic3DReadinessThresholds(
            minimum_reprojection_improvement=0.0,
            minimum_depth_stability_improvement=0.0,
            minimum_background_3d_stability_improvement=0.0,
        ),
    )
    associated = tmp_path / "associated.json"
    save_clip_observation(_associated_clip(5), associated)
    output = tmp_path / "object_dynamic"
    report = run_real_object_dynamic_3d_smoke(
        geometry_cache_manifest=manifest,
        readiness_path=readiness_dir / "dynamic_readiness.json",
        associated_observation_path=associated,
        output_dir=output,
    )
    expected = {
        "object_point_bindings.csv", "object_structure_graphs.json",
        "object_motion_predictions.csv", "direction_residuals.csv",
        "relative_velocity.csv", "structure_residuals.csv",
        "dynamic_reprojection_residuals.csv", "object_dynamic_summary.csv",
        "point_and_edge_diagnostics.png", "smoke_report.json",
    }
    assert expected.issubset(path.name for path in output.iterdir())
    assert all((output / name).stat().st_size > 0 for name in expected)
    assert report["shared_clip_reused"] is True
    assert report["depth_reestimated"] is False
    assert report["intrinsics_reestimated"] is False
    assert report["pose_reestimated"] is False
    assert report["current_frame_used_for_prediction"] is False
    assert report["bound_object_point_count"] > 0


def test_object_dynamic_smoke_source_does_not_instantiate_geometry_models() -> None:
    import scripts.run_real_object_dynamic_3d_smoke as smoke

    source = inspect.getsource(smoke)
    assert "RealDepthProvider" not in source
    assert "RealObjectProvider" not in source
    assert "stabilize_sequence_geometry(" not in source
