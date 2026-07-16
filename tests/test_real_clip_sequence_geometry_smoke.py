from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from scripts.run_real_clip_sequence_geometry_smoke import (
    run_real_clip_sequence_geometry_smoke,
)
from semantic3d.depth_provider import MockDepthProvider
from semantic3d.observations import ObjectObservationJSON
from semantic3d.providers import BaseObjectProvider


class _MovingForegroundProvider(BaseObjectProvider):
    def predict(
        self, frame_path: str | Path, frame_index: int, width: int, height: int
    ) -> list[ObjectObservationJSON]:
        del frame_path
        x1 = 35.0 + 8.0 * frame_index
        return [
            ObjectObservationJSON(
                object_id=f"person_f{frame_index}",
                label="person",
                mask_area=1_600.0,
                frame_area=float(width * height),
                depth=5.0,
                confidence=0.95,
                bbox=[x1, 50.0, x1 + 40.0, 90.0],
                canonical_label="person",
                provenance={"test_provider": True},
            )
        ]


def _write_static_background_moving_object_video(path: Path) -> None:
    rng = np.random.default_rng(20260716)
    background = rng.integers(0, 180, size=(180, 240, 3), dtype=np.uint8)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (240, 180)
    )
    assert writer.isOpened()
    for index in range(4):
        frame = background.copy()
        x1 = 35 + 8 * index
        cv2.rectangle(frame, (x1, 50), (x1 + 40, 90), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()


def test_real_clip_smoke_writes_sequence_geometry_contract(tmp_path: Path) -> None:
    video_path = tmp_path / "clip.mp4"
    output_dir = tmp_path / "output"
    _write_static_background_moving_object_video(video_path)

    clip, quality = run_real_clip_sequence_geometry_smoke(
        video_path=video_path,
        start_frame=0,
        num_frames=4,
        output_dir=output_dir,
        object_provider=_MovingForegroundProvider(),
        depth_provider=MockDepthProvider(),
    )

    expected = {
        "shared_3d_clip.json",
        "camera_trajectory.csv",
        "relative_poses.csv",
        "depth_alignment.csv",
        "background_tracks.csv",
        "background_reprojection.csv",
        "sequence_geometry_quality.json",
        "camera_trajectory.png",
        "reprojection_diagnostics.png",
    }
    assert expected.issubset(path.name for path in output_dir.iterdir())
    assert all((output_dir / filename).stat().st_size > 0 for filename in expected)
    assert len(clip.frames) == 4
    assert quality.valid_pose_ratio == 1.0
    assert quality.depth_alignment_valid_ratio == 1.0
    assert all(pose.valid for pose in clip.relative_poses)
    assert all(
        pose.is_identity_relative_pose for pose in clip.relative_poses[1:]
    )
    assert clip.metadata["foreground_exclusion"] is True

    payload = json.loads((output_dir / "shared_3d_clip.json").read_text())
    assert payload["claim"].endswith("not dynamic anomaly detection")
    assert payload["frames"][0]["camera_intrinsics_source"] == "approximate"
    assert payload["frames"][0]["depth_scale_status"] == "relative_shared_sequence"
    quality_payload = json.loads(
        (output_dir / "sequence_geometry_quality.json").read_text()
    )
    assert quality_payload["metric_camera_trajectory"] is False
    assert quality_payload["dynamic_anomaly_residuals_computed"] is False

