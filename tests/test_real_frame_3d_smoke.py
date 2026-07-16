from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from scripts.run_real_frame_3d_smoke import run_real_frame_3d_smoke
from semantic3d.depth_provider import MockDepthProvider
from semantic3d.observations import ObjectObservationJSON
from semantic3d.providers import BaseObjectProvider


class _InjectedObjectProvider(BaseObjectProvider):
    def predict(
        self, frame_path: str | Path, frame_index: int, width: int, height: int
    ) -> list[ObjectObservationJSON]:
        del frame_path
        return [
            ObjectObservationJSON(
                object_id=f"cup_f{frame_index}",
                label="cup",
                mask_area=1_600.0,
                frame_area=float(width * height),
                depth=5.0,
                confidence=0.95,
                bbox=[30.0, 20.0, 70.0, 60.0],
                canonical_label="cup",
                provenance={"provider": "test_injected_detector"},
                quality=0.95,
            )
        ]


def _write_video(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (128, 96)
    )
    assert writer.isOpened()
    for index in range(3):
        frame = np.zeros((96, 128, 3), dtype=np.uint8)
        cv2.rectangle(frame, (30 + index, 20), (70 + index, 60), (220, 220, 220), -1)
        writer.write(frame)
    writer.release()


def test_real_frame_smoke_outputs_shared_3d_and_qa(tmp_path: Path) -> None:
    video_path = tmp_path / "input.mp4"
    output_dir = tmp_path / "outputs"
    _write_video(video_path)

    shared = run_real_frame_3d_smoke(
        video_path=video_path,
        frame_index=1,
        output_dir=output_dir,
        object_provider=_InjectedObjectProvider(),
        depth_provider=MockDepthProvider(),
        keypoint_provider=None,
    )

    assert shared.valid
    assert len(shared.objects) == 1
    assert shared.objects[0].valid
    assert shared.objects[0].center_3d_camera is not None
    assert shared.objects[0].center_3d_world is None
    assert shared.camera.intrinsics_source == "approximate"
    assert not shared.camera.pose_valid
    assert shared.depth.metadata.get("legacy_normalized_depth", False) is False

    expected = {
        "shared_3d_frame.json",
        "object_3d_observations.csv",
        "reconstructed_points_3d.csv",
        "current_frame_reprojection.png",
        "reconstruction_quality_report.json",
    }
    assert expected.issubset(path.name for path in output_dir.iterdir())
    assert all((output_dir / filename).stat().st_size > 0 for filename in expected)

    payload = json.loads((output_dir / "shared_3d_frame.json").read_text())
    assert payload["reconstruction_description"] == "camera-frame relative sparse 3D"
    assert payload["metric_reconstruction"] is False
    assert payload["camera"]["approximate_intrinsics"] is True
    assert payload["camera"]["pose_valid"] is False
    assert payload["depth"]["canonical_geometry_depth"] is True
    assert payload["depth"]["legacy_normalized_depth_used"] is False

    report = json.loads(
        (output_dir / "reconstruction_quality_report.json").read_text()
    )
    record = report["quality_records"][0]
    assert record["cycle_evidence_role"] == "qa"
    assert record["cycle_is_anomaly_residual"] is False
    assert record["reconstruction_cycle_error"] is not None

