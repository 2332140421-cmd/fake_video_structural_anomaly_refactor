from __future__ import annotations

from _bootstrap import ensure_project_test_environment

ensure_project_test_environment(__file__)

import json
import subprocess
import sys
from pathlib import Path

import pytest

from semantic3d.io import (
    load_clip_observation,
    load_clip_residual_result,
    load_frame_observation,
    save_clip_observation,
    save_clip_residual_result,
    save_frame_observation,
)
from semantic3d.observations import (
    ClipObservationJSON,
    ClipResidualResultJSON,
    FrameObservationJSON,
    ObjectObservationJSON,
)
from semantic3d.scale_depth import ScalePrior, scale_depth_residual


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRAME_AREA = 1280.0 * 720.0
SCALE_PRIORS = {
    "soccer_ball": ScalePrior(min_size=0.20, max_size=0.24),
    "elephant": ScalePrior(min_size=2.40, max_size=3.40),
}


def _make_frame(frame_index: int = 0) -> FrameObservationJSON:
    """Create a reusable two-object frame observation."""

    return FrameObservationJSON(
        frame_index=frame_index,
        frame_id=f"demo_video_001_{frame_index:06d}",
        width=1280,
        height=720,
        image_path=f"frames/demo_video_001_{frame_index:06d}.png",
        objects=[
            ObjectObservationJSON(
                object_id=f"soccer_ball_f{frame_index}",
                label="soccer_ball",
                mask_area=5_000.0,
                frame_area=FRAME_AREA,
                depth=3.0,
                confidence=0.98,
                bbox=[120.0, 420.0, 190.0, 490.0],
                mask_path=f"masks/soccer_ball_f{frame_index}.png",
            ),
            ObjectObservationJSON(
                object_id=f"elephant_f{frame_index}",
                label="elephant",
                mask_area=80_000.0,
                frame_area=FRAME_AREA,
                depth=12.0,
                confidence=0.99,
                bbox=[690.0, 220.0, 1030.0, 570.0],
                mask_path=f"masks/elephant_f{frame_index}.png",
            ),
        ],
        depth_map_path=f"depth/demo_video_001_{frame_index:06d}.npy",
        flow_residual_map_path=f"residuals/flow_{frame_index:06d}.npy",
        depth_residual_map_path=f"residuals/depth_{frame_index:06d}.npy",
        corr_residual_map_path=f"residuals/corr_{frame_index:06d}.npy",
    )


def test_save_and_load_frame_observation(tmp_path: Path) -> None:
    frame = _make_frame()
    output_path = tmp_path / "nested" / "frame_000000.json"

    save_frame_observation(frame, output_path)
    loaded = load_frame_observation(output_path)

    assert output_path.exists()
    assert loaded.to_dict() == frame.to_dict()
    assert loaded.objects[0].bbox == [120.0, 420.0, 190.0, 490.0]
    assert loaded.depth_residual_map_path == "residuals/depth_000000.npy"


def test_save_and_load_clip_observation(tmp_path: Path) -> None:
    clip = ClipObservationJSON(
        clip_id="clip_001",
        video_id="demo_video_001",
        frame_indices=[0, 1],
        frames=[_make_frame(0), _make_frame(1)],
        metadata={"split": "demo", "synthetic": True},
    )
    output_path = tmp_path / "clip_001.json"

    save_clip_observation(clip, output_path)
    loaded = load_clip_observation(output_path)

    assert loaded.to_dict() == clip.to_dict()
    assert loaded.frames[1].objects[1].object_id == "elephant_f1"
    assert loaded.metadata["synthetic"] is True


def test_save_and_load_clip_residual_result(tmp_path: Path) -> None:
    result = ClipResidualResultJSON(
        clip_id="clip_001",
        object_residuals=[
            {"frame_index": 0, "object_id": "elephant_f0", "depth_cons": 0.2}
        ],
        pair_residuals=[
            {
                "frame_index": 0,
                "object_id_a": "soccer_ball_f0",
                "object_id_b": "elephant_f0",
                "scale_depth": 0.0,
            }
        ],
        clip_score=0.0,
        details={"score_rule": "max scale_depth"},
    )
    output_path = tmp_path / "results" / "clip_001_residual_result.json"

    save_clip_residual_result(result, output_path)
    loaded = load_clip_residual_result(output_path)

    assert output_path.exists()
    assert loaded.to_dict() == result.to_dict()
    assert loaded.pair_residuals[0]["scale_depth"] == pytest.approx(0.0)


def test_demo_json_can_compute_rsd() -> None:
    demo_script = PROJECT_ROOT / "examples" / "demo_observation_io.py"

    subprocess.run(
        [sys.executable, str(demo_script)],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    clip = load_clip_observation(
        PROJECT_ROOT / "data" / "demo_observations" / "clip_001.json"
    )

    frame = clip.frames[0]
    soccer_ball = frame.objects[0].to_scale_depth_observation()
    elephant = frame.objects[1].to_scale_depth_observation()
    residual, details = scale_depth_residual(soccer_ball, elephant, SCALE_PRIORS)

    assert residual == pytest.approx(0.0)
    assert details["lower"] <= details["depth_ratio"] <= details["upper"]
    assert (
        PROJECT_ROOT / "outputs" / "results" / "clip_001_residual_result.json"
    ).exists()


def test_missing_required_field_error(tmp_path: Path) -> None:
    bad_path = tmp_path / "bad_frame.json"
    bad_path.write_text(
        json.dumps(
            {
                "frame_index": 0,
                "frame_id": "bad_frame",
                "height": 720,
                "objects": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Missing required field.*width"):
        load_frame_observation(bad_path)
