"""Demo for saving/loading observation JSON and computing scale-depth R_sd."""

from __future__ import annotations

import sys
from statistics import mean

from _bootstrap import ensure_project_environment

PROJECT_ROOT = ensure_project_environment()
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from semantic3d.io import (  # noqa: E402
    load_clip_observation,
    save_clip_observation,
    save_clip_residual_result,
)
from semantic3d.observations import (  # noqa: E402
    ClipObservationJSON,
    ClipResidualResultJSON,
    FrameObservationJSON,
    ObjectObservationJSON,
)
from semantic3d.scale_depth import ScalePrior, scale_depth_residual  # noqa: E402


SCALE_PRIORS = {
    "soccer_ball": ScalePrior(min_size=0.20, max_size=0.24),
    "elephant": ScalePrior(min_size=2.40, max_size=3.40),
}


def _make_object(
    object_id: str,
    label: str,
    mask_area: float,
    frame_area: float,
    depth: float,
    bbox: list[float],
) -> ObjectObservationJSON:
    """Create one JSON object observation for the demo clip."""

    return ObjectObservationJSON(
        object_id=object_id,
        label=label,
        mask_area=mask_area,
        frame_area=frame_area,
        depth=depth,
        confidence=1.0,
        bbox=bbox,
    )


def build_demo_clip() -> ClipObservationJSON:
    """Construct a two-frame observation clip with soccer_ball and elephant."""

    width, height = 1280, 720
    frame_area = float(width * height)
    frames = [
        FrameObservationJSON(
            frame_index=0,
            frame_id="demo_video_001_000000",
            width=width,
            height=height,
            image_path="frames/demo_video_001_000000.png",
            objects=[
                _make_object(
                    "soccer_ball_f0",
                    "soccer_ball",
                    mask_area=5_000.0,
                    frame_area=frame_area,
                    depth=3.0,
                    bbox=[120.0, 420.0, 190.0, 490.0],
                ),
                _make_object(
                    "elephant_f0",
                    "elephant",
                    mask_area=80_000.0,
                    frame_area=frame_area,
                    depth=12.0,
                    bbox=[690.0, 220.0, 1030.0, 570.0],
                ),
            ],
            depth_map_path="depth/demo_video_001_000000.npy",
            flow_residual_map_path="residuals/flow_demo_video_001_000000.npy",
            depth_residual_map_path="residuals/depth_demo_video_001_000000.npy",
            corr_residual_map_path="residuals/corr_demo_video_001_000000.npy",
        ),
        FrameObservationJSON(
            frame_index=1,
            frame_id="demo_video_001_000001",
            width=width,
            height=height,
            image_path="frames/demo_video_001_000001.png",
            objects=[
                _make_object(
                    "soccer_ball_f1",
                    "soccer_ball",
                    mask_area=5_400.0,
                    frame_area=frame_area,
                    depth=3.2,
                    bbox=[132.0, 418.0, 206.0, 492.0],
                ),
                _make_object(
                    "elephant_f1",
                    "elephant",
                    mask_area=78_000.0,
                    frame_area=frame_area,
                    depth=11.6,
                    bbox=[700.0, 218.0, 1038.0, 568.0],
                ),
            ],
            depth_map_path="depth/demo_video_001_000001.npy",
            flow_residual_map_path="residuals/flow_demo_video_001_000001.npy",
            depth_residual_map_path="residuals/depth_demo_video_001_000001.npy",
            corr_residual_map_path="residuals/corr_demo_video_001_000001.npy",
        ),
    ]
    return ClipObservationJSON(
        clip_id="clip_001",
        video_id="demo_video_001",
        frame_indices=[frame.frame_index for frame in frames],
        frames=frames,
        metadata={
            "source": "synthetic observation json demo",
            "note": "No real vision model is invoked in this demo.",
        },
    )


def compute_clip_scale_depth_result(
    clip_obs: ClipObservationJSON,
) -> ClipResidualResultJSON:
    """Compute soccer_ball-elephant R_sd for every frame in a loaded clip."""

    pair_residuals: list[dict[str, object]] = []
    residual_values: list[float] = []

    for frame in clip_obs.frames:
        by_label = {obj.label: obj for obj in frame.objects}
        soccer_ball = by_label["soccer_ball"].to_scale_depth_observation()
        elephant = by_label["elephant"].to_scale_depth_observation()
        residual, details = scale_depth_residual(
            soccer_ball, elephant, SCALE_PRIORS
        )
        residual_values.append(residual)
        pair_residuals.append(
            {
                "frame_index": frame.frame_index,
                "object_id_a": soccer_ball.object_id,
                "object_id_b": elephant.object_id,
                "residual_type": "scale_depth",
                "scale_depth": residual,
                "details": details,
            }
        )

    clip_score = max(residual_values) if residual_values else 0.0
    return ClipResidualResultJSON(
        clip_id=clip_obs.clip_id,
        object_residuals=[],
        pair_residuals=pair_residuals,
        clip_score=clip_score,
        details={
            "score_rule": "max frame-level soccer_ball-elephant R_sd",
            "mean_scale_depth": mean(residual_values) if residual_values else 0.0,
            "num_frames": len(clip_obs.frames),
        },
    )


def main() -> None:
    """Save, load, compute R_sd, and save residual results for a demo clip."""

    clip_path = PROJECT_ROOT / "data" / "demo_observations" / "clip_001.json"
    result_path = PROJECT_ROOT / "outputs" / "results" / "clip_001_residual_result.json"

    save_clip_observation(build_demo_clip(), clip_path)
    loaded_clip = load_clip_observation(clip_path)
    result = compute_clip_scale_depth_result(loaded_clip)
    save_clip_residual_result(result, result_path)

    print(f"Saved clip observation: {clip_path}")
    print(f"Loaded clip: {loaded_clip.clip_id} from video {loaded_clip.video_id}")
    for frame in loaded_clip.frames:
        print(f"\nFrame {frame.frame_index} ({frame.frame_id})")
        for obj in frame.objects:
            print(
                f"  {obj.object_id}: label={obj.label}, mask_area={obj.mask_area:.1f}, "
                f"depth={obj.depth:.2f}, bbox={obj.bbox}"
            )

    print("\nScale-depth pair residuals")
    for pair in result.pair_residuals:
        print(
            f"  frame={pair['frame_index']}: "
            f"{pair['object_id_a']} -> {pair['object_id_b']}, "
            f"R_sd={pair['scale_depth']:.6f}"
        )
    print(f"\nclip_score = {result.clip_score:.6f}")
    print(f"Saved residual result: {result_path}")


if __name__ == "__main__":
    main()
