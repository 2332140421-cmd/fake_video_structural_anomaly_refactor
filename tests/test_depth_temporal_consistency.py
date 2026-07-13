from __future__ import annotations

from _bootstrap import ensure_project_test_environment

ensure_project_test_environment(__file__)

import csv
import subprocess
import sys
from pathlib import Path

import numpy as np

from semantic3d.depth_temporal_consistency import (
    aggregate_clip_depth_residuals,
    aggregate_track_depth_residuals,
    compute_depth_temporal_residual,
    compute_frame_depth_reference,
    depth_consistency_plot_series_from_csv,
    save_depth_consistency_tracks_plot_from_csv,
    save_depth_consistency_tracks_plot,
    with_frame_indices,
)
from semantic3d.io import load_clip_observation, save_clip_observation
from semantic3d.observations import (
    ClipObservationJSON,
    FrameObservationJSON,
    ObjectObservationJSON,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRAME_AREA = 640.0 * 480.0
REFERENCE = 10.0


def _object(
    object_id: str,
    track_id: str = "trk_000001",
    label: str = "unknown_label_without_scale_prior",
    depth: float = 5.0,
    projection_scale: float = 0.2,
    bbox: list[float] | None = None,
) -> ObjectObservationJSON:
    if bbox is None:
        bbox = [100.0, 100.0, 100.0 + projection_scale * 500.0, 180.0]
    return ObjectObservationJSON(
        object_id=object_id,
        label=label,
        canonical_label=label,
        track_id=track_id,
        mask_area=(projection_scale**2) * FRAME_AREA,
        frame_area=FRAME_AREA,
        depth=depth,
        confidence=1.0,
        bbox=bbox,
    )


def _frame(index: int, objects: list[ObjectObservationJSON]) -> FrameObservationJSON:
    return FrameObservationJSON(
        frame_index=index,
        frame_id=f"frame_{index:06d}",
        width=640,
        height=480,
        objects=objects,
    )


def test_compute_frame_depth_reference_from_depth_map() -> None:
    frame = _frame(0, [_object("obj_f0", depth=5.0)])
    depth_map = np.asarray([[1.0, 2.0, np.nan], [np.inf, -1.0, 4.0]])

    assert compute_frame_depth_reference(frame, depth_map) == 2.0


def test_compute_frame_depth_reference_from_objects() -> None:
    frame = _frame(
        0,
        [
            _object("obj_a_f0", depth=4.0),
            _object("obj_b_f0", depth=6.0),
        ],
    )

    assert compute_frame_depth_reference(frame) == 5.0


def test_stable_object_residual_near_zero() -> None:
    previous = _object("obj_f0", depth=5.0, projection_scale=0.2)
    current = _object("obj_f1", depth=5.02, projection_scale=0.2)

    result = compute_depth_temporal_residual(previous, current, REFERENCE, REFERENCE)

    assert result.valid
    assert result.residual == 0.0


def test_reasonable_approaching_motion_residual_low() -> None:
    previous = _object("obj_f0", depth=8.0, projection_scale=0.10)
    current = _object("obj_f1", depth=4.0, projection_scale=0.20)

    result = compute_depth_temporal_residual(previous, current, REFERENCE, REFERENCE)

    assert result.valid
    assert result.residual < 0.02


def test_depth_jump_residual_high() -> None:
    previous = _object("obj_f0", depth=4.0, projection_scale=0.12)
    current = _object("obj_f1", depth=9.0, projection_scale=0.12)

    result = compute_depth_temporal_residual(previous, current, REFERENCE, REFERENCE)

    assert result.valid
    assert result.residual == max(0.0, result.raw_residual - result.tolerance)
    assert result.residual > 0.6


def test_projection_scale_jump_residual_high() -> None:
    previous = _object("obj_f0", depth=6.0, projection_scale=0.08)
    current = _object("obj_f1", depth=6.0, projection_scale=0.22)

    result = compute_depth_temporal_residual(previous, current, REFERENCE, REFERENCE)

    assert result.valid
    assert result.residual > 0.8


def test_invalid_depth_is_skipped() -> None:
    previous = _object("obj_f0", depth=5.0, projection_scale=0.2)
    current = _object("obj_f1", depth=0.0, projection_scale=0.2)

    result = compute_depth_temporal_residual(previous, current, REFERENCE, REFERENCE)

    assert not result.valid
    assert result.skip_reason == "invalid_current_depth"


def test_missing_frame_depth_reference_is_skipped() -> None:
    frame = _frame(0, [_object("obj_f0", depth=0.0)])

    assert compute_frame_depth_reference(frame) is None


def test_unknown_label_without_scale_prior_can_compute() -> None:
    previous = _object("obj_f0", label="mystery_detector_label")
    current = _object("obj_f1", label="mystery_detector_label")

    result = compute_depth_temporal_residual(previous, current, REFERENCE, REFERENCE)

    assert result.valid
    assert result.label == "mystery_detector_label"


def test_track_and_clip_aggregation() -> None:
    results = [
        with_frame_indices(
            compute_depth_temporal_residual(
                _object("obj_f0", depth=5.0, projection_scale=0.2),
                _object("obj_f1", depth=5.0, projection_scale=0.2),
                REFERENCE,
                REFERENCE,
            ),
            0,
            1,
        ),
        with_frame_indices(
            compute_depth_temporal_residual(
                _object("obj_f1", depth=4.0, projection_scale=0.12),
                _object("obj_f2", depth=9.0, projection_scale=0.12),
                REFERENCE,
                REFERENCE,
            ),
            1,
            2,
        ),
    ]

    track_summary = aggregate_track_depth_residuals(results)[0]
    clip_summary = aggregate_clip_depth_residuals(results, [0, 1, 2], clip_id="clip_0")

    assert track_summary["num_transitions"] == 2
    assert track_summary["max_residual"] > 0.6
    assert clip_summary["clip_id"] == "clip_0"
    assert clip_summary["num_valid_transitions"] == 2


def test_visualization_png_can_be_generated(tmp_path: Path) -> None:
    result = with_frame_indices(
        compute_depth_temporal_residual(
            _object("obj_f0", depth=4.0, projection_scale=0.12),
            _object("obj_f1", depth=9.0, projection_scale=0.12),
            REFERENCE,
            REFERENCE,
        ),
        0,
        1,
    )
    output_path = tmp_path / "depth_consistency_tracks.png"

    save_depth_consistency_tracks_plot([result], output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_pipeline_generates_three_csvs_and_tracked_observations(tmp_path: Path) -> None:
    observation_dir = tmp_path / "observations"
    depth_map_dir = tmp_path / "depth_maps"
    depth_map_dir.mkdir(parents=True)
    depth0 = depth_map_dir / "frame_000000_depth.npy"
    depth1 = depth_map_dir / "frame_000001_depth.npy"
    np.save(depth0, np.full((16, 16), REFERENCE, dtype=np.float32))
    np.save(depth1, np.full((16, 16), REFERENCE, dtype=np.float32))
    frame_area = FRAME_AREA
    frames = [
        FrameObservationJSON(
            frame_index=0,
            frame_id="frame_000000",
            width=640,
            height=480,
            depth_map_path=str(depth0),
            objects=[
                ObjectObservationJSON(
                    object_id="unknown_f0",
                    label="unknown_label_without_scale_prior",
                    mask_area=0.12**2 * frame_area,
                    frame_area=frame_area,
                    depth=4.0,
                    confidence=1.0,
                    bbox=[100, 100, 160, 180],
                )
            ],
        ),
        FrameObservationJSON(
            frame_index=1,
            frame_id="frame_000001",
            width=640,
            height=480,
            depth_map_path=str(depth1),
            objects=[
                ObjectObservationJSON(
                    object_id="unknown_f1",
                    label="unknown_label_without_scale_prior",
                    mask_area=0.12**2 * frame_area,
                    frame_area=frame_area,
                    depth=9.0,
                    confidence=1.0,
                    bbox=[104, 102, 164, 182],
                )
            ],
        ),
    ]
    clip = ClipObservationJSON(
        clip_id="clip_depth_cons",
        video_id="video_depth_cons",
        frame_indices=[0, 1],
        frames=frames,
    )
    save_clip_observation(clip, observation_dir / "clip_depth_cons.json")

    pair_csv = tmp_path / "pairs.csv"
    track_csv = tmp_path / "tracks.csv"
    clip_csv = tmp_path / "clips.csv"
    associated_dir = tmp_path / "tracked"
    png_path = tmp_path / "depth_consistency_tracks.png"
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_depth_consistency_pipeline.py"),
        "--observation_dir",
        str(observation_dir),
        "--depth_map_dir",
        str(depth_map_dir),
        "--output_pair_csv",
        str(pair_csv),
        "--output_track_csv",
        str(track_csv),
        "--output_clip_csv",
        str(clip_csv),
        "--associated_observation_dir",
        str(associated_dir),
        "--visualization_path",
        str(png_path),
        "--iou_threshold",
        "0.1",
        "--center_distance_threshold",
        "0.25",
        "--max_frame_gap",
        "1",
        "--tolerance",
        "0.10",
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True, capture_output=True, text=True)

    assert pair_csv.exists()
    assert track_csv.exists()
    assert clip_csv.exists()
    assert png_path.exists()
    with pair_csv.open("r", encoding="utf-8", newline="") as file:
        pair_rows = list(csv.DictReader(file))
    assert len(pair_rows) == 1
    assert pair_rows[0]["valid"] == "True"
    assert float(pair_rows[0]["residual"]) > 0.6

    tracked_clip = load_clip_observation(
        associated_dir
        / "video_depth_cons"
        / "video_depth_cons_associated_tracks.json"
    )
    assert tracked_clip.frames[0].objects[0].track_id
    assert (
        tracked_clip.frames[0].objects[0].track_id
        == tracked_clip.frames[1].objects[0].track_id
    )


def test_overlapping_clips_are_deduplicated_before_association(tmp_path: Path) -> None:
    observation_dir = tmp_path / "observations"
    frame_area = FRAME_AREA
    frames = [
        _frame(
            index,
            [
                ObjectObservationJSON(
                    object_id=f"person_f{index}",
                    label="person",
                    mask_area=0.2**2 * frame_area,
                    frame_area=frame_area,
                    depth=5.0 + index,
                    confidence=1.0,
                    bbox=[100 + index, 100, 180 + index, 260],
                )
            ],
        )
        for index in range(3)
    ]
    save_clip_observation(
        ClipObservationJSON(
            clip_id="clip_a",
            video_id="video_overlap",
            frame_indices=[0, 1],
            frames=[frames[0], frames[1]],
        ),
        observation_dir / "clip_a.json",
    )
    save_clip_observation(
        ClipObservationJSON(
            clip_id="clip_b",
            video_id="video_overlap",
            frame_indices=[1, 2],
            frames=[frames[1], frames[2]],
        ),
        observation_dir / "clip_b.json",
    )

    pair_csv = tmp_path / "pairs.csv"
    associated_dir = tmp_path / "tracked"
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_depth_consistency_pipeline.py"),
        "--observation_dir",
        str(observation_dir),
        "--depth_map_dir",
        str(tmp_path / "missing_depth_maps"),
        "--output_pair_csv",
        str(pair_csv),
        "--output_track_csv",
        str(tmp_path / "tracks.csv"),
        "--output_clip_csv",
        str(tmp_path / "clips.csv"),
        "--associated_observation_dir",
        str(associated_dir),
        "--visualization_path",
        str(tmp_path / "tracks.png"),
        "--iou_threshold",
        "0.1",
        "--center_distance_threshold",
        "0.25",
    ]
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "duplicate_frame_count: 1" in completed.stdout
    assert "duplicate_track_frame_count: 0" in completed.stdout
    tracked_clip = load_clip_observation(
        associated_dir / "video_overlap" / "video_overlap_associated_tracks.json"
    )
    seen = set()
    for frame in tracked_clip.frames:
        for obj in frame.objects:
            key = ("video_overlap", frame.frame_index, obj.track_id)
            assert key not in seen
            seen.add(key)
    assert [frame.frame_index for frame in tracked_clip.frames] == [0, 1, 2]


def test_invalid_transition_is_not_plotted_as_zero(tmp_path: Path) -> None:
    csv_path = tmp_path / "pairs.csv"
    csv_path.write_text(
        "video_id,track_id,label,previous_frame_index,current_frame_index,"
        "previous_depth,current_depth,previous_depth_reference,current_depth_reference,"
        "previous_relative_depth,current_relative_depth,previous_projection_scale,"
        "current_projection_scale,previous_geometry_state,current_geometry_state,"
        "raw_residual,tolerance,residual,confidence_weight,weighted_residual,"
        "valid,skip_reason\n"
        "video,trk_1,person,0,1,5,0,10,10,0,0,0,0,0,0,0,0.1,0,1,0,False,"
        "invalid_current_depth\n",
        encoding="utf-8",
    )

    series = depth_consistency_plot_series_from_csv(csv_path)
    output_path = tmp_path / "plot.png"
    save_depth_consistency_tracks_plot_from_csv(csv_path, output_path)

    assert series == {}
    assert output_path.exists()


def test_plot_residual_count_equals_valid_transition_count(tmp_path: Path) -> None:
    csv_path = tmp_path / "pairs.csv"
    csv_path.write_text(
        "video_id,track_id,label,previous_frame_index,current_frame_index,"
        "previous_depth,current_depth,previous_depth_reference,current_depth_reference,"
        "previous_relative_depth,current_relative_depth,previous_projection_scale,"
        "current_projection_scale,previous_geometry_state,current_geometry_state,"
        "raw_residual,tolerance,residual,confidence_weight,weighted_residual,"
        "valid,skip_reason\n"
        "video,trk_1,person,0,1,5,6,10,10,0.5,0.6,0.2,0.2,-2,-1.8,0.2,0.1,"
        "0.1,1,0.1,True,\n"
        "video,trk_1,person,1,2,6,7,10,10,0.6,0.7,0.2,0.2,-1.8,-1.7,0.1,"
        "0.1,0,1,0,True,\n"
        "video,trk_1,person,2,3,7,0,10,10,0,0,0,0,0,0,0,0.1,0,1,0,"
        "False,invalid_current_depth\n",
        encoding="utf-8",
    )

    series = depth_consistency_plot_series_from_csv(csv_path)
    residual_count = sum(
        len(track_series["residual"]) for track_series in series.values()
    )

    assert residual_count == 2
