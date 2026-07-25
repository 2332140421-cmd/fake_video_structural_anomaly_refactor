from pathlib import Path

import numpy as np

from data.schemas import ClipObservation, FrameObservation, ObjectObservation
from models.object_semantic import compute_object_semantic_residuals


def _prior(
    path: Path,
    *,
    category: str = "cup",
    dimension: str = "height",
    orientation: str = "upright",
) -> Path:
    path.write_text(
        "metric_scale_priors:\n"
        f"- category: {category}\n"
        f"  dimension: {dimension}\n"
        "  min_meters: 0.8\n"
        "  max_meters: 1.0\n"
        f"  orientation_requirement: {orientation}\n"
        "  minimum_observability: 0.5\n"
        "  source_note: synthetic geometry test\n",
        encoding="utf-8",
    )
    return path


def _clip(
    depths,
    *,
    category="cup",
    valid_depth=True,
    truncated=False,
    occlusion=0.0,
    viewpoint="unknown",
    mask_box=(8, 3, 12, 17),
    mask_quality=1.0,
    track_ids="stable_track",
    clip_id="clip",
):
    mask = np.zeros((20, 20), dtype=bool)
    x1, y1, x2, y2 = mask_box
    mask[y1:y2, x1:x2] = True
    intrinsics = np.array([[10.0, 0.0, 10.0], [0.0, 10.0, 10.0], [0.0, 0.0, 1.0]])
    frames = []
    identities = (
        [track_ids] * len(depths) if isinstance(track_ids, str) else list(track_ids)
    )
    for index, (depth, track_id) in enumerate(zip(depths, identities, strict=True)):
        depth_valid = np.ones((20, 20), dtype=bool)
        if isinstance(valid_depth, bool):
            depth_valid[:] = valid_depth
        else:
            depth_valid[:] = False
            locations = np.argwhere(mask)
            keep = int(round(len(locations) * float(valid_depth)))
            selected = locations[:keep]
            if len(selected):
                depth_valid[selected[:, 0], selected[:, 1]] = True
        obj = ObjectObservation(
            f"object_{index}",
            track_id,
            category,
            tuple(float(value) for value in mask_box),
            1.0,
            instance_mask=mask,
            truncated=truncated,
            occlusion_ratio=occlusion,
            viewpoint=viewpoint,
            mask_quality=mask_quality,
        )
        frames.append(
            FrameObservation(
                "video",
                clip_id,
                index,
                float(index),
                np.zeros((20, 20, 3), dtype=np.uint8),
                [obj],
                np.full((20, 20), depth, dtype=float),
                depth_valid,
                intrinsics=intrinsics,
                confidence={"metric_depth": 1.0},
            )
        )
    return ClipObservation("video", clip_id, frames)


def _semantic(clip, prior):
    return compute_object_semantic_residuals(clip, prior_path=prior)


def test_metric_size_inside_above_and_below_prior(tmp_path):
    prior = _prior(tmp_path / "prior.yaml")
    inside = _semantic(_clip([1.0]), prior)[0]
    above = _semantic(_clip([4.0]), prior)[0]
    below = _semantic(_clip([0.3]), prior)[0]
    assert inside.valid_mask and inside.raw_value < 0.3
    assert above.raw_value > inside.raw_value
    assert below.raw_value > inside.raw_value


def test_semantic_gates_missing_inputs_and_observability(tmp_path):
    prior = _prior(tmp_path / "prior.yaml")
    assert not _semantic(_clip([1.0], valid_depth=False), prior)[0].valid_mask
    assert _semantic(_clip([1.0], category="unknown"), prior)[0].reason == "missing_category_metric_prior"
    truncated = _semantic(
        _clip([1.0], truncated=True, mask_box=(8, 0, 12, 17)), prior
    )[0]
    assert truncated.reason == "dimension_truncated"
    height = next(
        item
        for item in truncated.metadata["dimension_observability"]
        if item["dimension"] == "height"
    )
    assert height["observable"] is False
    assert height["axis_source"] == "robust_metric_surface_principal_axis_xy"
    assert _semantic(_clip([1.0], occlusion=0.9), prior)[0].reason == "severe_object_occlusion"
    assert _semantic(_clip([1.0], valid_depth=0.4), prior)[0].reason == "insufficient_valid_metric_depth_ratio"
    assert _semantic(_clip([1.0], mask_quality=0.1), prior)[0].reason == "insufficient_mask_quality"


def test_vertical_axis_is_observable_without_global_viewpoint(tmp_path):
    row = _semantic(_clip([1.0]), _prior(tmp_path / "prior.yaml"))[0]
    dimensions = {item["dimension"]: item for item in row.metadata["dimension_observability"]}
    assert row.valid_mask
    assert dimensions["height"]["observable"] is True
    assert dimensions["height"]["axis_source"] == "robust_metric_surface_principal_axis_xy"
    assert dimensions["width"]["observable"] is False
    assert dimensions["length"]["observable"] is False
    assert row.metadata["estimated_size_m"] > 0.0
    assert row.metadata["old_object_pair_rsd_used"] is False
    assert row.metadata["authenticity_label_used"] is False


def test_container_axis_need_not_match_camera_y(tmp_path):
    row = _semantic(
        _clip([1.0], mask_box=(3, 8, 17, 12)),
        _prior(tmp_path / "prior.yaml"),
    )[0]
    dimensions = {item["dimension"]: item for item in row.metadata["dimension_observability"]}
    assert row.valid_mask
    assert dimensions["height"]["observable"] is True
    assert dimensions["height"]["axis_diagnostics"]["camera_y_alignment"] < 0.2
    assert dimensions["width"]["observable"] is False
    assert dimensions["length"]["observable"] is False


def test_unreliable_axis_is_missing_not_zero(tmp_path):
    row = _semantic(
        _clip([1.0], mask_box=(5, 5, 15, 15)),
        _prior(tmp_path / "prior.yaml"),
    )[0]
    assert not row.valid_mask
    assert row.reason == "no_reliable_dimension_axis"
    assert np.isnan(row.raw_value)


def test_same_track_temporal_stability_and_jump(tmp_path):
    prior = _prior(tmp_path / "prior.yaml")
    stable = [row for row in _semantic(_clip([1.0, 1.0]), prior) if row.name == "semantic_metric_temporal"][-1]
    jump = [row for row in _semantic(_clip([1.0, 4.0]), prior) if row.name == "semantic_metric_temporal"][-1]
    assert stable.valid_mask and stable.raw_value < 1e-8
    assert jump.valid_mask and jump.raw_value > 1.0


def test_temporal_history_never_mixes_tracks_or_clips(tmp_path):
    prior = _prior(tmp_path / "prior.yaml")
    mixed_tracks = [
        row
        for row in _semantic(_clip([1.0, 1.0], track_ids=["a", "b"]), prior)
        if row.name == "semantic_metric_temporal"
    ]
    assert mixed_tracks and not any(row.valid_mask for row in mixed_tracks)
    first_clip = _semantic(_clip([1.0], clip_id="clip_a"), prior)
    second_clip = _semantic(_clip([4.0], clip_id="clip_b"), prior)
    assert not any(
        row.valid_mask
        for row in first_clip + second_clip
        if row.name == "semantic_metric_temporal"
    )


def test_active_semantic_route_does_not_import_pair_rsd():
    source = (
        Path(__file__).resolve().parents[1] / "models/object_semantic.py"
    ).read_text(encoding="utf-8")
    assert "dimension_aligned_scale_depth" not in source
    assert "scale_depth_residual" not in source
