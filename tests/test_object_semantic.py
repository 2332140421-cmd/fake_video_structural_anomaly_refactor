from pathlib import Path

import numpy as np
import pytest

from data.schemas import ClipObservation, FrameObservation, ObjectObservation
from models.object_semantic import (
    _mask_completeness_support,
    compute_object_semantic_residuals,
    load_canonical_axis_registry,
    load_scale_prior_registry,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_AXIS_PATH = PROJECT_ROOT / "configs/canonical_axis_v1.yaml"


def _prior(
    path: Path,
    *,
    category: str = "bottle",
    dimension: str = "height",
    orientation: str = "upright",
) -> Path:
    source = path.with_name(f"{path.stem}_sources.csv")
    source.write_text(
        "derivation_id,source_type,source_title,publisher,source_identifier,"
        "source_version,accessed_at,sample_count,raw_measurements_or_range,"
        "derivation_method,review_status\n"
        "SYNTHETIC_TEST_ONLY,formal_research_dataset,Synthetic geometry fixture,"
        "test suite,fixture,1,2026-07-26,3,synthetic 0.8 to 1.0 m,"
        "fixed unit-test fixture,APPROVED_SOURCE_BACKED\n",
        encoding="utf-8",
    )
    path.write_text(
        "schema_version: paper_core_scale_priors_v1\n"
        "unit: meter\n"
        f"source_table: {source.name}\n"
        "priors:\n"
        "- entry_id: synthetic_test_prior\n"
        f"  class_name: {category}\n"
        "  aliases: []\n"
        f"  supported_dimension: {dimension}\n"
        "  min_m: 0.8\n"
        "  max_m: 1.0\n"
        "  dimension_definition: Synthetic metric dimension.\n"
        f"  applicable_scope: Synthetic {orientation} geometry fixture.\n"
        "  excluded_scope: All non-test observations.\n"
        "  confidence: high\n"
        "  minimum_observability: 0.5\n"
        "  derivation_id: SYNTHETIC_TEST_ONLY\n"
        "unsupported_classes: []\n",
        encoding="utf-8",
    )
    return path


def _clip(
    depths,
    *,
    category="bottle",
    valid_depth=True,
    truncated=False,
    occlusion=0.0,
    viewpoint="unknown",
    mask_box=(8, 3, 12, 17),
    mask_quality=1.0,
    track_ids="stable_track",
    clip_id="clip",
    pose_status="upright_shape_compatible",
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
            metadata={"pose_estimate_status": pose_status},
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


def _semantic(clip, prior, *, canonical_axis_path=CANONICAL_AXIS_PATH):
    return compute_object_semantic_residuals(
        clip,
        prior_path=prior,
        canonical_axis_path=canonical_axis_path,
    )


def test_mask_completeness_uses_discrete_bbox_intersection():
    mask = np.zeros((8, 10), dtype=bool)
    mask[2:5, 3:7] = True
    support = _mask_completeness_support(mask, (2.0, 1.0, 8.0, 6.0), mask.shape)

    assert support["valid"]
    assert support["mask_area_total"] == 12
    assert support["mask_area_inside_bbox"] == 12
    assert support["bbox_area_clipped"] == 30
    assert support["mask_completeness"] == 12 / 30
    assert 0.0 <= support["mask_completeness"] <= 1.0


def test_mask_completeness_excludes_mask_spill_outside_bbox():
    mask = np.zeros((8, 10), dtype=bool)
    mask[1:7, 1:9] = True
    support = _mask_completeness_support(mask, (3.0, 2.0, 7.0, 6.0), mask.shape)

    assert support["valid"]
    assert support["mask_area_total"] == 48
    assert support["mask_area_inside_bbox"] == 16
    assert support["bbox_area_clipped"] == 16
    assert support["mask_spill_area"] == 32
    assert support["mask_completeness"] == 1.0
    assert support["legacy_total_mask_over_raw_bbox_ratio"] == 3.0
    assert support["mask_completeness"] != support["legacy_total_mask_over_raw_bbox_ratio"]


def test_mask_completeness_clips_bbox_before_using_pixel_area():
    mask = np.zeros((8, 10), dtype=bool)
    mask[0:3, 0:4] = True
    support = _mask_completeness_support(mask, (-2.2, -1.4, 4.0, 3.0), mask.shape)

    assert support["valid"]
    assert support["bbox_clipped_xyxy"] == [0, 0, 4, 3]
    assert support["bbox_area_clipped"] == 12
    assert support["mask_area_inside_bbox"] == 12
    assert support["mask_completeness"] == 1.0


def test_mask_completeness_rejects_empty_bbox_and_shape_mismatch():
    mask = np.ones((8, 10), dtype=bool)
    empty = _mask_completeness_support(mask, (4.0, 3.0, 4.0, 6.0), mask.shape)
    mismatch = _mask_completeness_support(
        np.ones((7, 10), dtype=bool),
        (1.0, 1.0, 5.0, 5.0),
        mask.shape,
    )

    assert not empty["valid"]
    assert empty["reason"] == "invalid_clipped_bbox"
    assert np.isnan(empty["mask_completeness"])
    assert not mismatch["valid"]
    assert mismatch["reason"] == "mask_shape_mismatch"
    assert np.isnan(mismatch["mask_completeness"])


def test_invalid_mask_support_becomes_explicit_semantic_unavailable(tmp_path):
    prior = _prior(tmp_path / "prior.yaml")
    empty_bbox_clip = _clip([1.0])
    empty_bbox_clip.frames[0].objects[0].bbox_xyxy = (4.0, 3.0, 4.0, 6.0)
    mismatched_mask_clip = _clip([1.0])
    mismatched_mask_clip.frames[0].objects[0].instance_mask = np.ones(
        (19, 20), dtype=bool
    )

    empty = _semantic(empty_bbox_clip, prior)[0]
    mismatch = _semantic(mismatched_mask_clip, prior)[0]
    assert not empty.valid_mask and np.isnan(empty.raw_value)
    assert empty.reason == "invalid_clipped_bbox"
    assert not mismatch.valid_mask and np.isnan(mismatch.raw_value)
    assert mismatch.reason == "mask_shape_mismatch"


def test_mask_completeness_is_label_blind():
    mask = np.zeros((8, 10), dtype=bool)
    mask[2:5, 3:7] = True
    real_input = _mask_completeness_support(mask, (2, 1, 8, 6), mask.shape)
    fake_input = _mask_completeness_support(mask.copy(), (2, 1, 8, 6), mask.shape)

    assert real_input == fake_input


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
    assert truncated.reason == "BOTTLE_TRUNCATED"
    height = next(
        item
        for item in truncated.metadata["dimension_observability"]
        if item["dimension"] == "height"
    )
    assert height["observable"] is False
    assert height["axis_source"] == ""
    assert _semantic(_clip([1.0], occlusion=0.9), prior)[0].reason == "severe_object_occlusion"
    assert _semantic(_clip([1.0], valid_depth=0.4), prior)[0].reason == "insufficient_valid_metric_depth_ratio"
    assert _semantic(_clip([1.0], mask_quality=0.1), prior)[0].reason == "insufficient_mask_quality"


def test_bottle_canonical_axis_is_observable_without_global_viewpoint(tmp_path):
    row = _semantic(_clip([1.0]), _prior(tmp_path / "prior.yaml"))[0]
    dimensions = {item["dimension"]: item for item in row.metadata["dimension_observability"]}
    assert row.valid_mask
    assert dimensions["height"]["observable"] is True
    assert (
        dimensions["height"]["axis_source"]
        == "robust_visible_surface_pca_3d_primary_axis"
    )
    assert dimensions["width"]["observable"] is False
    assert dimensions["length"]["observable"] is False
    assert row.metadata["estimated_size_m"] > 0.0
    assert row.metadata["old_object_pair_rsd_used"] is False
    assert row.metadata["authenticity_label_used"] is False


def test_horizontal_bottle_axis_need_not_match_camera_y(tmp_path):
    row = _semantic(
        _clip([1.0], mask_box=(3, 8, 17, 12)),
        _prior(tmp_path / "prior.yaml"),
    )[0]
    dimensions = {item["dimension"]: item for item in row.metadata["dimension_observability"]}
    assert row.valid_mask
    assert dimensions["height"]["observable"] is True
    assert abs(dimensions["height"]["axis_vector"][0]) > 0.9
    assert dimensions["width"]["observable"] is False
    assert dimensions["length"]["observable"] is False


def test_unreliable_axis_is_missing_not_zero(tmp_path):
    row = _semantic(
        _clip([1.0], mask_box=(5, 5, 15, 15)),
        _prior(tmp_path / "prior.yaml"),
    )[0]
    assert not row.valid_mask
    assert row.reason == "BOTTLE_NOT_ELONGATED_IN_3D"
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


def test_empty_formal_prior_table_has_no_prior_evidence_or_hidden_fallback(tmp_path):
    source = tmp_path / "sources.csv"
    source.write_text(
        "derivation_id,source_type,source_title,publisher,source_identifier,"
        "source_version,accessed_at,sample_count,raw_measurements_or_range,"
        "derivation_method,review_status\n",
        encoding="utf-8",
    )
    prior = tmp_path / "empty.yaml"
    prior.write_text(
        "schema_version: paper_core_scale_priors_v1\n"
        "unit: meter\n"
        "source_table: sources.csv\n"
        "priors: []\n"
        "unsupported_classes: []\n",
        encoding="utf-8",
    )

    registry = load_scale_prior_registry(prior)
    rows = _semantic(_clip([1.0, 1.0], category="cup"), prior)
    prior_rows = [row for row in rows if row.name == "semantic_metric_prior"]
    temporal_rows = [row for row in rows if row.name == "semantic_metric_temporal"]

    assert not registry.priors_by_label
    assert all(not row.valid_mask for row in prior_rows)
    assert {row.reason for row in prior_rows} == {"missing_category_metric_prior"}
    assert temporal_rows[-1].valid_mask
    assert temporal_rows[-1].metadata["category_prior_required"] is False


def test_formal_prior_loads_only_with_approved_source_and_exact_alias(tmp_path):
    prior = _prior(tmp_path / "prior.yaml", category="person")
    payload = prior.read_text(encoding="utf-8").replace(
        "  aliases: []", "  aliases: [pedestrian]"
    )
    prior.write_text(payload, encoding="utf-8")
    registry = load_scale_prior_registry(prior)

    assert registry.resolve("person") is registry.resolve("pedestrian")
    assert registry.resolve("Person") is None
    assert registry.resolve("person ") is None
    assert registry.resolve("pedestr") is None


def test_unapproved_or_missing_source_derivation_is_rejected(tmp_path):
    prior = _prior(tmp_path / "prior.yaml")
    source = tmp_path / "prior_sources.csv"
    source.write_text(
        source.read_text(encoding="utf-8").replace(
            "APPROVED_SOURCE_BACKED", "UNREVIEWED"
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not source-approved"):
        load_scale_prior_registry(prior)

    source.write_text(
        source.read_text(encoding="utf-8").replace("UNREVIEWED", "APPROVED_SOURCE_BACKED"),
        encoding="utf-8",
    )
    prior.write_text(
        prior.read_text(encoding="utf-8").replace(
            "SYNTHETIC_TEST_ONLY\nunsupported_classes",
            "MISSING_DERIVATION\nunsupported_classes",
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="no approved source derivation"):
        load_scale_prior_registry(prior)


def test_unsupported_and_broad_categories_remain_unavailable(tmp_path):
    prior = _prior(tmp_path / "prior.yaml")
    prior.write_text(
        prior.read_text(encoding="utf-8").replace(
            "unsupported_classes: []",
            "unsupported_classes:\n"
            "- class_name: sports ball\n"
            "  aliases: []\n"
            "  reason: CATEGORY_TOO_BROAD_WITHOUT_SUBTYPE",
        ),
        encoding="utf-8",
    )
    registry = load_scale_prior_registry(prior)

    assert registry.resolve("sports ball") is None
    assert registry.resolve("basketball") is None
    assert (
        registry.unsupported_reason("sports ball")
        == "CATEGORY_TOO_BROAD_WITHOUT_SUBTYPE"
    )
    row = _semantic(_clip([1.0], category="sports ball"), prior)[0]
    assert not row.valid_mask
    assert row.reason == "category_too_broad_without_subtype"


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ("unit: meter", ""),
        ("unit: centimeter", "unit must be exactly"),
    ],
)
def test_formal_prior_unit_is_meter(tmp_path, replacement, message):
    prior = _prior(tmp_path / "prior.yaml")
    prior.write_text(
        prior.read_text(encoding="utf-8").replace("unit: meter", replacement),
        encoding="utf-8",
    )
    if message:
        with pytest.raises(ValueError, match=message):
            load_scale_prior_registry(prior)
    else:
        assert load_scale_prior_registry(prior).resolve("bottle") is not None


def test_formal_prior_requires_strict_increasing_metric_interval(tmp_path):
    prior = _prior(tmp_path / "prior.yaml")
    prior.write_text(
        prior.read_text(encoding="utf-8").replace("  max_m: 1.0", "  max_m: 0.8"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="min_m < max_m"):
        load_scale_prior_registry(prior)


def test_prior_does_not_bypass_dimension_observability(tmp_path):
    row = _semantic(
        _clip([1.0], truncated=True, mask_box=(8, 0, 12, 17)),
        _prior(tmp_path / "prior.yaml"),
    )[0]
    assert not row.valid_mask
    assert row.reason == "BOTTLE_TRUNCATED"
    assert np.isnan(row.raw_value)


def test_semantic_temporal_is_prior_independent_and_label_blind(tmp_path):
    prior = _prior(tmp_path / "prior.yaml", category="other_category")
    clip = _clip([1.0, 1.0], category="cup")
    for frame, label in zip(clip.frames, ("real", "fake"), strict=True):
        frame.objects[0].metadata["authenticity_label"] = label
    rows = _semantic(clip, prior)
    prior_rows = [row for row in rows if row.name == "semantic_metric_prior"]
    temporal_rows = [row for row in rows if row.name == "semantic_metric_temporal"]

    assert all(not row.valid_mask for row in prior_rows)
    assert temporal_rows[-1].valid_mask
    assert temporal_rows[-1].raw_value < 1e-8
    assert temporal_rows[-1].metadata["authenticity_label_used"] is False
    assert temporal_rows[-1].metadata["category_prior_required"] is False


def test_bottle_canonical_prior_and_principal_temporal_are_independent(
    tmp_path,
):
    prior = _prior(tmp_path / "prior.yaml", category="bottle")
    rows = _semantic(
        _clip([1.0, 1.0], category="bottle", pose_status="unavailable"),
        prior,
    )
    prior_rows = [row for row in rows if row.name == "semantic_metric_prior"]
    temporal_rows = [row for row in rows if row.name == "semantic_metric_temporal"]
    dimensions = {
        item["dimension"]: item
        for item in prior_rows[-1].metadata["dimension_observability"]
    }

    assert all(row.valid_mask for row in prior_rows)
    assert dimensions["height"]["observable"] is True
    assert dimensions["principal_extent"]["observable"] is True
    assert temporal_rows[-1].valid_mask
    assert temporal_rows[-1].metadata["dimension"] == "principal_extent"
    assert (
        temporal_rows[-1].metadata["legacy_container_upright_fallback_used"]
        is False
    )


def test_active_semantic_route_does_not_import_pair_rsd():
    source = (
        Path(__file__).resolve().parents[1] / "models/object_semantic.py"
    ).read_text(encoding="utf-8")
    assert "dimension_aligned_scale_depth" not in source
    assert "scale_depth_residual" not in source


def test_person_height_reports_missing_active_pose_provider(tmp_path):
    clip = _clip([1.0], category="person", mask_box=(5, 3, 16, 17))
    row = _semantic(
        clip,
        _prior(tmp_path / "person.yaml", category="person"),
    )[0]
    record = next(
        item
        for item in row.metadata["dimension_observability"]
        if item["dimension"] == "height"
    )

    assert not row.valid_mask
    assert row.reason == "PERSON_POSE_PROVIDER_MISSING"
    assert np.isnan(row.raw_value)
    assert record["observable"] is False
    assert record["axis_source"] == ""
    assert record["axis_vector"] == []


def test_person_missing_pose_never_uses_bbox_mask_axis_or_body_ratio(tmp_path):
    clip = _clip([1.0], category="person", mask_box=(5, 3, 16, 17))
    clip.frames[0].objects[0].metadata["authenticity_label"] = "fake"
    row = _semantic(
        clip,
        _prior(tmp_path / "person_no_fallback.yaml", category="person"),
    )[0]
    record = next(
        item
        for item in row.metadata["dimension_observability"]
        if item["dimension"] == "height"
    )
    diagnostics = record["axis_diagnostics"]

    assert not row.valid_mask
    assert row.reason == "PERSON_POSE_PROVIDER_MISSING"
    assert diagnostics["bbox_height_used"] is False
    assert diagnostics["mask_image_axis_used"] is False
    assert diagnostics["human_ratio_extrapolation_used"] is False
    assert row.metadata["authenticity_label_used"] is False


@pytest.mark.parametrize(
    ("instance_mask", "mask_quality"),
    [
        (None, 1.0),
        ("present", 0.1),
    ],
)
def test_person_prior_reason_is_never_overridden_by_input_quality(
    instance_mask,
    mask_quality,
    tmp_path,
):
    clip = _clip(
        [1.0],
        category="person",
        mask_box=(5, 3, 16, 17),
        mask_quality=mask_quality,
    )
    if instance_mask is None:
        clip.frames[0].objects[0].instance_mask = None
    row = _semantic(
        clip,
        _prior(tmp_path / "person_fixed_reason.yaml", category="person"),
    )[0]

    assert not row.valid_mask
    assert row.reason == "PERSON_POSE_PROVIDER_MISSING"
    assert np.isnan(row.raw_value)


def test_bottle_3d_pca_provenance_and_horizontal_mapping(tmp_path):
    row = _semantic(
        _clip([1.0], mask_box=(3, 8, 17, 12)),
        _prior(tmp_path / "horizontal_bottle.yaml"),
    )[0]

    assert row.valid_mask
    assert row.metadata["canonical_dimension"] == "height"
    assert row.metadata["principal_extent"] > row.metadata["secondary_extent"]
    assert row.metadata["elongation_ratio"] >= 1.8
    assert row.metadata["canonical_mapping_rule"].startswith("exact_bottle_")
    assert abs(row.metadata["axis_vector"][0]) > 0.9


def test_spherical_or_truncated_bottle_is_unavailable_not_zero(tmp_path):
    prior = _prior(tmp_path / "strict_bottle.yaml")
    spherical = _semantic(
        _clip([1.0], mask_box=(5, 5, 15, 15)),
        prior,
    )[0]
    truncated = _semantic(
        _clip([1.0], truncated=True, mask_box=(8, 0, 12, 17)),
        prior,
    )[0]

    assert not spherical.valid_mask
    assert spherical.reason == "BOTTLE_NOT_ELONGATED_IN_3D"
    assert np.isnan(spherical.raw_value)
    assert not truncated.valid_mask
    assert truncated.reason == "BOTTLE_TRUNCATED"


@pytest.mark.parametrize("category", ["cup", "vase"])
def test_other_container_classes_never_map_to_bottle(category, tmp_path):
    row = _semantic(
        _clip([1.0], category=category),
        _prior(tmp_path / f"{category}.yaml"),
    )[0]
    assert not row.valid_mask
    assert row.reason in {
        "missing_category_metric_prior",
        "category_too_broad_without_subtype",
    }
    assert row.metadata["canonical_mapping_rule"] == ""


def test_car_longest_pca_axis_does_not_imply_length(tmp_path):
    row = _semantic(
        _clip(
            [1.0],
            category="car",
            mask_box=(2, 8, 18, 12),
            viewpoint="side",
        ),
        _prior(tmp_path / "car.yaml", category="car", dimension="length"),
    )[0]

    assert not row.valid_mask
    assert row.reason == "CAR_INDEPENDENT_VIEWPOINT_AXIS_UNAVAILABLE"
    assert row.metadata["axis_source"] == ""
    assert "independent" in row.metadata["canonical_mapping_rule"]


def test_formal_prior_values_and_hashes_are_frozen():
    root = Path(__file__).resolve().parents[1]
    registry = load_scale_prior_registry(root / "configs/scale_priors_v1.yaml")

    assert (
        registry.prior_sha256
        == "6ce420b0af6ac913569f3d5599231dc2f89e0e7d180d1c1cec3180abfd8b055b"
    )
    assert (
        registry.source_table_sha256
        == "191a79154fb529fdd304f8cd7198c3c9f22c173e494eaff5e6ff43a27044bdc8"
    )
    assert (
        registry.resolve("person").min_meters,
        registry.resolve("person").max_meters,
    ) == (1.50, 1.99)
    assert (
        registry.resolve("bottle").min_meters,
        registry.resolve("bottle").max_meters,
    ) == (0.137, 0.339)
    assert (
        registry.resolve("car").min_meters,
        registry.resolve("car").max_meters,
    ) == (2.94, 4.92)


def test_authenticity_label_does_not_change_canonical_axis_mapping(tmp_path):
    prior = _prior(tmp_path / "label_blind_bottle.yaml")
    real_clip = _clip([1.0])
    fake_clip = _clip([1.0])
    real_clip.frames[0].objects[0].metadata["authenticity_label"] = "real"
    fake_clip.frames[0].objects[0].metadata["authenticity_label"] = "fake"
    real = _semantic(real_clip, prior)[0]
    fake = _semantic(fake_clip, prior)[0]

    assert real.valid_mask and fake.valid_mask
    assert real.raw_value == pytest.approx(fake.raw_value)
    assert real.metadata["axis_vector"] == fake.metadata["axis_vector"]
    assert real.metadata["authenticity_label_used"] is False
    assert fake.metadata["authenticity_label_used"] is False


def test_canonical_axis_v1_is_frozen_and_source_runtime_dimension_matched():
    registry = load_canonical_axis_registry(CANONICAL_AXIS_PATH)

    assert registry.schema_version == "paper_core_canonical_axis_v1"
    assert (
        registry.config_sha256
        == "78aedb8863999d17bddce1289828a1afdb98fc9ebf0092e047c7134f5e02a4dc"
    )
    assert registry.bottle_source_runtime_dimension_match is True
    assert registry.bottle_source_field == "Sequence.objects[].scale[1]"
    assert "base-to-cap" in registry.bottle_source_dimension_definition
    assert "base-to-cap" in registry.bottle_runtime_dimension_definition
    assert registry.thresholds["bottle_eigenvalue_ratio_min"] == 2.0
    assert registry.thresholds["bottle_extent_ratio_min"] == 1.8
    assert registry.thresholds["bottle_point_support_min"] == 20
    assert registry.thresholds["bottle_depth_coverage_min"] == 0.5
    assert registry.thresholds["bottle_axis_stability_min"] == 0.6
    assert registry.thresholds["bottle_track_axis_compatibility_min"] == 0.8
    assert registry.thresholds["truncation_threshold"] == 0.0
    assert registry.thresholds["occlusion_threshold"] == 0.5
    assert registry.thresholds["robust_projection_low_quantile"] == 0.05
    assert registry.thresholds["robust_projection_high_quantile"] == 0.95


def test_bottle_source_runtime_dimension_mismatch_is_hard_rejected(tmp_path):
    config = tmp_path / "canonical_axis_mismatch.yaml"
    config.write_text(
        CANONICAL_AXIS_PATH.read_text(encoding="utf-8").replace(
            "    ready: true",
            "    ready: false",
            1,
        ),
        encoding="utf-8",
    )
    row = _semantic(
        _clip([1.0]),
        _prior(tmp_path / "bottle_prior.yaml"),
        canonical_axis_path=config,
    )[0]

    assert not row.valid_mask
    assert row.reason == "BOTTLE_SOURCE_RUNTIME_DIMENSION_MISMATCH"
    assert np.isnan(row.raw_value)
    assert row.metadata["source_runtime_dimension_match"] is False


def test_bottle_canonical_provenance_contains_frozen_geometry_fields(tmp_path):
    row = _semantic(
        _clip([1.0], mask_box=(3, 8, 17, 12)),
        _prior(tmp_path / "provenance_bottle.yaml"),
    )[0]

    assert row.valid_mask
    assert row.metadata["class_name"] == "bottle"
    assert row.metadata["canonical_dimension"] == "height"
    assert row.metadata["axis_source"] == "robust_visible_surface_pca_3d_primary_axis"
    assert len(row.metadata["axis_vector"]) == 3
    assert row.metadata["principal_extent"] > row.metadata["secondary_extent"]
    assert row.metadata["eigenvalue_ratio"] >= 2.0
    assert row.metadata["extent_ratio"] >= 1.8
    assert row.metadata["axis_point_support"] >= 20
    assert row.metadata["axis_stability"] >= 0.6
    assert row.metadata["estimated_size_m"] > 0.0
    assert row.metadata["prior_min_m"] < row.metadata["prior_max_m"]
    assert np.isfinite(row.metadata["residual"])
    assert row.metadata["confidence"] > 0.0
    assert row.metadata["visible_surface_only"] is True
    assert row.metadata["scale_prior_entry_id"] == "synthetic_test_prior"
    assert row.metadata["scale_prior_sha256"]
    assert row.metadata["scale_prior_source_table_sha256"]
    assert (
        row.metadata["canonical_threshold_config_sha256"]
        == "78aedb8863999d17bddce1289828a1afdb98fc9ebf0092e047c7134f5e02a4dc"
    )
    assert row.metadata["authenticity_label_used"] is False
