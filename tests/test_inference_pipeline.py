from pathlib import Path

import numpy as np

from data.observations import build_shared_observations
from data.schemas import ObjectObservation, TrackObservation, VideoClip
from inference.outputs import save_analysis_outputs
from inference.pipeline import ForgeryAnalysisPipeline
from models.providers import LegacyObjectProviderAdapter
from semantic3d.real_object_provider import RealObjectProvider


class ObjectProvider:
    def __init__(self):
        self.calls = 0

    def predict(self, frame, frame_index):
        self.calls += 1
        mask = np.zeros(frame.shape[:2], dtype=bool)
        mask[6:26, 14:18] = True
        return [
            ObjectObservation(
                "object",
                "track",
                "bottle",
                (14, 6, 18, 26),
                1.0,
                instance_mask=mask,
                metadata={"pose_estimate_status": "upright_shape_compatible"},
            )
        ]


class DepthProvider:
    def __init__(self):
        self.calls = 0

    def predict(self, frame, frame_index):
        self.calls += 1
        shape = frame.shape[:2]
        intrinsics = np.array([[20.0, 0.0, 16.0], [0.0, 20.0, 16.0], [0.0, 0.0, 1.0]])
        return np.full(shape, 2.0), np.ones(shape, dtype=bool), None, intrinsics, 1.0


class PoseProvider:
    def __init__(self):
        self.calls = 0

    def estimate(self, *args):
        self.calls += 1
        return np.eye(4), 1.0, "estimated_valid"


class TrackProvider:
    def track(self, clip, frames):
        points = np.array([[0.0, 0.0, 2.0], [0.1, 0.0, 2.0], [0.2, 0.0, 2.0]])
        return [
            TrackObservation(
                "point_track", "object", clip.frame_indices,
                np.array([[16.0, 16.0], [17.0, 16.0], [18.0, 16.0]]),
                points_3d=points,
            )
        ]


def test_detector_keeps_unsupported_exact_category_for_downstream_unavailability():
    detector = RealObjectProvider(
        detector=lambda _: [
            {
                "bbox": [2.0, 2.0, 12.0, 12.0],
                "label": "sports ball",
                "confidence": 0.9,
            }
        ],
        allowed_labels=[],
        skip_unknown_scale_prior=False,
    )
    rows = LegacyObjectProviderAdapter(detector).predict(
        np.zeros((16, 16, 3), dtype=np.uint8),
        0,
    )

    assert len(rows) == 1
    assert rows[0].category == "sports_ball"
    assert rows[0].keypoints_xy is None


def test_paper_core_active_cli_does_not_attach_pose_provider():
    root = Path(__file__).resolve().parents[1]
    cli_source = (root / "inference/cli.py").read_text(encoding="utf-8")
    provider_source = (root / "models/providers.py").read_text(encoding="utf-8")

    assert "RealHumanKeypointProvider(" not in cli_source
    adapter_signature = provider_source.split(
        "class LegacyObjectProviderAdapter:", 1
    )[1].split("def predict", 1)[0]
    assert "keypoint_provider" not in adapter_signature


def test_synthetic_shared_observation_to_outputs(tmp_path):
    prior = tmp_path / "priors.yaml"
    source = tmp_path / "prior_sources.csv"
    source.write_text(
        "derivation_id,source_type,source_title,publisher,source_identifier,"
        "source_version,accessed_at,sample_count,raw_measurements_or_range,"
        "derivation_method,review_status\n"
        "SYNTHETIC_TEST_ONLY,formal_research_dataset,Synthetic geometry fixture,"
        "test suite,fixture,1,2026-07-26,3,synthetic 1.0 to 2.0 m,"
        "fixed unit-test fixture,APPROVED_SOURCE_BACKED\n",
        encoding="utf-8",
    )
    prior.write_text(
        "schema_version: paper_core_scale_priors_v1\n"
        "unit: meter\n"
        "source_table: prior_sources.csv\n"
        "priors:\n"
        "- entry_id: synthetic_bottle_height\n"
        "  class_name: bottle\n"
        "  aliases: []\n"
        "  supported_dimension: height\n"
        "  min_m: 1.0\n"
        "  max_m: 2.0\n"
        "  dimension_definition: Synthetic metric height.\n"
        "  applicable_scope: Synthetic complete bottle fixture.\n"
        "  excluded_scope: All non-test observations.\n"
        "  confidence: high\n"
        "  minimum_observability: 0.5\n"
        "  derivation_id: SYNTHETIC_TEST_ONLY\n"
        "unsupported_classes: []\n",
        encoding="utf-8",
    )
    frames = tuple(np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(3))
    video_clip = VideoClip("clip", "video", (0, 1, 2), (0.0, 0.1, 0.2), frames)
    objects, depth, pose = ObjectProvider(), DepthProvider(), PoseProvider()
    observation = build_shared_observations(
        video_clip,
        object_provider=objects,
        depth_provider=depth,
        pose_provider=pose,
        track_provider=TrackProvider(),
    )
    assert (objects.calls, depth.calls, pose.calls) == (3, 3, 2)
    config = {
        "video": {"clip_length": 3, "clip_stride": 2, "resize": None},
        "object_semantic": {
            "prior_path": str(prior),
            "canonical_axis_path": str(
                Path(__file__).resolve().parents[1]
                / "configs/canonical_axis_v1.yaml"
            ),
            "min_depth_coverage": 0.5,
            "max_occlusion_ratio": 0.5,
            "min_mask_quality": 0.3,
        },
        "fusion": {"suspicious_clip_threshold": 0.0, "merge_gap_frames": 1},
    }
    pipeline = ForgeryAnalysisPipeline(
        config=config,
        object_provider=objects,
        depth_provider=depth,
        pose_provider=pose,
        track_provider=TrackProvider(),
    )
    result = pipeline.analyze_observations([observation])
    assert result.timeline and result.suspicious_clips
    assert result.object_scores and result.track_scores
    assert result.metadata["historical_csv_read"] is False
    assert result.metadata["authenticity_label_used"] is False
    assert (
        result.metadata["canonical_threshold_config_sha256"]
        == "78aedb8863999d17bddce1289828a1afdb98fc9ebf0092e047c7134f5e02a4dc"
    )
    assert result.metadata["bottle_source_runtime_dimension_match"] is True
    assert result.metadata["m6_to_a2_bridge_called"] is False
    assert result.metadata["object_semantic_funnel"]["objects_total"] == 3
    assert result.metadata["object_semantic_funnel"]["objects_with_instance_mask"] == 3
    assert result.metadata["object_semantic_funnel"]["objects_with_dimension_axis"] == 3
    assert result.metadata["object_semantic_funnel"]["objects_with_viewpoint_evidence"] == 3
    assert result.metadata["object_semantic_funnel"]["observable_height"] == 3
    assert result.metadata["object_semantic_funnel"]["observable_width"] == 0
    assert result.metadata["object_semantic_funnel"]["observable_length"] == 0
    assert result.metadata["object_semantic_funnel"]["objects_with_any_observable_dimension"] == 3
    assert result.metadata["object_semantic_funnel"]["objects_with_semantic_prior_residual"] == 3
    assert result.metadata["branch_evidence_counts"]["semantic_prior"]["total"] == 3
    assert sum(result.metadata["visibility_state_counts"].values()) == 0
    assert result.metadata["point_track_diagnostics"]["index_alignment_ok"] is True
    assert result.metadata["point_track_diagnostics"]["track_ids_unique"] is True
    paths = save_analysis_outputs(result, [observation], tmp_path / "outputs", heatmap_sigma=1.0)
    required = {
        "result", "timeline", "clip_scores", "object_scores", "track_scores",
        "suspicious_clips", "abnormal_tracks", "structural_heatmap",
    }
    assert required <= paths.keys()
    assert all(paths[name].is_file() and paths[name].stat().st_size > 0 for name in required)
