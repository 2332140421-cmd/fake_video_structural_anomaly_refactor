"""D3 facade: build relation graphs from shared metric object surfaces."""

from __future__ import annotations

import numpy as np

from data.schemas import ClipObservation, ResidualEvidence
from semantic3d.d3 import (
    D3GraphNode,
    D3NodeType,
    D3StructureResidualExecutor,
    D3TransitionContext,
    build_d3_frame_graph,
    classify_occlusion_event,
    compute_reappearance_residual,
)
from semantic3d.pose_d2.contracts import PoseProviderStatus
from semantic3d.occlusion.reappearance import evaluate_reappearance
from semantic3d.occlusion.visibility_residual import (
    compute_visibility_explanation_residual,
)
from semantic3d.occlusion.visibility_state import ObjectVisibilityObservation
from .geometry import transform_points


def compute_relation_residuals(clip: ClipObservation) -> list[ResidualEvidence]:
    graphs = []
    transform_frame_from_clip = np.eye(4)
    for frame in clip.frames:
        if frame is not clip.frames[0]:
            if frame.relative_pose_from_previous is None:
                transform_frame_from_clip = None
            elif transform_frame_from_clip is not None:
                transform_frame_from_clip = (
                    frame.relative_pose_from_previous @ transform_frame_from_clip
                )
        nodes = []
        for obj in frame.objects:
            points = obj.metric_surface_xyz
            if points is None or not len(points) or transform_frame_from_clip is None:
                continue
            aligned = transform_points(points, np.linalg.inv(transform_frame_from_clip))
            center = tuple(float(value) for value in np.median(aligned, axis=0))
            nodes.append(
                D3GraphNode(
                    node_id=f"object:{obj.track_id}",
                    node_type=D3NodeType.OBJECT_NODE,
                    frame_index=frame.frame_index,
                    object_id=obj.object_id,
                    track_id=obj.track_id,
                    semantic_label=obj.category,
                    xyz_m=center,
                    coordinate_frame="clip_local_aligned",
                    source_observation_id=f"{clip.clip_id}:{frame.frame_index}:{obj.object_id}",
                    confidence=min(obj.confidence, obj.mask_quality),
                    identity_reliable=obj.track_identity_stable,
                    visibility="visible",
                    valid=True,
                    localization_reference={
                        "kind": "object_mask",
                        "mask": obj.instance_mask,
                        "frame_index": frame.frame_index,
                        "object_id": obj.object_id,
                        "track_id": obj.track_id,
                    },
                )
            )
        graphs.append(
            build_d3_frame_graph(
                graph_id=f"{clip.clip_id}:{frame.frame_index}",
                video_id=clip.video_id,
                clip_id=clip.clip_id,
                frame_index=frame.frame_index,
                nodes=nodes,
                pose_source="paper_core_clip_local_pose",
            )
        )
    output: list[ResidualEvidence] = []
    executor = D3StructureResidualExecutor()
    for previous, current, frame in zip(graphs, graphs[1:], clip.frames[1:]):
        valid = bool(frame.relative_pose_from_previous is not None and previous.valid and current.valid)
        context = D3TransitionContext(
            video_id=clip.video_id,
            clip_id=clip.clip_id,
            frame_t=previous.frame_index,
            frame_t1=current.frame_index,
            pose_status=(
                PoseProviderStatus.ESTIMATED_VALID
                if valid
                else PoseProviderStatus.BLOCKED_BY_CORRESPONDENCE
            ),
            pose_confidence=frame.confidence.get("relative_pose", 0.0) if valid else 0.0,
            pose_valid=valid,
            correspondence_identity_reliable=valid,
            source_coordinate_frame="clip_local_aligned",
            target_coordinate_frame="clip_local_aligned",
            valid=valid,
            failure_reason="" if valid else "blocked_by_pose_or_correspondence",
        )
        for item in executor.compare_graphs(previous, current, context):
            if item.valid:
                output.append(
                    ResidualEvidence.observed(
                        "relation",
                        "object_pair",
                        item.value,
                        confidence=item.confidence,
                        spatial_support=item.localization_reference,
                        temporal_support={"frame_index": item.frame_t1},
                        metadata={"d3_residual_name": item.residual_name},
                    )
                )
            else:
                output.append(
                    ResidualEvidence.unavailable(
                        "relation",
                        "object_pair",
                        item.failure_reason,
                        spatial_support=item.localization_reference,
                        temporal_support={"frame_index": item.frame_t1},
                    )
                )
    for frame in clip.frames:
        formal_visibility = set()
        for track_id, payload in frame.visibility_observations.items():
            formal_visibility.add(track_id)
            observation = (
                payload
                if isinstance(payload, ObjectVisibilityObservation)
                else ObjectVisibilityObservation(**payload)
            )
            result = compute_visibility_explanation_residual(observation)
            support = {
                "kind": "track",
                "track_id": track_id,
                "frame_index": frame.frame_index,
            }
            if result.residual_evidence.valid:
                output.append(
                    ResidualEvidence.observed(
                        "occlusion",
                        "track",
                        result.residual_evidence.value,
                        confidence=result.residual_evidence.quality,
                        spatial_support=support,
                        temporal_support={"frame_index": frame.frame_index},
                        metadata={"explanation": result.explanation.value},
                    )
                )
            else:
                output.append(
                    ResidualEvidence.unavailable(
                        "occlusion",
                        "track",
                        result.residual_evidence.missing_reason,
                        availability=(
                            "not_applicable"
                            if result.explanation.value == "no_visibility_event"
                            else "blocked_by_input"
                        ),
                        spatial_support=support,
                        temporal_support={"frame_index": frame.frame_index},
                        metadata={"explanation": result.explanation.value},
                    )
                )
        for track_id, state in frame.occlusion_states.items():
            if track_id.startswith("_") or track_id in formal_visibility:
                continue
            output.append(
                ResidualEvidence.unavailable(
                    "occlusion",
                    "object",
                    "event_state_requires_formal_d3_event_inputs",
                    availability="not_applicable" if state == "none" else "blocked_by_input",
                    temporal_support={"frame_index": frame.frame_index, "track_id": track_id},
                )
            )
        formal_reappearance = set()
        for payload in frame.reappearance_observations:
            result = evaluate_reappearance(**payload)
            track_id = result.observation.candidate_object_track_id
            formal_reappearance.add(track_id)
            support = {
                "kind": "track",
                "track_id": track_id,
                "frame_index": frame.frame_index,
            }
            if result.evidence.valid:
                output.append(
                    ResidualEvidence.observed(
                        "reappearance",
                        "track",
                        result.evidence.value,
                        confidence=result.evidence.quality,
                        spatial_support=support,
                        temporal_support={"frame_index": frame.frame_index},
                        metadata={"reid_source": result.observation.reid_source},
                    )
                )
            else:
                output.append(
                    ResidualEvidence.unavailable(
                        "reappearance",
                        "track",
                        result.missing_reason,
                        spatial_support=support,
                        temporal_support={"frame_index": frame.frame_index},
                    )
                )
        for track_id, state in frame.reappearance_states.items():
            if track_id in formal_reappearance:
                continue
            output.append(
                ResidualEvidence.unavailable(
                    "reappearance",
                    "track",
                    "event_state_requires_formal_d3_reappearance_inputs",
                    availability="not_applicable" if state == "none" else "blocked_by_input",
                    temporal_support={"frame_index": frame.frame_index, "track_id": track_id},
                )
            )
    return output


__all__ = [
    "build_d3_frame_graph",
    "classify_occlusion_event",
    "compute_reappearance_residual",
    "compute_relation_residuals",
]
