"""Static camera-frame depth-order diagnostics and occlusion consistency."""

from __future__ import annotations

import math

from ..shared_3d_observation import Object3DObservation, Shared3DFrameObservation
from ..validity import ResidualEvidence
from .residual_types import EvidenceRole, Static3DContext


def _objects(
    frame: Shared3DFrameObservation, object_a_id: str, object_b_id: str
) -> tuple[Object3DObservation, Object3DObservation] | None:
    try:
        context = Static3DContext(frame)
        return context.object_by_id(object_a_id), context.object_by_id(object_b_id)
    except KeyError:
        return None


def _valid_centers(
    objects: tuple[Object3DObservation, Object3DObservation] | None,
) -> bool:
    return bool(
        objects is not None
        and all(
            obj.valid
            and obj.center_3d_camera is not None
            and obj.center_3d_camera.valid
            and obj.center_3d_camera.z is not None
            and math.isfinite(float(obj.center_3d_camera.z))
            for obj in objects
        )
    )


class DepthOrder3DResidual:
    """Read depth order from one shared frame and evaluate explicit occlusion claims."""

    def center_depth_order(
        self,
        frame: Shared3DFrameObservation,
        object_a_id: str,
        object_b_id: str,
    ) -> ResidualEvidence:
        """Record signed centre order as diagnostic, not anomaly evidence."""

        objects = _objects(frame, object_a_id, object_b_id)
        if not _valid_centers(objects):
            return ResidualEvidence.missing(
                "center_depth_order",
                "invalid_object_center_depth",
                source_ids=(object_a_id, object_b_id),
            )
        assert objects is not None
        object_a, object_b = objects
        assert object_a.center_3d_camera is not None
        assert object_b.center_3d_camera is not None
        z_a = float(object_a.center_3d_camera.z)  # type: ignore[arg-type]
        z_b = float(object_b.center_3d_camera.z)  # type: ignore[arg-type]
        signed_difference = z_b - z_a
        relation = "a_in_front_of_b" if signed_difference > 0 else (
            "b_in_front_of_a" if signed_difference < 0 else "same_depth"
        )
        quality = min(object_a.reconstruction_quality, object_b.reconstruction_quality)
        return ResidualEvidence.observed(
            "center_depth_order",
            signed_difference,
            quality=quality,
            source_ids=(object_a_id, object_b_id),
            metadata={
                "evidence_role": EvidenceRole.DIAGNOSTIC.value,
                "anomaly_residual": False,
                "object_a_depth_z": z_a,
                "object_b_depth_z": z_b,
                "relation": relation,
                "depth_definition": frame.camera.depth_definition.value,
            },
        )

    def occlusion_depth_consistency(
        self,
        frame: Shared3DFrameObservation,
        object_a_id: str,
        object_b_id: str,
        *,
        front_object_id: str | None,
        overlap_ratio: float | None,
        relation_quality: float = 1.0,
        tolerance_ratio: float = 0.01,
    ) -> ResidualEvidence:
        """Compare an observed 2D foreground relation with camera Z order."""

        if (
            front_object_id not in {object_a_id, object_b_id}
            or overlap_ratio is None
            or not math.isfinite(float(overlap_ratio))
            or overlap_ratio <= 0.0
        ):
            return ResidualEvidence.missing(
                "occlusion_depth_consistency",
                "no_occlusion_relation_evidence",
                source_ids=(object_a_id, object_b_id),
                metadata={
                    "front_object_id": front_object_id,
                    "overlap_ratio": overlap_ratio,
                },
            )
        if not 0.0 <= relation_quality <= 1.0:
            raise ValueError("relation_quality must be in [0, 1].")
        objects = _objects(frame, object_a_id, object_b_id)
        if not _valid_centers(objects):
            return ResidualEvidence.missing(
                "occlusion_depth_consistency",
                "invalid_object_center_depth",
                source_ids=(object_a_id, object_b_id),
            )
        assert objects is not None
        by_id = {object_a_id: objects[0], object_b_id: objects[1]}
        back_object_id = object_b_id if front_object_id == object_a_id else object_a_id
        front = by_id[front_object_id]
        back = by_id[back_object_id]
        assert front.center_3d_camera is not None and back.center_3d_camera is not None
        front_z = float(front.center_3d_camera.z)  # type: ignore[arg-type]
        back_z = float(back.center_3d_camera.z)  # type: ignore[arg-type]
        normalizer = max(abs(front_z), abs(back_z), 1e-12)
        signed_margin = (back_z - front_z) / normalizer
        residual = max(0.0, -signed_margin - tolerance_ratio)
        quality = min(
            front.reconstruction_quality,
            back.reconstruction_quality,
            float(relation_quality),
            float(min(1.0, overlap_ratio)),
        )
        return ResidualEvidence.observed(
            "occlusion_depth_consistency",
            residual,
            quality=quality,
            source_ids=(object_a_id, object_b_id),
            metadata={
                "evidence_role": EvidenceRole.ANOMALY_RESIDUAL.value,
                "front_object_id": front_object_id,
                "back_object_id": back_object_id,
                "front_depth_z": front_z,
                "back_depth_z": back_z,
                "normalized_depth_margin": signed_margin,
                "tolerance_ratio": tolerance_ratio,
                "overlap_ratio": float(overlap_ratio),
                "consistent": residual == 0.0,
            },
        )
