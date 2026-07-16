from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from semantic3d.static_3d import BoundaryDepth3DResidual

from synthetic_geometry import synthetic_object_3d, synthetic_shared_3d_frame


def _boundary_scene():
    depth = np.full((128, 128), 5.0, dtype=np.float32)
    mask = np.zeros((128, 128), dtype=bool)
    mask[30:90, 30:90] = True
    depth[mask] = 2.0
    obj = synthetic_object_3d("object_1", metric=False)
    frame = synthetic_shared_3d_frame(
        (obj,), depth_map=depth, metric=False
    )
    return frame, mask


def test_instance_mask_boundary_is_preferred_and_high_quality() -> None:
    frame, mask = _boundary_scene()
    evidence = BoundaryDepth3DResidual().evaluate(
        frame, "object_1", instance_mask=mask, bbox=[30, 30, 89, 89]
    )
    assert evidence.valid
    assert evidence.metadata["boundary_source"] == "instance_mask"
    assert evidence.metadata["bbox_is_not_instance_contour"] is False
    assert evidence.metadata["aligned_boundary_ratio"] > 0.9
    assert evidence.value == 0.0


def test_bbox_fallback_is_explicitly_lower_quality() -> None:
    frame, mask = _boundary_scene()
    module = BoundaryDepth3DResidual()
    mask_evidence = module.evaluate(frame, "object_1", instance_mask=mask)
    bbox_evidence = module.evaluate(
        frame, "object_1", bbox=np.asarray([30, 30, 89, 89], dtype=float)
    )
    assert bbox_evidence.valid
    assert bbox_evidence.metadata["boundary_source"] == "bbox_sparse_fallback"
    assert bbox_evidence.metadata["bbox_is_not_instance_contour"] is True
    assert bbox_evidence.quality < mask_evidence.quality


def test_missing_boundary_observation_is_nan() -> None:
    frame, _ = _boundary_scene()
    obj = frame.objects[0]
    metadata = dict(obj.metadata)
    metadata.pop("source_bbox", None)
    frame = replace(frame, objects=(replace(obj, metadata=metadata),))
    evidence = BoundaryDepth3DResidual().evaluate(frame, "object_1")
    assert not evidence.valid
    assert math.isnan(evidence.value)
    assert evidence.missing_reason == "no_boundary_observation"
