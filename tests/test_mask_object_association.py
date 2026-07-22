from __future__ import annotations

import numpy as np

from semantic3d.observations import FrameObservationJSON, ObjectObservationJSON
from semantic3d.occlusion import InstanceMaskCandidate, associate_instance_masks


def _mask(x1, y1, x2, y2):
    value = np.zeros((64, 64), dtype=bool)
    value[y1:y2, x1:x2] = True
    return value


def _candidate(identifier: str, label: str, mask: np.ndarray) -> InstanceMaskCandidate:
    return InstanceMaskCandidate(identifier, 0, label, mask, 0.9, identifier, "test_segmenter", "model", "v1", "cpu")


def test_mask_object_association_is_one_to_one() -> None:
    frame = FrameObservationJSON(0, "f", 64, 64, [
        ObjectObservationJSON("a", "person", 400, 4096, 5, bbox=[5, 5, 25, 25], track_id="ta"),
        ObjectObservationJSON("b", "person", 400, 4096, 5, bbox=[35, 35, 55, 55], track_id="tb"),
    ])
    result = associate_instance_masks(video_id="v", frame=frame, candidates=(
        _candidate("ca", "person", _mask(5, 5, 25, 25)),
        _candidate("cb", "person", _mask(35, 35, 55, 55)),
    ))
    assert all(mask.valid for mask in result.masks)
    assert len(result.assigned_candidate_ids) == len(set(result.assigned_candidate_ids)) == 2
    assert {mask.object_track_id for mask in result.masks} == {"ta", "tb"}


def test_category_conflict_is_rejected_with_candidate_reason() -> None:
    frame = FrameObservationJSON(0, "f", 64, 64, [ObjectObservationJSON("a", "person", 400, 4096, 5, bbox=[5, 5, 25, 25], track_id="ta")])
    result = associate_instance_masks(video_id="v", frame=frame, candidates=(_candidate("c", "car", _mask(5, 5, 25, 25)),))
    assert not result.masks[0].valid
    assert result.diagnostics[0].missing_reason == "no_compatible_instance_mask_candidate"
    assert result.diagnostics[0].candidate_details[0]["rejection_reason"] == "semantic_category_conflict"


def test_same_source_detection_id_has_priority_among_compatible_masks() -> None:
    frame = FrameObservationJSON(0, "f", 64, 64, [ObjectObservationJSON("same", "person", 400, 4096, 5, bbox=[5, 5, 25, 25], track_id="ta")])
    result = associate_instance_masks(video_id="v", frame=frame, candidates=(
        _candidate("other", "person", _mask(5, 5, 25, 25)),
        _candidate("same", "person", _mask(6, 6, 26, 26)),
    ))
    assert result.diagnostics[0].candidate_id == "same"
    assert result.diagnostics[0].association_source == "source_detection_id"
