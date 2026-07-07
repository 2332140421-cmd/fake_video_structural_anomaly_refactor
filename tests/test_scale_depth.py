from __future__ import annotations

from _bootstrap import ensure_project_test_environment

ensure_project_test_environment(__file__)

import numpy as np
import pytest

from semantic3d.scale_depth import (
    ObjectObservation,
    ScalePrior,
    pairwise_scale_depth_residuals,
    scale_depth_residual,
)


FRAME_AREA = 1920.0 * 1080.0
SCALE_PRIORS = {
    "soccer_ball": ScalePrior(min_size=0.20, max_size=0.24),
    "elephant": ScalePrior(min_size=2.40, max_size=3.40),
}


def test_soccer_ball_near_elephant_far_reasonable_projection_has_zero_residual() -> None:
    soccer_ball = ObjectObservation(
        object_id="obj_soccer",
        label="soccer_ball",
        mask_area=4_800.0,
        frame_area=FRAME_AREA,
        depth=12.0,
    )
    elephant = ObjectObservation(
        object_id="obj_elephant",
        label="elephant",
        mask_area=120_000.0,
        frame_area=FRAME_AREA,
        depth=36.0,
    )

    residual, details = scale_depth_residual(soccer_ball, elephant, SCALE_PRIORS)

    assert details["lower"] <= details["depth_ratio"] <= details["upper"]
    assert residual == pytest.approx(0.0)


def test_close_depth_but_elephant_projection_too_small_has_large_residual() -> None:
    soccer_ball = ObjectObservation(
        object_id="obj_soccer",
        label="soccer_ball",
        mask_area=4_800.0,
        frame_area=FRAME_AREA,
        depth=10.0,
    )
    elephant = ObjectObservation(
        object_id="obj_elephant_small",
        label="elephant",
        mask_area=1_200.0,
        frame_area=FRAME_AREA,
        depth=10.0,
    )

    residual, details = scale_depth_residual(soccer_ball, elephant, SCALE_PRIORS)

    assert details["depth_ratio"] > details["upper"]
    assert residual > 0.8


def test_missing_scale_prior_raises_clear_exception() -> None:
    soccer_ball = ObjectObservation(
        object_id="obj_soccer",
        label="soccer_ball",
        mask_area=4_800.0,
        frame_area=FRAME_AREA,
        depth=10.0,
    )
    unknown = ObjectObservation(
        object_id="obj_unknown",
        label="dragon",
        mask_area=10_000.0,
        frame_area=FRAME_AREA,
        depth=20.0,
    )

    with pytest.raises(KeyError, match="Missing scale prior.*dragon.*obj_unknown"):
        scale_depth_residual(soccer_ball, unknown, SCALE_PRIORS)


@pytest.mark.parametrize(
    "bad_object, error_text",
    [
        (
            ObjectObservation(
                object_id="bad_mask",
                label="soccer_ball",
                mask_area=0.0,
                frame_area=FRAME_AREA,
                depth=10.0,
            ),
            "mask_area must be > 0",
        ),
        (
            ObjectObservation(
                object_id="bad_depth",
                label="soccer_ball",
                mask_area=4_800.0,
                frame_area=FRAME_AREA,
                depth=-1.0,
            ),
            "depth must be > 0",
        ),
    ],
)
def test_invalid_mask_area_or_depth_raises_clear_exception(
    bad_object: ObjectObservation, error_text: str
) -> None:
    elephant = ObjectObservation(
        object_id="obj_elephant",
        label="elephant",
        mask_area=120_000.0,
        frame_area=FRAME_AREA,
        depth=36.0,
    )

    with pytest.raises(ValueError, match=error_text):
        scale_depth_residual(bad_object, elephant, SCALE_PRIORS)


def test_pairwise_matrix_shape_and_diagonal_are_correct() -> None:
    objects = [
        ObjectObservation(
            object_id="obj_soccer",
            label="soccer_ball",
            mask_area=4_800.0,
            frame_area=FRAME_AREA,
            depth=12.0,
        ),
        ObjectObservation(
            object_id="obj_elephant",
            label="elephant",
            mask_area=120_000.0,
            frame_area=FRAME_AREA,
            depth=36.0,
        ),
        ObjectObservation(
            object_id="obj_elephant_small",
            label="elephant",
            mask_area=1_200.0,
            frame_area=FRAME_AREA,
            depth=10.0,
        ),
    ]

    matrix, details = pairwise_scale_depth_residuals(
        objects, SCALE_PRIORS, use_log=True
    )

    assert matrix.shape == (3, 3)
    assert np.allclose(np.diag(matrix), 0.0)
    assert ("obj_soccer", "obj_elephant") in details
    assert ("obj_elephant", "obj_soccer") in details
