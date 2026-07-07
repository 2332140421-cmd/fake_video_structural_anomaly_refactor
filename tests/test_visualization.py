from __future__ import annotations

from _bootstrap import ensure_project_test_environment

ensure_project_test_environment(__file__)

from pathlib import Path

import numpy as np
import pytest

from semantic3d.multilevel_residuals import (
    ObjectMaskObservation,
    ObjectLevelResidual,
    ObjectPairResidual,
)
from semantic3d.visualization import (
    draw_multilevel_summary,
    draw_object_residual_map,
    draw_pairwise_residual_graph,
)


def _mask(shape: tuple[int, int], x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask


def _scene() -> tuple[list[ObjectMaskObservation], list[ObjectLevelResidual], list[ObjectPairResidual]]:
    shape = (32, 32)
    objects = [
        ObjectMaskObservation("soccer_ball", "soccer_ball", _mask(shape, 4, 10, 12, 18)),
        ObjectMaskObservation("elephant", "elephant", _mask(shape, 18, 6, 30, 24)),
    ]
    object_residuals = [
        ObjectLevelResidual("soccer_ball", "soccer_ball", 0.1, 0.2, 0.1, 0.1),
        ObjectLevelResidual("elephant", "elephant", 0.3, 0.4, 0.5, 0.2),
    ]
    pair_residuals = [
        ObjectPairResidual("soccer_ball", "elephant", "soccer_ball", "elephant", 1.2)
    ]
    return objects, object_residuals, pair_residuals


def test_object_residual_map_generates_png(tmp_path: Path) -> None:
    objects, object_residuals, _ = _scene()
    output_path = tmp_path / "missing" / "object_residual_map.png"

    fig, _, heatmap = draw_object_residual_map(
        (32, 32), objects, object_residuals, output_path=output_path
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0
    assert heatmap.shape == (32, 32)
    # Object map contains only single-object residuals, not pairwise R_sd=1.2.
    assert float(np.max(heatmap)) == pytest.approx(1.4)
    fig.clf()


def test_pairwise_residual_graph_generates_png(tmp_path: Path) -> None:
    objects, _, pair_residuals = _scene()
    output_path = tmp_path / "pairwise_residual_graph.png"

    fig, _ = draw_pairwise_residual_graph(
        objects, pair_residuals, output_path=output_path, threshold=0.1
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0
    fig.clf()


def test_multilevel_summary_generates_png(tmp_path: Path) -> None:
    objects, object_residuals, pair_residuals = _scene()
    output_path = tmp_path / "summary" / "multilevel_summary.png"

    fig, _ = draw_multilevel_summary(
        objects,
        object_residuals,
        pair_residuals,
        clip_score=1.5,
        output_path=output_path,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0
    fig.clf()


def test_empty_object_list_raises_clear_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="objects must not be empty"):
        draw_object_residual_map(
            (32, 32), [], [], output_path=tmp_path / "empty.png"
        )


def test_pairwise_graph_without_high_residual_still_generates_png(tmp_path: Path) -> None:
    objects, _, _ = _scene()
    pair_residuals = [
        ObjectPairResidual("soccer_ball", "elephant", "soccer_ball", "elephant", 0.01)
    ]
    output_path = tmp_path / "no_high_pairs.png"

    fig, _ = draw_pairwise_residual_graph(
        objects, pair_residuals, output_path=output_path, threshold=0.5
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0
    fig.clf()


def test_pairwise_edge_disappears_when_scale_depth_is_zero(tmp_path: Path) -> None:
    objects, _, _ = _scene()
    high_pair = [
        ObjectPairResidual("soccer_ball", "elephant", "soccer_ball", "elephant", 1.2)
    ]
    zero_pair = [
        ObjectPairResidual("soccer_ball", "elephant", "soccer_ball", "elephant", 0.0)
    ]

    high_fig, high_ax = draw_pairwise_residual_graph(
        objects, high_pair, output_path=tmp_path / "high.png", threshold=0.1
    )
    zero_fig, zero_ax = draw_pairwise_residual_graph(
        objects, zero_pair, output_path=tmp_path / "zero.png", threshold=0.1
    )

    assert len(high_ax.lines) > len(zero_ax.lines)
    assert len(zero_ax.lines) == 0
    high_fig.clf()
    zero_fig.clf()
