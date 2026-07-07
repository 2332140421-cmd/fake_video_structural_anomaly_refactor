"""Residual fusion utilities for clip-level and video-level anomaly scoring.

The scale-depth residual R_sd is intentionally kept separate from the temporal
depth consistency residual R_depth_cons:

    R_total =
        alpha     * R_flow
      + beta      * R_track
      + gamma     * R_depth_cons
      + delta     * R_occ
      + eta       * R_corr
      + lambda_sd * R_sd
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Dict, Iterable, List, Sequence, Tuple, Union

import numpy as np


@dataclass(frozen=True)
class ResidualWeights:
    """Weights for structural residual fusion.

    Attributes:
        flow: Weight alpha for optical-flow residual R_flow.
        track: Weight beta for point-track residual R_track.
        depth_cons: Weight gamma for temporal depth consistency R_depth_cons.
        occ: Weight delta for occlusion residual R_occ.
        corr: Weight eta for spatial correspondence residual R_corr.
        scale_depth: Weight lambda_sd for object scale-depth residual R_sd.
    """

    flow: float
    track: float
    depth_cons: float
    occ: float
    corr: float
    scale_depth: float


@dataclass(frozen=True)
class ResidualValues:
    """Residual values for one clip or video segment.

    R_depth_cons and R_sd are represented by distinct fields:
    `depth_cons` is the depth temporal consistency residual, while
    `scale_depth` is the object-level scale-depth consistency residual.
    """

    flow: float
    track: float
    depth_cons: float
    occ: float
    corr: float
    scale_depth: float


ResidualInput = Union[ResidualValues, Sequence[ResidualValues]]


def _field_names() -> List[str]:
    """Return residual field names in a stable fusion order."""

    return [field.name for field in fields(ResidualValues)]


def _is_sequence_input(values: ResidualInput) -> bool:
    """Return True when values represents multiple segment residuals."""

    return not isinstance(values, ResidualValues)


def _ensure_residual_values(value: object, index: int | None = None) -> ResidualValues:
    """Validate one ResidualValues object and all of its fields."""

    if not isinstance(value, ResidualValues):
        location = "" if index is None else f" at index {index}"
        raise TypeError(f"Expected ResidualValues{location}, got {type(value).__name__}.")

    for name in _field_names():
        residual = getattr(value, name)
        if residual is None:
            raise ValueError(f"Missing residual '{name}' in ResidualValues.")
        if not np.isfinite(float(residual)):
            raise ValueError(
                f"Residual '{name}' must be finite, got {residual!r}."
            )
    return value


def _as_list(values: ResidualInput) -> Tuple[List[ResidualValues], bool]:
    """Convert single or multi-segment input into a list."""

    if isinstance(values, ResidualValues):
        return [_ensure_residual_values(values)], False

    value_list = list(values)
    if not value_list:
        raise ValueError("ResidualValues list must not be empty.")
    return [
        _ensure_residual_values(value, index=index)
        for index, value in enumerate(value_list)
    ], True


def _values_to_matrix(values: Sequence[ResidualValues]) -> np.ndarray:
    """Convert residual dataclasses to an N x 6 numpy matrix."""

    names = _field_names()
    return np.asarray(
        [[float(getattr(value, name)) for name in names] for value in values],
        dtype=float,
    )


def _matrix_to_values(matrix: np.ndarray) -> List[ResidualValues]:
    """Convert an N x 6 matrix back to ResidualValues objects."""

    names = _field_names()
    return [
        ResidualValues(**{name: float(row[column]) for column, name in enumerate(names)})
        for row in matrix
    ]


def _weights_to_vector(weights: ResidualWeights) -> np.ndarray:
    """Convert weights to a vector aligned with ResidualValues field order."""

    if not isinstance(weights, ResidualWeights):
        raise TypeError(f"Expected ResidualWeights, got {type(weights).__name__}.")

    vector = np.asarray([float(getattr(weights, name)) for name in _field_names()])
    if not np.all(np.isfinite(vector)):
        raise ValueError("All residual weights must be finite.")
    return vector


def normalize_residuals(
    values: ResidualInput,
    method: str = "minmax",
) -> Union[ResidualValues, List[ResidualValues]]:
    """Normalize residual values field-wise.

    Args:
        values: One ResidualValues object or a sequence of ResidualValues.
        method: "minmax" or "zscore".

    Returns:
        Normalized residuals with the same single-vs-list shape as the input.

    Notes:
        For a single segment, normalization is performed over that single row.
        Constant fields are assigned 0 to avoid division by zero.
    """

    value_list, was_sequence = _as_list(values)
    matrix = _values_to_matrix(value_list)

    if method == "minmax":
        minimum = matrix.min(axis=0)
        maximum = matrix.max(axis=0)
        denom = maximum - minimum
        normalized = np.divide(
            matrix - minimum,
            denom,
            out=np.zeros_like(matrix),
            where=denom > 0,
        )
    elif method == "zscore":
        mean = matrix.mean(axis=0)
        std = matrix.std(axis=0)
        normalized = np.divide(
            matrix - mean,
            std,
            out=np.zeros_like(matrix),
            where=std > 0,
        )
    else:
        raise ValueError(
            f"Unknown normalization method '{method}'. Use 'minmax' or 'zscore'."
        )

    normalized_values = _matrix_to_values(normalized)
    return normalized_values if was_sequence else normalized_values[0]


def fuse_residuals(
    values: ResidualInput,
    weights: ResidualWeights,
    normalize: bool = False,
) -> Union[float, np.ndarray]:
    """Fuse residuals into clip-level anomaly scores.

    Args:
        values: One segment residual or a sequence of segment residuals.
        weights: ResidualWeights aligned with the six residual terms.
        normalize: If True, apply field-wise min-max normalization before fusion.

    Returns:
        A float for a single segment, or an N-element numpy array for multiple
        segments.
    """

    value_list, was_sequence = _as_list(values)
    if normalize:
        normalized = normalize_residuals(value_list, method="minmax")
        value_list = list(normalized)  # type: ignore[arg-type]

    matrix = _values_to_matrix(value_list)
    weight_vector = _weights_to_vector(weights)
    scores = matrix @ weight_vector
    return scores if was_sequence else float(scores[0])


def video_risk_score(
    scores: Sequence[float],
    w_mean: float = 0.5,
    w_max: float = 0.3,
    w_topk: float = 0.2,
    topk: int = 3,
) -> Tuple[float, Dict[str, float]]:
    """Aggregate segment scores into one video-level risk score.

    Formula:
        score_video =
            w_mean * mean(scores)
          + w_max * max(scores)
          + w_topk * topk_mean(scores)
    """

    score_array = np.asarray(scores, dtype=float)
    if score_array.ndim != 1 or score_array.size == 0:
        raise ValueError("scores must be a non-empty 1D sequence.")
    if not np.all(np.isfinite(score_array)):
        raise ValueError("scores must contain only finite values.")
    if topk <= 0:
        raise ValueError(f"topk must be > 0, got topk={topk}.")

    k = min(int(topk), score_array.size)
    topk_values = np.sort(score_array)[-k:]
    mean_score = float(score_array.mean())
    max_score = float(score_array.max())
    topk_mean = float(topk_values.mean())

    score_video = (
        float(w_mean) * mean_score
        + float(w_max) * max_score
        + float(w_topk) * topk_mean
    )
    details = {
        "mean": mean_score,
        "max": max_score,
        "topk_mean": topk_mean,
        "topk": float(k),
        "w_mean": float(w_mean),
        "w_max": float(w_max),
        "w_topk": float(w_topk),
        "score_video": float(score_video),
    }
    return float(score_video), details


def residual_values_to_dict(values: ResidualValues) -> Dict[str, float]:
    """Return a plain dictionary for logging or serialization."""

    _ensure_residual_values(values)
    return {name: float(value) for name, value in asdict(values).items()}
