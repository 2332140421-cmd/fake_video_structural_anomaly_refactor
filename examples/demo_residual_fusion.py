"""Demo for segment-level residual fusion and video-level risk scoring."""

from __future__ import annotations

import sys
from pathlib import Path

from _bootstrap import ensure_project_environment

PROJECT_ROOT = ensure_project_environment()
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from semantic3d.residual_fusion import (  # noqa: E402
    ResidualValues,
    ResidualWeights,
    fuse_residuals,
    video_risk_score,
)


def main() -> None:
    weights = ResidualWeights(
        flow=0.20,
        track=0.20,
        depth_cons=0.15,
        occ=0.10,
        corr=0.15,
        scale_depth=0.20,
    )

    segments = [
        ResidualValues(
            flow=0.12,
            track=0.18,
            depth_cons=0.10,
            occ=0.05,
            corr=0.08,
            scale_depth=0.15,
        ),
        ResidualValues(
            flow=0.25,
            track=0.30,
            depth_cons=0.20,
            occ=0.10,
            corr=0.12,
            scale_depth=0.85,
        ),
        ResidualValues(
            flow=0.08,
            track=0.12,
            depth_cons=0.14,
            occ=0.04,
            corr=0.20,
            scale_depth=0.10,
        ),
    ]

    scores = fuse_residuals(segments, weights, normalize=False)
    score_video, details = video_risk_score(
        scores, w_mean=0.5, w_max=0.3, w_topk=0.2, topk=2
    )

    print("Segment scores:")
    for index, score in enumerate(scores):
        print(f"  segment_{index}: {score:.6f}")

    print("\nVideo risk:")
    print(f"  mean = {details['mean']:.6f}")
    print(f"  max = {details['max']:.6f}")
    print(f"  topk_mean = {details['topk_mean']:.6f}")
    print(f"  score_video = {score_video:.6f}")


if __name__ == "__main__":
    main()
