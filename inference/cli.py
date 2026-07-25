"""Two commands: analyze one video or train the small residual head."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from experiments.train import build_manifest_samples, train_residual_head
from models.providers import (
    LegacyDepthIntrinsicsProviderAdapter,
    LegacyObjectProviderAdapter,
    LegacyPoseProviderAdapter,
    LegacyTrackProviderAdapter,
    RealInstanceMaskProvider,
    RealObjectProvider,
    UniDepthV2Adapter,
)
from utils.config import load_config, resolve_path, validate_config
from utils.logging import configure_logging

from .outputs import save_analysis_outputs
from .pipeline import ForgeryAnalysisPipeline


def _pipeline(config_path: str | Path) -> tuple[ForgeryAnalysisPipeline, dict]:
    config = load_config(config_path)
    validate_config(config)
    providers = config["providers"]
    device = str(providers["device"])
    object_weights = resolve_path(config, providers["object_weights"])
    instance_weights = resolve_path(config, providers["instance_weights"])
    depth_weights = resolve_path(config, providers["unidepth_weights"])
    detector = RealObjectProvider(model_path=object_weights, device=device)
    segmenter = RealInstanceMaskProvider(model_path=instance_weights, device=device)
    depth = UniDepthV2Adapter(
        weights_path=depth_weights,
        expected_weight_sha256=str(providers["unidepth_weight_sha256"]),
        device=device,
    )
    config["object_semantic"]["prior_path"] = str(
        resolve_path(config, config["object_semantic"]["prior_path"])
    )
    pose = LegacyPoseProviderAdapter()
    return (
        ForgeryAnalysisPipeline(
            config=config,
            object_provider=LegacyObjectProviderAdapter(detector, segmenter),
            depth_provider=LegacyDepthIntrinsicsProviderAdapter(depth),
            pose_provider=pose,
            track_provider=LegacyTrackProviderAdapter(pose),
        ),
        config,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true")
    commands = parser.add_subparsers(dest="command", required=True)
    analyze = commands.add_parser("analyze", help="analyze one local video")
    analyze.add_argument("--video", required=True)
    analyze.add_argument("--config", default="configs/default.yaml")
    analyze.add_argument("--output", required=True)
    analyze.add_argument("--max-frames", type=int)
    analyze.add_argument("--max-clips", type=int)
    train = commands.add_parser("train", help="train only the residual temporal head")
    train.add_argument("--manifest", required=True)
    train.add_argument("--config", default="configs/default.yaml")
    train.add_argument("--epochs", type=int, default=3, choices=(3, 4, 5))
    train.add_argument("--output", required=True)
    train.add_argument("--channel-schema", required=True)
    train.add_argument("--resume")
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    configure_logging(arguments.verbose)
    if arguments.command == "analyze":
        pipeline, config = _pipeline(arguments.config)
        result = pipeline.analyze_video(
            arguments.video,
            max_frames=arguments.max_frames,
            max_clips=arguments.max_clips,
        )
        save_analysis_outputs(
            result,
            pipeline.last_observations,
            arguments.output,
            heatmap_sigma=float(config["localization"]["heatmap_sigma"]),
        )
        return 0
    config_path = Path(arguments.config).resolve()
    config = load_config(config_path)
    validate_config(config)
    bundle = build_manifest_samples(arguments.manifest, arguments.channel_schema)
    config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
    if config_sha256 != bundle.config_sha256:
        raise ValueError("Current config does not match the frozen residual manifest.")
    training = config["training"]
    train_residual_head(
        bundle.samples["train"],
        bundle.samples["validation"],
        output_dir=arguments.output,
        channel_schema=bundle.channel_schema,
        source_commit=bundle.source_commit,
        config_sha256=bundle.config_sha256,
        epochs=arguments.epochs,
        hidden_size=int(training["hidden_size"]),
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training.get("weight_decay", 1e-4)),
        batch_size=2,
        random_seed=42,
        resume=arguments.resume,
        device=str(config["providers"]["device"]),
        amp=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
