"""Two commands: analyze one video or train the small residual head."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from experiments.artifacts import initialize_run_artifacts
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
from semantic3d.real_object_provider import normalize_label

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
    detector = RealObjectProvider(
        model_path=object_weights,
        device=device,
        skip_unknown_scale_prior=False,
    )
    detector_names = getattr(detector.detector, "names", ())
    detector.allowed_labels = {
        normalize_label(str(name))
        for name in (
            detector_names.values()
            if hasattr(detector_names, "values")
            else detector_names
        )
    }
    segmenter = RealInstanceMaskProvider(model_path=instance_weights, device=device)
    depth = UniDepthV2Adapter(
        weights_path=depth_weights,
        expected_weight_sha256=str(providers["unidepth_weight_sha256"]),
        device=device,
    )
    config["object_semantic"]["prior_path"] = str(
        resolve_path(config, config["object_semantic"]["prior_path"])
    )
    config["object_semantic"]["canonical_axis_path"] = str(
        resolve_path(config, config["object_semantic"]["canonical_axis_path"])
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


def _positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "--epochs must be an integer greater than or equal to 1"
        ) from error
    if parsed < 1:
        raise argparse.ArgumentTypeError(
            "--epochs must be an integer greater than or equal to 1"
        )
    return parsed


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
    train.add_argument(
        "--runtime-path-manifest",
        help="read-only server path mapping; frozen provenance is not rewritten",
    )
    train.add_argument("--config", default="configs/training_default.yaml")
    train.add_argument(
        "--epochs",
        type=_positive_integer,
        default=3,
        metavar="N",
        help="number of epochs; N must be an integer greater than or equal to 1",
    )
    train.add_argument("--output", required=True)
    train.add_argument("--channel-schema")
    train.add_argument("--batch-size", type=int)
    train.add_argument("--learning-rate", type=float)
    train.add_argument("--weight-decay", type=float)
    train.add_argument("--seed", type=int)
    train.add_argument("--device")
    train.add_argument("--num-workers", type=int)
    train.add_argument("--log-every", type=int)
    train.add_argument("--classification-threshold", type=float)
    amp = train.add_mutually_exclusive_group()
    amp.add_argument("--amp", dest="amp", action="store_true")
    amp.add_argument("--no-amp", dest="amp", action="store_false")
    train.set_defaults(amp=None)
    train.add_argument("--resume")
    return parser


def _training_value(arguments, section: dict, name: str):
    value = getattr(arguments, name)
    return section[name] if value is None else value


def _print_data_preflight(bundle) -> None:
    from experiments.train import RESIDUAL_NAMES, sequence_statistics

    for split in ("train", "validation", "test"):
        samples = bundle.samples[split]
        if not samples:
            print(f"[DATA] split={split} video_count=0")
            continue
        stats = sequence_statistics(samples)
        print(f"[DATA] split={split}")
        for key, value in stats.items():
            print(f"{key}={value}")
        if {sample.label for sample in samples} != {0, 1}:
            raise ValueError(f"{split} must contain both real and fake samples.")
    combined = bundle.samples["train"] + bundle.samples["validation"]
    groups = {
        "D1": tuple(range(2, 6)),
        "D2": tuple(range(6, 9)),
        "semantic": tuple(range(0, 2)),
    }
    for name, indices in groups.items():
        count = sum(
            int(sample.availability[:, indices].sum()) for sample in combined
        )
        print(f"[DATA] {name}_available_value_count={count}")
        if count == 0:
            raise ValueError(f"Training gate failed: no valid {name} evidence.")
    print(f"[DATA] channel_names={list(RESIDUAL_NAMES)}")
    eligibility = bundle.eligibility_summary or {}
    for split, summary in eligibility.get("splits", {}).items():
        print(
            f"[ELIGIBILITY] split={split} "
            f"original={summary['original_count']} "
            f"eligible={summary['eligible_count']} "
            f"excluded={summary['excluded_count']} "
            f"coverage={summary['coverage']:.6f}"
        )
    print(
        "[ELIGIBILITY] status_counts="
        f"{eligibility.get('status_counts', {})}"
    )
    audit = bundle.leakage_audit
    print(
        "[DATA] leakage_check="
        f"{audit['status']} mode={audit['mode']} scope={audit['scope']} "
        f"finding_count={audit['finding_count']}",
        flush=True,
    )


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
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Training config root must be a mapping.")
    bundle = build_manifest_samples(
        arguments.manifest,
        arguments.channel_schema,
        runtime_path_manifest=arguments.runtime_path_manifest,
        leakage_check=config.get("leakage_check"),
        load_splits=("train", "validation"),
        no_valid_residual_policy=config.get("data", {}).get(
            "no_valid_residual_policy", "error"
        ),
    )
    training = config["training"]
    data = config["data"]
    model = config["model"]
    final_training = {
        "epochs": arguments.epochs,
        "batch_size": int(_training_value(arguments, training, "batch_size")),
        "learning_rate": float(
            _training_value(arguments, training, "learning_rate")
        ),
        "weight_decay": float(
            _training_value(arguments, training, "weight_decay")
        ),
        "seed": int(_training_value(arguments, training, "seed")),
        "num_workers": int(_training_value(arguments, training, "num_workers")),
        "log_every": int(_training_value(arguments, training, "log_every")),
        "checkpoint_every": int(training["checkpoint_every"]),
        "amp": bool(training["amp"] if arguments.amp is None else arguments.amp),
        "device": str(_training_value(arguments, training, "device")),
        "progress": dict(training.get("progress", {})),
    }
    threshold = float(
        data["classification_threshold"]
        if arguments.classification_threshold is None
        else arguments.classification_threshold
    )
    run_config = {
        "data": {**data, "classification_threshold": threshold},
        "training": final_training,
        "model": model,
        "metrics": config["metrics"],
        "leakage_check": config.get("leakage_check", {}),
        "eligibility": {
            "no_valid_residual_policy": data.get(
                "no_valid_residual_policy", "error"
            ),
            "loaded_splits": ["train", "validation"],
        },
        "source_config_path": str(config_path),
        "resume": str(Path(arguments.resume).resolve()) if arguments.resume else None,
    }
    if int(model["expected_input_channels"]) != 12:
        raise ValueError("Training config must expect exactly 12 residual channels.")
    _print_data_preflight(bundle)
    project_root = Path(__file__).resolve().parents[1]
    initialize_run_artifacts(
        arguments.output,
        bundle=bundle,
        run_config=run_config,
        project_root=project_root,
    )
    print("[MODEL] name=ResidualTemporalHead")
    print(f"[MODEL] hidden_size={int(model.get('hidden_size', 32))}")
    train_residual_head(
        bundle.samples["train"],
        bundle.samples["validation"],
        output_dir=arguments.output,
        channel_schema=bundle.channel_schema,
        source_commit=bundle.source_commit,
        source_config_sha256=bundle.source_config_sha256,
        manifest_sha256=bundle.manifest_sha256,
        epochs=arguments.epochs,
        hidden_size=int(model.get("hidden_size", 32)),
        learning_rate=final_training["learning_rate"],
        weight_decay=final_training["weight_decay"],
        batch_size=final_training["batch_size"],
        random_seed=final_training["seed"],
        resume=arguments.resume,
        device=final_training["device"],
        amp=final_training["amp"],
        num_workers=final_training["num_workers"],
        log_every=final_training["log_every"],
        checkpoint_every=final_training["checkpoint_every"],
        classification_threshold=threshold,
        progress_enabled=bool(
            training.get("progress", {}).get("enabled", True)
        ),
        progress_update_interval=int(
            training.get("progress", {}).get("update_interval", 1)
        ),
        progress_log_interval=int(
            training.get("progress", {}).get("log_interval", 20)
        ),
        bundle=bundle,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
