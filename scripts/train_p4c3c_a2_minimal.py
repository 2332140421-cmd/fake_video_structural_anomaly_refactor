#!/usr/bin/env python3
"""Run the P4-C3C-A2 precomputed-evidence engineering training loop."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from semantic3d.minimal_training.engine import (  # noqa: E402
    load_training_config,
    run_training,
    with_config_overrides,
)
from semantic3d.minimal_training.synthetic import create_synthetic_fixture  # noqa: E402


def _ensure_project_environment() -> None:
    if not PROJECT_PYTHON.exists():
        raise RuntimeError(f"Project environment is missing: {PROJECT_PYTHON}")
    if Path(sys.executable).resolve() != PROJECT_PYTHON.resolve():
        os.execv(str(PROJECT_PYTHON), [str(PROJECT_PYTHON), *sys.argv])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a small missing-aware head from precomputed M6 evidence. "
            "This A2 entry never decodes videos or runs providers."
        )
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs/p4c3c_a2_minimal_training_v1.yaml"),
    )
    parser.add_argument("--train-formal-manifest")
    parser.add_argument("--train-manifest")
    parser.add_argument("--validation-formal-manifest")
    parser.add_argument("--validation-manifest")
    parser.add_argument("--output-dir")
    parser.add_argument("--resume-checkpoint")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--gradient-clip-norm", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--log-interval", type=int)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-validation-batches", type=int)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--synthetic-fixture",
        action="store_true",
        help=(
            "Create non-video engineering manifests under OUTPUT_DIR/fixture "
            "and use them for this run."
        ),
    )
    return parser.parse_args()


def main() -> int:
    _ensure_project_environment()
    args = parse_args()
    config = load_training_config(args.config, project_root=PROJECT_ROOT)
    overrides = {
        key: value
        for key, value in {
            "train_formal_manifest": args.train_formal_manifest,
            "train_manifest": args.train_manifest,
            "validation_formal_manifest": args.validation_formal_manifest,
            "validation_manifest": args.validation_manifest,
            "output_dir": args.output_dir,
            "resume_checkpoint": args.resume_checkpoint,
            "device": args.device,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "gradient_clip_norm": args.gradient_clip_norm,
            "seed": args.seed,
            "log_interval": args.log_interval,
            "max_train_batches": args.max_train_batches,
            "max_validation_batches": args.max_validation_batches,
            "amp": args.amp,
            "deterministic": args.deterministic,
        }.items()
        if value is not None
    }
    config = with_config_overrides(config, **overrides)
    if args.synthetic_fixture:
        fixture = create_synthetic_fixture(
            config.output_dir / "fixture",
            feature_contract_path=config.feature_contract,
        )
        config = with_config_overrides(
            config,
            train_formal_manifest=fixture.train_formal_manifest,
            train_manifest=fixture.train_manifest,
            validation_formal_manifest=fixture.validation_formal_manifest,
            validation_manifest=fixture.validation_manifest,
        )
    result = run_training(config)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
