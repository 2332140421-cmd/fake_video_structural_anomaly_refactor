"""The train CLI accepts runtime paths, never training hyperparameters."""

from __future__ import annotations

import contextlib
import io
from pathlib import Path

import yaml

from inference.cli import _parser


def _arguments() -> list[str]:
    return [
        "train",
        "--config",
        "configs/training_default.yaml",
        "--manifest",
        "manifest.csv",
        "--runtime-path-manifest",
        "runtime.csv",
        "--output",
        "run",
    ]


def test_train_cli_contains_only_runtime_arguments() -> None:
    arguments = _parser().parse_args(_arguments())
    assert vars(arguments) == {
        "verbose": False,
        "command": "train",
        "manifest": "manifest.csv",
        "runtime_path_manifest": "runtime.csv",
        "config": "configs/training_default.yaml",
        "output": "run",
    }


def test_training_default_is_the_epoch_source() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(
        (project_root / "configs" / "training_default.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert config["training"]["epochs"] == 50


def test_train_cli_rejects_training_overrides() -> None:
    overrides = (
        ("--epochs", "3"),
        ("--channel-schema", "schema.json"),
        ("--batch-size", "2"),
        ("--learning-rate", "0.001"),
        ("--weight-decay", "0.0001"),
        ("--seed", "42"),
        ("--device", "cuda"),
        ("--num-workers", "0"),
        ("--log-every", "1"),
        ("--classification-threshold", "0.5"),
        ("--amp",),
        ("--no-amp",),
        ("--resume", "last.pt"),
    )
    for override in overrides:
        with contextlib.redirect_stderr(io.StringIO()):
            try:
                _parser().parse_args([*_arguments(), *override])
            except SystemExit as error:
                assert error.code == 2
            else:
                raise AssertionError(f"training override unexpectedly parsed: {override}")


if __name__ == "__main__":
    tests = (
        test_train_cli_contains_only_runtime_arguments,
        test_training_default_is_the_epoch_source,
        test_train_cli_rejects_training_overrides,
    )
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
