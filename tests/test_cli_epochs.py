"""Pure argparse coverage for the unbounded positive epoch contract."""

from __future__ import annotations

import contextlib
import io

from inference.cli import _parser


def _arguments(value: str) -> list[str]:
    return [
        "train",
        "--manifest",
        "manifest.csv",
        "--output",
        "run",
        "--epochs",
        value,
    ]


def test_positive_epochs_parse() -> None:
    parser = _parser()
    for value in ("1", "5", "30"):
        arguments = parser.parse_args(_arguments(value))
        assert arguments.epochs == int(value)


def test_default_epoch_semantics_are_unchanged() -> None:
    arguments = _parser().parse_args(
        ["train", "--manifest", "manifest.csv", "--output", "run"]
    )
    assert arguments.epochs == 3


def test_invalid_epochs_fail_with_clear_argparse_error() -> None:
    for value in ("0", "-1", "abc"):
        parser = _parser()
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            try:
                parser.parse_args(_arguments(value))
            except SystemExit as error:
                assert error.code == 2
            else:
                raise AssertionError(f"--epochs {value!r} unexpectedly parsed")
        assert "must be an integer greater than or equal to 1" in stderr.getvalue()


if __name__ == "__main__":
    tests = (
        test_positive_epochs_parse,
        test_default_epoch_semantics_are_unchanged,
        test_invalid_epochs_fail_with_clear_argparse_error,
    )
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
