"""Terminal output must contain startup plus one summary line per epoch."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from data.residual_dataset import (
    RESIDUAL_NAMES,
    ResidualSequence,
    residual_channel_schema,
)
from experiments.train import train_residual_head


def _sample(sample_id: str, label: int) -> ResidualSequence:
    values = np.full((2, len(RESIDUAL_NAMES)), 0.1 + 0.1 * label, np.float32)
    return ResidualSequence(
        residuals=values,
        availability=np.ones_like(values, dtype=bool),
        confidence=np.full_like(values, 0.9),
        label=label,
        sample_id=sample_id,
        clip_ids=(f"{sample_id}-0", f"{sample_id}-1"),
    )


def test_two_epochs_emit_only_two_epoch_summaries() -> None:
    train = (_sample("train-real", 0), _sample("train-fake", 1))
    validation = (
        _sample("validation-real", 0),
        _sample("validation-fake", 1),
    )
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary) / "run"
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            train_residual_head(
                train,
                validation,
                output_dir=output,
                channel_schema=residual_channel_schema("commit", "config"),
                source_commit="commit",
                source_config_sha256="config",
                epochs=2,
                hidden_size=8,
                batch_size=2,
                device="cpu",
                amp=False,
                progress_mode="epoch",
            )

        text = stdout.getvalue()
        lines = [line for line in text.splitlines() if line]
        epoch_lines = [line for line in lines if line.startswith("Epoch ")]
        assert len(lines) == 3
        assert lines[0].startswith("[RUN] ")
        assert len(epoch_lines) == 2
        assert epoch_lines[0].startswith("Epoch 001/002 |")
        assert epoch_lines[1].startswith("Epoch 002/002 |")
        for line in epoch_lines:
            for token in (
                "train loss=",
                "acc=",
                "precision=",
                "recall=",
                "specificity=",
                "f1=",
                "roc_auc=",
                "pr_auc=",
                "val loss=",
                "lr=",
                "time=",
                "gpu_peak=",
            ):
                assert token in line
        assert "batch=" not in text
        assert "[Train]" not in text
        assert "[TrainEval]" not in text
        assert "[Validation]" not in text
        assert "\r" not in text

        epoch_metrics = [
            json.loads(line)
            for line in (output / "logs" / "epoch_metrics.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert [row["epoch"] for row in epoch_metrics] == [1, 2]
        checkpoint = torch.load(
            output / "checkpoints" / "last.pt",
            map_location="cpu",
            weights_only=False,
        )
        assert checkpoint["epoch"] == 2


if __name__ == "__main__":
    test_two_epochs_emit_only_two_epoch_summaries()
    print("PASS test_two_epochs_emit_only_two_epoch_summaries")
