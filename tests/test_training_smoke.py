import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.train import (
    RESIDUAL_NAMES,
    ResidualSequence,
    build_manifest_samples,
    evaluate_residual_head,
    residual_channel_schema,
    train_residual_head,
)
from models.temporal_head import ResidualTemporalHead


def _samples(count, seed, channels=12):
    random = np.random.default_rng(seed)
    output = []
    for index in range(count):
        label = index % 2
        steps = 3 + index % 2
        values = random.normal(
            0.2 + 0.6 * label, 0.05, size=(steps, channels)
        ).astype(np.float32)
        availability = random.random((steps, channels)) > 0.15
        values[~availability] = np.nan
        confidence = np.where(availability, 0.9, 0.0).astype(np.float32)
        output.append(ResidualSequence(values, availability, confidence, label))
    return output


def test_temporal_head_forward_backward_and_three_epoch_resume(tmp_path):
    head = ResidualTemporalHead(6, hidden_size=16)
    residuals = torch.rand(2, 4, 6, requires_grad=True)
    availability = torch.ones_like(residuals, dtype=torch.bool)
    confidence = torch.ones_like(residuals)
    logits = head(residuals, availability, confidence)
    logits.sum().backward()
    assert logits.shape == (2,)
    assert residuals.grad is not None
    assert head.parameter_count < 100_000
    schema = residual_channel_schema("commit", "config")
    model, history = train_residual_head(
        _samples(16, 1),
        _samples(8, 2),
        output_dir=tmp_path,
        channel_schema=schema,
        source_commit="commit",
        config_sha256="config",
        epochs=3,
        hidden_size=16,
        batch_size=8,
        device="cpu",
        amp=False,
    )
    assert len(history) == 3
    assert all(np.isfinite(row["train_loss"]) for row in history)
    assert (tmp_path / "checkpoints" / "last.pt").is_file()
    assert (tmp_path / "checkpoints" / "best_validation_loss.pt").is_file()
    assert (tmp_path / "training_history.csv").is_file()
    assert (tmp_path / "training_summary.json").is_file()
    checkpoint = torch.load(
        tmp_path / "checkpoints" / "last.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert checkpoint["random_seed"] == 42
    assert checkpoint["channel_schema"] == schema
    first = evaluate_residual_head(
        model, _samples(8, 2), batch_size=8, device="cpu", amp=False
    )
    second = evaluate_residual_head(
        model, _samples(8, 2), batch_size=8, device="cpu", amp=False
    )
    assert first["logits"] == second["logits"]
    restored = ResidualTemporalHead(12, hidden_size=16)
    restored.load_state_dict(checkpoint["model_state"])
    recovered = evaluate_residual_head(
        restored, _samples(8, 2), batch_size=8, device="cpu", amp=False
    )
    assert recovered["logits"] == first["logits"]
    resumed, resumed_history = train_residual_head(
        _samples(16, 1),
        _samples(8, 2),
        output_dir=tmp_path / "resumed",
        channel_schema=schema,
        source_commit="commit",
        config_sha256="config",
        epochs=4,
        hidden_size=16,
        batch_size=8,
        resume=tmp_path / "checkpoints" / "last.pt",
        device="cpu",
        amp=False,
    )
    assert [row["epoch"] for row in resumed_history] == [4]
    assert resumed.parameter_count == model.parameter_count


def _result(path, video_path):
    clips = []
    for index in range(2):
        clips.append(
            {
                "clip_id": f"{path.stem}_clip_{index}",
                "start_frame": index * 8,
                "residuals": [
                    {
                        "name": "dynamic_reprojection",
                        "normalized_value": 0.2 + index,
                        "availability": "observed",
                        "valid_mask": True,
                        "confidence": 0.8,
                    },
                    {
                        "name": "relation",
                        "normalized_value": None,
                        "availability": "blocked_by_input",
                        "valid_mask": False,
                        "confidence": 0.0,
                    },
                ],
            }
        )
    path.write_text(
        json.dumps(
            {
                "video_path": str(video_path),
                "risk_score": 0.99,
                "suspicious_clips": [{"forbidden": True}],
                "clips": clips,
                "metadata": {
                    "authenticity_label_used": False,
                    "historical_csv_read": False,
                    "m6_to_a2_bridge_called": False,
                },
            }
        ),
        encoding="utf-8",
    )


def test_frozen_result_loader_preserves_mask_order_and_video_split(tmp_path):
    schema = residual_channel_schema("commit", "config")
    schema_path = tmp_path / "schema.json"
    schema_path.write_text(json.dumps(schema), encoding="utf-8")
    rows = []
    for sample_id, label, split in (
        ("train_real", 0, "train"),
        ("train_fake", 1, "train"),
        ("validation_real", 0, "validation"),
        ("validation_fake", 1, "validation"),
    ):
        video = tmp_path / f"{sample_id}.mp4"
        video.write_bytes(b"video")
        residual = tmp_path / f"{sample_id}.json"
        _result(residual, video)
        rows.append(
            {
                "sample_id": sample_id,
                "dataset_name": "fixture",
                "source_video_id": sample_id,
                "group_id": sample_id,
                "label": label,
                "split": split,
                "residual_sequence_path": residual,
                "source_video_path": video,
                "source_commit": "commit",
                "source_config_sha256": "config",
            }
        )
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    bundle = build_manifest_samples(manifest, schema_path)
    assert bundle.sample_ids["train"] == ("train_real", "train_fake")
    assert bundle.sample_ids["validation"] == (
        "validation_real",
        "validation_fake",
    )
    sample = bundle.samples["train"][0]
    assert sample.residuals.shape == (2, 12)
    assert sample.availability.shape == sample.confidence.shape == (2, 12)
    dynamic = RESIDUAL_NAMES.index("dynamic_reprojection")
    relation = RESIDUAL_NAMES.index("relation")
    assert np.all(sample.availability[:, dynamic])
    assert np.all(np.isfinite(sample.residuals[:, dynamic]))
    assert not np.any(sample.availability[:, relation])
    assert np.all(np.isnan(sample.residuals[:, relation]))
    assert np.all(sample.confidence[:, relation] == 0.0)
    assert len(RESIDUAL_NAMES) == 12
    assert "label" not in RESIDUAL_NAMES

    rows[-1]["source_video_path"] = rows[0]["source_video_path"]
    _result(Path(rows[-1]["residual_sequence_path"]), Path(rows[0]["source_video_path"]))
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="leakage"):
        build_manifest_samples(manifest, schema_path)
