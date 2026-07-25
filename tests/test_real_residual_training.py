import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from data.residual_dataset import (
    RESIDUAL_NAMES,
    ResidualSequence,
    ResidualSequenceDataset,
    build_manifest_samples,
    collate_residual_sequences,
    residual_channel_schema,
)
from experiments.ablation import apply_input_ablation, parse_ablation_config
from experiments.artifacts import initialize_run_artifacts
from experiments.evaluate import evaluate_checkpoint
from experiments.metrics import binary_classification_metrics
from experiments.train import train_residual_head


def _write_result(path: Path, video: Path, clip_count: int = 2) -> None:
    clips = []
    for clip_index in range(clip_count):
        clips.append(
            {
                "clip_id": f"{path.stem}_clip_{clip_index}",
                "start_frame": clip_index * 8,
                "residuals": [
                    {
                        "name": name,
                        "normalized_value": 0.1 + 0.01 * channel_index,
                        "availability": "observed",
                        "valid_mask": True,
                        "confidence": 0.8,
                    }
                    for channel_index, name in enumerate(RESIDUAL_NAMES[:9])
                ]
                + [
                    {
                        "name": name,
                        "normalized_value": None,
                        "availability": "blocked_by_input",
                        "valid_mask": False,
                        "confidence": 0.0,
                        "reason": "fixture_unavailable",
                    }
                    for name in RESIDUAL_NAMES[9:]
                ],
            }
        )
    path.write_text(
        json.dumps(
            {
                "video_path": str(video),
                "risk_score": 1.0,
                "suspicious_clips": ["ignored"],
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


def _manifest_fixture(tmp_path: Path):
    rows = []
    identities = (
        ("train_real", 0, "train", 2),
        ("train_fake", 1, "train", 3),
        ("validation_real", 0, "validation", 2),
        ("validation_fake", 1, "validation", 4),
    )
    for sample_id, label, split, clips in identities:
        video = tmp_path / f"{sample_id}.mp4"
        video.write_bytes(b"video")
        residual = tmp_path / f"{sample_id}.json"
        _write_result(residual, video, clips)
        rows.append(
            {
                "sample_id": sample_id,
                "dataset_name": "fixture",
                "source_video_id": f"source_{sample_id}",
                "group_id": f"group_{sample_id}",
                "split": split,
                "label": label,
                "residual_sequence_path": str(residual),
                "source_video_path": str(video),
                "source_commit": "frozen-commit",
                "source_config_sha256": "frozen-config",
            }
        )
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return manifest, rows


def test_generic_manifest_padding_masks_and_no_label_channel(tmp_path):
    manifest, _ = _manifest_fixture(tmp_path)
    bundle = build_manifest_samples(manifest)
    assert bundle.sample_ids["train"] == ("train_real", "train_fake")
    dataset = ResidualSequenceDataset(bundle.samples["train"])
    values, availability, confidence, padding, labels, sample_ids = (
        collate_residual_sequences([dataset[0], dataset[1]])
    )
    assert values.shape == availability.shape == confidence.shape == (2, 3, 12)
    assert padding.shape == (2, 3)
    assert padding[0].tolist() == [True, True, False]
    assert not availability[0, 2].any()
    assert np.isnan(values[0, 2].numpy()).all()
    assert labels.tolist() == [0.0, 1.0]
    assert sample_ids == ["train_real", "train_fake"]
    assert "label" not in RESIDUAL_NAMES
    assert np.isfinite(values.numpy()[availability.numpy()]).all()
    assert np.all(confidence.numpy()[~availability.numpy()] == 0)


@pytest.mark.parametrize("identity", ["group_id", "source_video_id"])
def test_manifest_rejects_group_and_source_leakage(tmp_path, identity):
    manifest, rows = _manifest_fixture(tmp_path)
    rows[-1][identity] = rows[0][identity]
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="leakage"):
        build_manifest_samples(manifest)


def test_manifest_rejects_unknown_residual_channel(tmp_path):
    manifest, rows = _manifest_fixture(tmp_path)
    residual = Path(rows[0]["residual_sequence_path"])
    payload = json.loads(residual.read_text())
    payload["clips"][0]["residuals"][0]["name"] = "silently_added_channel"
    residual.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown residual channel"):
        build_manifest_samples(manifest)


def test_metrics_single_class_auc_is_unavailable():
    metrics = binary_classification_metrics([0, 0], [-1.0, 1.0])
    assert metrics["roc_auc"] is None
    assert metrics["pr_auc"] is None
    assert metrics["unavailable_reason"] == {
        "roc_auc": "split_contains_only_one_class",
        "pr_auc": "split_contains_only_one_class",
    }
    assert np.isfinite(metrics["precision"])
    assert np.isfinite(metrics["recall"])
    perfect = binary_classification_metrics([0, 1], [-2.0, 2.0])
    assert perfect["roc_auc"] == pytest.approx(1.0)
    assert perfect["pr_auc"] == pytest.approx(1.0)


def test_artifact_snapshot_hash_checkpoint_evaluation_and_resume(tmp_path):
    manifest, _ = _manifest_fixture(tmp_path)
    bundle = build_manifest_samples(manifest)
    output = tmp_path / "run"
    initialize_run_artifacts(
        output,
        bundle=bundle,
        run_config={"training": {"epochs": 3}},
        project_root=Path(__file__).resolve().parents[1],
    )
    _, history = train_residual_head(
        bundle.samples["train"],
        bundle.samples["validation"],
        output_dir=output,
        channel_schema=bundle.channel_schema,
        source_commit=bundle.source_commit,
        source_config_sha256=bundle.source_config_sha256,
        manifest_sha256=bundle.manifest_sha256,
        epochs=3,
        hidden_size=8,
        batch_size=2,
        device="cpu",
        amp=False,
        bundle=bundle,
    )
    assert len(history) == 3
    assert (output / "logs" / "batch_history.csv").is_file()
    assert (output / "logs" / "epoch_history.csv").is_file()
    assert (output / "split_snapshot.csv").is_file()
    inputs = json.loads((output / "input_artifacts.json").read_text())
    assert len(inputs) == 4
    assert all(len(row["sha256"]) == 64 and not row["copied"] for row in inputs)
    git_state = json.loads((output / "git_state.json").read_text())
    assert len(git_state["tracked_diff_sha256"]) == 64
    assert all(len(row["sha256"]) == 64 for row in git_state["untracked_files"])
    checkpoint_path = output / "checkpoints" / "last.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    required = {
        "model_state",
        "optimizer_state",
        "amp_scaler_state",
        "epoch",
        "global_step",
        "random_seed",
        "model_config",
        "training_config",
        "channel_schema",
        "source_commit",
        "source_config_sha256",
        "manifest_sha256",
        "train_sample_ids",
        "validation_sample_ids",
        "classification_threshold",
        "best_validation_loss",
    }
    assert required.issubset(checkpoint)
    evaluated = evaluate_checkpoint(
        manifest_path=manifest,
        checkpoint_path=checkpoint_path,
        split="validation",
        output_dir=output / "evaluation",
        batch_size=2,
        device="cpu",
    )
    assert [row["sample_id"] for row in evaluated["predictions"]] == list(
        bundle.sample_ids["validation"]
    )
    assert (output / "evaluation" / "predictions.csv").is_file()
    _, resumed = train_residual_head(
        bundle.samples["train"],
        bundle.samples["validation"],
        output_dir=tmp_path / "resume",
        channel_schema=bundle.channel_schema,
        source_commit=bundle.source_commit,
        source_config_sha256=bundle.source_config_sha256,
        manifest_sha256=bundle.manifest_sha256,
        epochs=4,
        hidden_size=8,
        batch_size=2,
        device="cpu",
        amp=False,
        resume=checkpoint_path,
    )
    assert [row["epoch"] for row in resumed] == [4]


def test_ablation_masks_copies_without_label_conditioning():
    values = np.ones((2, 12), dtype=np.float32)
    valid = np.ones((2, 12), dtype=bool)
    confidence = np.full((2, 12), 0.7, dtype=np.float32)
    original = ResidualSequence(values, valid, confidence, label=1)
    config = parse_ablation_config({"use_d3": False, "use_confidence": False})
    masked, changed = apply_input_ablation(original, config)
    assert changed == ("use_d3", "use_confidence")
    assert np.all(masked.availability[:, :9])
    assert not np.any(masked.availability[:, 9:])
    assert np.isnan(masked.residuals[:, 9:]).all()
    assert np.all(masked.confidence[:, :9] == 1.0)
    assert np.all(original.availability)
