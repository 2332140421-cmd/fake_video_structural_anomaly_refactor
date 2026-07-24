"""P4-C3C-A2 synthetic-only minimal training loop tests."""

from __future__ import annotations

import copy
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from semantic3d.minimal_training.checkpoint import restore_training_checkpoint
from semantic3d.minimal_training.contracts import (
    EVIDENCE_MANIFEST_SCHEMA_VERSION,
    TRAINING_CHECKPOINT_SCHEMA_VERSION,
    load_feature_contract,
)
from semantic3d.minimal_training.data import (
    EvidenceTrainingDataset,
    build_training_dataloaders,
    collate_evidence_samples,
)
from semantic3d.minimal_training.engine import (
    load_training_config,
    run_training,
    set_reproducible_seed,
    train_one_epoch,
    validate_one_epoch,
    with_config_overrides,
)
from semantic3d.minimal_training.loss import MaskedBinaryLoss
from semantic3d.minimal_training.metrics import binary_validation_metrics
from semantic3d.minimal_training.model import MinimalMissingAwareEvidenceHead
from semantic3d.minimal_training.synthetic import (
    SyntheticFixturePaths,
    create_synthetic_fixture,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = PROJECT_ROOT / "configs/p4c3c_a2_m6_feature_contract_v1.yaml"
CONFIG_PATH = PROJECT_ROOT / "configs/p4c3c_a2_minimal_training_v1.yaml"


@pytest.fixture
def synthetic_fixture(tmp_path: Path) -> SyntheticFixturePaths:
    return create_synthetic_fixture(
        tmp_path / "fixture",
        feature_contract_path=CONTRACT_PATH,
    )


def _dataset(paths: SyntheticFixturePaths, split: str) -> EvidenceTrainingDataset:
    return EvidenceTrainingDataset(
        formal_manifest=getattr(paths, f"{split}_formal_manifest"),
        evidence_manifest=getattr(paths, f"{split}_manifest"),
        feature_contract=CONTRACT_PATH,
        expected_split=split,
    )


def _config(
    paths: SyntheticFixturePaths,
    output_dir: Path,
    **overrides: object,
):
    config = load_training_config(CONFIG_PATH, project_root=PROJECT_ROOT)
    values = {
        "train_formal_manifest": paths.train_formal_manifest,
        "train_manifest": paths.train_manifest,
        "validation_formal_manifest": paths.validation_formal_manifest,
        "validation_manifest": paths.validation_manifest,
        "output_dir": output_dir,
        "device": "cpu",
        "epochs": 1,
        "batch_size": 2,
        "num_workers": 0,
        "deterministic": True,
        "amp": False,
    }
    values.update(overrides)
    return with_config_overrides(config, **values)


def _batch(dataset: EvidenceTrainingDataset) -> dict[str, object]:
    return collate_evidence_samples([dataset[index] for index in range(len(dataset))])


def _model() -> MinimalMissingAwareEvidenceHead:
    contract = load_feature_contract(CONTRACT_PATH)
    return MinimalMissingAwareEvidenceHead(
        branch_count=contract.branch_count,
        feature_dim=contract.feature_dim,
        hidden_dim=8,
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_feature_contract_matches_frozen_m6_branches() -> None:
    contract = load_feature_contract(CONTRACT_PATH)
    assert contract.branch_count == 9
    assert contract.feature_names == ("bounded_risk",)
    assert contract.reliability_names == ("confidence",)
    assert contract.raw["invariants"]["provider_failure_is_anomaly_evidence"] is False
    assert contract.raw["invariants"]["test_split_allowed"] is False


def test_dataset_reads_a1_and_precomputed_evidence(
    synthetic_fixture: SyntheticFixturePaths,
) -> None:
    dataset = _dataset(synthetic_fixture, "train")
    sample = dataset[0]
    assert len(dataset) == 5
    assert sample["features"].shape == (9, 1)
    assert sample["observability"].shape == (9,)
    assert sample["source_dataset"] == "p4c3c_a2_synthetic_engineering"
    assert sample["generator"] == "deterministic_fixture"
    assert sample["metadata"]["evidence_metadata"]["video_decoded"] is False


def test_train_validation_loaders_are_isolated(
    synthetic_fixture: SyntheticFixturePaths,
) -> None:
    bundle = build_training_dataloaders(
        train_formal_manifest=synthetic_fixture.train_formal_manifest,
        train_manifest=synthetic_fixture.train_manifest,
        validation_formal_manifest=synthetic_fixture.validation_formal_manifest,
        validation_manifest=synthetic_fixture.validation_manifest,
        feature_contract=CONTRACT_PATH,
        batch_size=2,
        num_workers=0,
        seed=17,
    )
    assert not (
        bundle.train_dataset.sample_ids & bundle.validation_dataset.sample_ids
    )
    assert {item["split"] for item in bundle.train_dataset} == {"train"}
    assert {item["split"] for item in bundle.validation_dataset} == {"validation"}


def test_test_split_is_rejected(
    synthetic_fixture: SyntheticFixturePaths,
) -> None:
    formal_rows = _read_jsonl(synthetic_fixture.train_formal_manifest)
    evidence_rows = _read_jsonl(synthetic_fixture.train_manifest)
    formal_rows[0]["split"] = "test"
    evidence_rows[0]["split"] = "test"
    _write_jsonl(synthetic_fixture.train_formal_manifest, formal_rows)
    _write_jsonl(synthetic_fixture.train_manifest, evidence_rows)
    with pytest.raises(ValueError, match="refuses test split|Split mismatch"):
        _dataset(synthetic_fixture, "train")


def test_duplicate_sample_id_is_rejected(
    synthetic_fixture: SyntheticFixturePaths,
) -> None:
    rows = _read_jsonl(synthetic_fixture.train_manifest)
    rows.append(copy.deepcopy(rows[0]))
    _write_jsonl(synthetic_fixture.train_manifest, rows)
    with pytest.raises(ValueError, match="Duplicate sample_id"):
        _dataset(synthetic_fixture, "train")


def test_train_validation_sample_id_leakage_is_rejected(
    synthetic_fixture: SyntheticFixturePaths,
) -> None:
    train_formal = _read_jsonl(synthetic_fixture.train_formal_manifest)
    train_evidence = _read_jsonl(synthetic_fixture.train_manifest)
    validation_formal = _read_jsonl(synthetic_fixture.validation_formal_manifest)
    validation_evidence = _read_jsonl(synthetic_fixture.validation_manifest)
    validation_formal[0]["sample_id"] = train_formal[0]["sample_id"]
    validation_evidence[0]["sample_id"] = train_evidence[0]["sample_id"]
    _write_jsonl(synthetic_fixture.validation_formal_manifest, validation_formal)
    _write_jsonl(synthetic_fixture.validation_manifest, validation_evidence)
    with pytest.raises(ValueError, match="leakage"):
        build_training_dataloaders(
            train_formal_manifest=synthetic_fixture.train_formal_manifest,
            train_manifest=synthetic_fixture.train_manifest,
            validation_formal_manifest=synthetic_fixture.validation_formal_manifest,
            validation_manifest=synthetic_fixture.validation_manifest,
            feature_contract=CONTRACT_PATH,
            batch_size=2,
            num_workers=0,
            seed=1,
        )


def test_missing_label_has_explicit_mask(
    synthetic_fixture: SyntheticFixturePaths,
) -> None:
    batch = _batch(_dataset(synthetic_fixture, "train"))
    assert int((~batch["label_mask"]).sum()) == 1
    unlabeled = batch["sample_ids"].index("train-unlabeled")
    assert not batch["label_mask"][unlabeled]
    assert batch["labels"][unlabeled] == 0


def test_partial_missing_evidence_keeps_masks(
    synthetic_fixture: SyntheticFixturePaths,
) -> None:
    dataset = _dataset(synthetic_fixture, "train")
    sample = next(item for item in dataset if item["sample_id"] == "train-partial")
    assert int(sample["feature_mask"].sum()) == 1
    assert int(sample["missing_mask"].sum()) == 8
    assert torch.equal(sample["missing_mask"], ~sample["feature_mask"])
    assert sample["observability"][:2].tolist() == [True, True]


def test_all_missing_evidence_is_marked_invalid(
    synthetic_fixture: SyntheticFixturePaths,
) -> None:
    dataset = _dataset(synthetic_fixture, "train")
    sample = next(
        item for item in dataset if item["sample_id"] == "train-no-evidence"
    )
    output = _model()(
        features=sample["features"].unsqueeze(0),
        feature_mask=sample["feature_mask"].unsqueeze(0),
        observability=sample["observability"].unsqueeze(0),
        reliability=sample["reliability"].unsqueeze(0),
    )
    assert output["valid_sample_mask"].tolist() == [False]
    assert torch.isnan(output["logits"]).all()


def test_collate_preserves_fixed_shapes_and_metadata(
    synthetic_fixture: SyntheticFixturePaths,
) -> None:
    batch = _batch(_dataset(synthetic_fixture, "validation"))
    assert batch["features"].shape == (5, 9, 1)
    assert batch["feature_mask"].dtype == torch.bool
    assert batch["observability"].shape == (5, 9)
    assert len(batch["metadata"]) == 5


def test_collate_rejects_shape_mismatch_without_truncation(
    synthetic_fixture: SyntheticFixturePaths,
) -> None:
    dataset = _dataset(synthetic_fixture, "train")
    bad = dict(dataset[0])
    bad["features"] = bad["features"][:-1]
    with pytest.raises(ValueError, match="never truncates"):
        collate_evidence_samples([dataset[1], bad])


def test_parquet_evidence_manifest_is_supported(
    synthetic_fixture: SyntheticFixturePaths,
    tmp_path: Path,
) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    parquet_path = tmp_path / "evidence.parquet"
    pq.write_table(
        pa.Table.from_pylist(_read_jsonl(synthetic_fixture.train_manifest)),
        parquet_path,
    )
    dataset = EvidenceTrainingDataset(
        formal_manifest=synthetic_fixture.train_formal_manifest,
        evidence_manifest=parquet_path,
        feature_contract=CONTRACT_PATH,
        expected_split="train",
    )
    assert len(dataset) == 5


def test_model_forward_shape_and_dtype(
    synthetic_fixture: SyntheticFixturePaths,
) -> None:
    batch = _batch(_dataset(synthetic_fixture, "train"))
    output = _model()(
        features=batch["features"],
        feature_mask=batch["feature_mask"],
        observability=batch["observability"],
        reliability=batch["reliability"],
    )
    assert output["logits"].shape == (5,)
    assert output["logits"].dtype == torch.float32
    assert output["anomaly_probability"].shape == (5,)
    assert torch.all(
        (output["anomaly_probability"][output["valid_sample_mask"]] >= 0)
        & (output["anomaly_probability"][output["valid_sample_mask"]] <= 1)
    )
    assert output["valid_sample_mask"].dtype == torch.bool


def test_model_rejects_nan_in_observed_feature(
    synthetic_fixture: SyntheticFixturePaths,
) -> None:
    batch = _batch(_dataset(synthetic_fixture, "train"))
    batch["features"][0, 0, 0] = float("nan")
    batch["feature_mask"][0, 0, 0] = True
    with pytest.raises(ValueError, match="NaN or Inf"):
        _model()(
            features=batch["features"],
            feature_mask=batch["feature_mask"],
            observability=batch["observability"],
            reliability=batch["reliability"],
        )


def test_masked_bce_loss_is_finite(
    synthetic_fixture: SyntheticFixturePaths,
) -> None:
    batch = _batch(_dataset(synthetic_fixture, "train"))
    output = _model()(
        features=batch["features"],
        feature_mask=batch["feature_mask"],
        observability=batch["observability"],
        reliability=batch["reliability"],
    )
    result = MaskedBinaryLoss()(
        logits=output["logits"],
        labels=batch["labels"],
        label_mask=batch["label_mask"],
        valid_sample_mask=output["valid_sample_mask"],
    )
    assert result.loss is not None and torch.isfinite(result.loss)
    assert result.supervised_count == 3
    assert result.missing_label_count == 1
    assert result.no_evidence_count == 1


def test_backward_produces_finite_gradients(
    synthetic_fixture: SyntheticFixturePaths,
) -> None:
    batch = _batch(_dataset(synthetic_fixture, "train"))
    model = _model()
    output = model(
        features=batch["features"],
        feature_mask=batch["feature_mask"],
        observability=batch["observability"],
        reliability=batch["reliability"],
    )
    result = MaskedBinaryLoss()(
        logits=output["logits"],
        labels=batch["labels"],
        label_mask=batch["label_mask"],
        valid_sample_mask=output["valid_sample_mask"],
    )
    assert result.loss is not None
    result.loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_gradient_clipping_is_applied(
    synthetic_fixture: SyntheticFixturePaths,
) -> None:
    batch = _batch(_dataset(synthetic_fixture, "train"))
    model = _model()
    output = model(
        features=batch["features"],
        feature_mask=batch["feature_mask"],
        observability=batch["observability"],
        reliability=batch["reliability"],
    )
    result = MaskedBinaryLoss()(
        logits=output["logits"] * 100,
        labels=batch["labels"],
        label_mask=batch["label_mask"],
        valid_sample_mask=output["valid_sample_mask"],
    )
    result.loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)
    assert max(
        float(parameter.grad.norm())
        for parameter in model.parameters()
        if parameter.grad is not None
    ) <= 0.010001


def test_no_supervision_batch_does_not_step_optimizer(
    synthetic_fixture: SyntheticFixturePaths,
) -> None:
    dataset = _dataset(synthetic_fixture, "train")
    no_evidence = next(
        item for item in dataset if item["sample_id"] == "train-no-evidence"
    )
    second_no_evidence = dict(no_evidence)
    second_no_evidence["sample_id"] = "train-no-evidence-control"
    samples = [no_evidence, second_no_evidence]
    loader = DataLoader(samples, batch_size=2, collate_fn=collate_evidence_samples)
    model = _model()
    before = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    metrics, global_step = train_one_epoch(
        model=model,
        loader=loader,
        optimizer=optimizer,
        criterion=MaskedBinaryLoss(),
        scaler=scaler,
        device=torch.device("cpu"),
        amp_enabled=False,
        gradient_clip_norm=1.0,
        global_step=0,
    )
    assert global_step == 0
    assert metrics["skipped_batch_count"] == 1
    assert metrics["no_evidence_count"] == 2
    assert metrics["supervised_sample_count"] == 0
    assert all(torch.equal(before[key], model.state_dict()[key]) for key in before)


def test_pos_weight_is_derived_from_train_split_only(
    synthetic_fixture: SyntheticFixturePaths,
    tmp_path: Path,
) -> None:
    result = run_training(
        _config(
            synthetic_fixture,
            tmp_path / "pos-weight",
            pos_weight="train_split",
            max_train_batches=1,
            max_validation_batches=1,
        )
    )
    assert result["global_step"] == 1


def test_cpu_single_step_training(
    synthetic_fixture: SyntheticFixturePaths,
    tmp_path: Path,
) -> None:
    result = run_training(
        _config(
            synthetic_fixture,
            tmp_path / "cpu-step",
            max_train_batches=1,
            max_validation_batches=1,
        )
    )
    assert result["device"] == "cpu"
    assert result["global_step"] == 1
    assert result["test_split_loaded"] is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_single_step_training(
    synthetic_fixture: SyntheticFixturePaths,
    tmp_path: Path,
) -> None:
    result = run_training(
        _config(
            synthetic_fixture,
            tmp_path / "cuda-step",
            device="cuda",
            max_train_batches=1,
            max_validation_batches=1,
        )
    )
    assert result["device"].startswith("cuda")
    assert result["global_step"] == 1
    assert result["amp_enabled"] is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_amp_single_step_training(
    synthetic_fixture: SyntheticFixturePaths,
    tmp_path: Path,
) -> None:
    output = tmp_path / "cuda-amp-step"
    first = run_training(
        _config(
            synthetic_fixture,
            output,
            device="cuda",
            amp=True,
            max_train_batches=1,
            max_validation_batches=1,
        )
    )
    first_checkpoint = torch.load(
        output / "last_checkpoint.pt", map_location="cpu", weights_only=False
    )
    second = run_training(
        _config(
            synthetic_fixture,
            output,
            device="cuda",
            amp=True,
            epochs=2,
            max_train_batches=1,
            max_validation_batches=1,
            resume_checkpoint=output / "last_checkpoint.pt",
        )
    )
    second_checkpoint = torch.load(
        output / "last_checkpoint.pt", map_location="cpu", weights_only=False
    )
    assert first["amp_enabled"] is True
    assert first["global_step"] == 1
    assert first_checkpoint["amp_scaler_state"]
    assert second_checkpoint["amp_scaler_state"]
    assert second["completed_epochs"] == [2]
    assert second["global_step"] == 2


def test_checkpoint_saves_required_state(
    synthetic_fixture: SyntheticFixturePaths,
    tmp_path: Path,
) -> None:
    output = tmp_path / "checkpoint"
    run_training(_config(synthetic_fixture, output))
    checkpoint = torch.load(
        output / "last_checkpoint.pt",
        map_location="cpu",
        weights_only=False,
    )
    required = {
        "schema_version",
        "feature_contract_version",
        "model_state",
        "optimizer_state",
        "scheduler_state",
        "amp_scaler_state",
        "epoch",
        "global_step",
        "best_validation_metric",
        "configuration",
        "random_states",
        "git_commit",
        "train_manifest_checksum",
        "validation_manifest_checksum",
    }
    assert required <= set(checkpoint)
    assert checkpoint["schema_version"] == TRAINING_CHECKPOINT_SCHEMA_VERSION
    assert {
        "python_random_state",
        "numpy_random_state",
        "torch_cpu_random_state",
        "torch_cuda_random_state",
    } <= set(checkpoint["random_states"])


def test_resume_continues_global_step_and_metrics(
    synthetic_fixture: SyntheticFixturePaths,
    tmp_path: Path,
) -> None:
    output = tmp_path / "resume"
    first = run_training(_config(synthetic_fixture, output, epochs=1))
    first_checkpoint = torch.load(
        output / "last_checkpoint.pt", map_location="cpu", weights_only=False
    )
    second = run_training(
        _config(
            synthetic_fixture,
            output,
            epochs=2,
            resume_checkpoint=output / "last_checkpoint.pt",
        )
    )
    assert second["completed_epochs"] == [2]
    assert second["global_step"] > first["global_step"]
    assert len(_read_jsonl(output / "metrics.jsonl")) == 4
    second_checkpoint = torch.load(
        output / "last_checkpoint.pt", map_location="cpu", weights_only=False
    )
    assert first_checkpoint["optimizer_state"]["state"]
    assert second_checkpoint["optimizer_state"]["state"]
    assert second_checkpoint["epoch"] == first_checkpoint["epoch"] + 1
    assert (
        second_checkpoint["best_validation_metric"]
        <= first_checkpoint["best_validation_metric"]
    )
    assert second_checkpoint["amp_scaler_state"] == first_checkpoint["amp_scaler_state"]


def test_resume_rejects_manifest_checksum_mismatch(
    synthetic_fixture: SyntheticFixturePaths,
    tmp_path: Path,
) -> None:
    output = tmp_path / "checksum"
    run_training(_config(synthetic_fixture, output))
    original_train = synthetic_fixture.train_manifest.read_bytes()
    with synthetic_fixture.train_manifest.open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="train manifest checksum mismatch"):
        run_training(
            _config(
                synthetic_fixture,
                output,
                epochs=2,
                resume_checkpoint=output / "last_checkpoint.pt",
            )
        )
    synthetic_fixture.train_manifest.write_bytes(original_train)
    with synthetic_fixture.validation_manifest.open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="validation manifest checksum mismatch"):
        run_training(
            _config(
                synthetic_fixture,
                output,
                epochs=2,
                resume_checkpoint=output / "last_checkpoint.pt",
            )
        )


def test_resume_rejects_feature_contract_mismatch(
    synthetic_fixture: SyntheticFixturePaths,
    tmp_path: Path,
) -> None:
    output = tmp_path / "contract"
    run_training(_config(synthetic_fixture, output))
    checkpoint_path = output / "last_checkpoint.pt"
    original = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    checkpoint = copy.deepcopy(original)
    checkpoint["schema_version"] = "wrong-schema"
    torch.save(checkpoint, checkpoint_path)
    with pytest.raises(ValueError, match="schema version mismatch"):
        run_training(
            _config(
                synthetic_fixture,
                output,
                epochs=2,
                resume_checkpoint=checkpoint_path,
            )
        )

    checkpoint = copy.deepcopy(original)
    checkpoint["feature_contract_version"] = "wrong-contract"
    torch.save(checkpoint, checkpoint_path)
    with pytest.raises(ValueError, match="feature contract version mismatch"):
        run_training(
            _config(
                synthetic_fixture,
                output,
                epochs=2,
                resume_checkpoint=checkpoint_path,
            )
        )

    checkpoint = copy.deepcopy(original)
    checkpoint["feature_contract"]["feature_dim"] = 2
    torch.save(checkpoint, checkpoint_path)
    with pytest.raises(ValueError, match="feature contract mismatch"):
        run_training(
            _config(
                synthetic_fixture,
                output,
                epochs=2,
                resume_checkpoint=checkpoint_path,
            )
        )

    checkpoint = copy.deepcopy(original)
    checkpoint["model_config"]["hidden_dim"] += 1
    torch.save(checkpoint, checkpoint_path)
    with pytest.raises(ValueError, match="model config mismatch"):
        run_training(
            _config(
                synthetic_fixture,
                output,
                epochs=2,
                resume_checkpoint=checkpoint_path,
            )
        )

    checkpoint = copy.deepcopy(original)
    checkpoint["splits"] = {"train": "validation", "validation": "train"}
    torch.save(checkpoint, checkpoint_path)
    with pytest.raises(ValueError, match="split identity mismatch"):
        run_training(
            _config(
                synthetic_fixture,
                output,
                epochs=2,
                resume_checkpoint=checkpoint_path,
            )
        )


def test_fixed_seed_is_reproducible(
    synthetic_fixture: SyntheticFixturePaths,
    tmp_path: Path,
) -> None:
    first_dir = tmp_path / "seed-first"
    second_dir = tmp_path / "seed-second"
    run_training(_config(synthetic_fixture, first_dir, seed=77))
    run_training(_config(synthetic_fixture, second_dir, seed=77))
    first = torch.load(
        first_dir / "last_checkpoint.pt", map_location="cpu", weights_only=False
    )
    second = torch.load(
        second_dir / "last_checkpoint.pt", map_location="cpu", weights_only=False
    )
    assert first["global_step"] == second["global_step"]
    assert all(
        torch.equal(first["model_state"][key], second["model_state"][key])
        for key in first["model_state"]
    )


def test_validation_does_not_update_parameters_or_state(
    synthetic_fixture: SyntheticFixturePaths,
) -> None:
    dataset = _dataset(synthetic_fixture, "validation")
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_evidence_samples,
    )
    model = _model()
    before = copy.deepcopy(model.state_dict())
    metrics = validate_one_epoch(
        model=model,
        loader=loader,
        criterion=MaskedBinaryLoss(),
        device=torch.device("cpu"),
        amp_enabled=False,
    )
    assert metrics["valid_sample_count"] == 4
    assert metrics["no_evidence_count"] == 1
    assert metrics["missing_label_count"] == 1
    assert metrics["supervised_sample_count"] == 3
    for name in ("loss", "accuracy", "precision", "recall", "f1", "roc_auc"):
        assert math.isfinite(metrics[name])
    assert all(torch.equal(before[key], model.state_dict()[key]) for key in before)


def test_binary_metrics_include_auc_only_with_both_classes() -> None:
    metrics = binary_validation_metrics(
        logits=[-1.0, 1.0],
        labels=[0, 1],
        loss_sum=1.0,
        valid_sample_count=2,
        missing_label_count=0,
        no_evidence_count=0,
        skipped_batch_count=0,
    )
    assert metrics["roc_auc"] == 1.0
    one_class = binary_validation_metrics(
        logits=[1.0],
        labels=[1],
        loss_sum=0.5,
        valid_sample_count=1,
        missing_label_count=0,
        no_evidence_count=0,
        skipped_batch_count=0,
    )
    assert one_class["roc_auc"] is None
    assert metrics["classification_threshold"] == 0.5


def test_config_and_environment_snapshots_are_written(
    synthetic_fixture: SyntheticFixturePaths,
    tmp_path: Path,
) -> None:
    output = tmp_path / "snapshots"
    run_training(_config(synthetic_fixture, output))
    environment = json.loads(
        (output / "environment_snapshot.json").read_text(encoding="utf-8")
    )
    assert (output / "config_snapshot.yaml").is_file()
    assert environment["provider_inference_executed"] is False
    assert environment["video_decoded"] is False
    assert environment["formal_training_executed"] is False


def test_cli_help_uses_project_interpreter() -> None:
    result = subprocess.run(
        [
            str(PROJECT_ROOT / ".venv/bin/python"),
            str(PROJECT_ROOT / "scripts/train_p4c3c_a2_minimal.py"),
            "--help",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--synthetic-fixture" in result.stdout
    assert "--train-manifest" in result.stdout


def test_evidence_manifest_schema_version_is_required(
    synthetic_fixture: SyntheticFixturePaths,
) -> None:
    rows = _read_jsonl(synthetic_fixture.train_manifest)
    rows[0]["schema_version"] = "unknown"
    _write_jsonl(synthetic_fixture.train_manifest, rows)
    with pytest.raises(ValueError, match="schema_version"):
        _dataset(synthetic_fixture, "train")
    assert EVIDENCE_MANIFEST_SCHEMA_VERSION.endswith(".v1")
