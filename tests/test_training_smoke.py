import numpy as np
import torch

from experiments.train import ResidualSequence, train_residual_head
from models.temporal_head import ResidualTemporalHead


def _samples(count, seed):
    random = np.random.default_rng(seed)
    output = []
    for index in range(count):
        label = index % 2
        steps = 3 + index % 2
        values = random.normal(0.2 + 0.6 * label, 0.05, size=(steps, 6)).astype(np.float32)
        availability = random.random((steps, 6)) > 0.15
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
    model, history = train_residual_head(
        _samples(16, 1),
        _samples(8, 2),
        output_dir=tmp_path,
        epochs=3,
        hidden_size=16,
        batch_size=8,
    )
    assert len(history) == 3
    assert all(np.isfinite(row["train"]["loss"]) for row in history)
    assert (tmp_path / "last.pt").is_file()
    assert (tmp_path / "best.pt").is_file()
    resumed, resumed_history = train_residual_head(
        _samples(16, 3),
        _samples(8, 4),
        output_dir=tmp_path / "resumed",
        epochs=3,
        hidden_size=16,
        batch_size=8,
        resume=tmp_path / "last.pt",
    )
    assert len(resumed_history) == 3
    assert resumed.parameter_count == model.parameter_count
