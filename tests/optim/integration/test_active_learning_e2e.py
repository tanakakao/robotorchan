"""Representative end-to-end workflows for regression active learning."""

import torch

from robotorchan.acquisition import PosteriorVariance, Straddle
from robotorchan.models import SingleTaskGP
from robotorchan.optim.backends import optimize_acqf_botorch


def _model() -> SingleTaskGP:
    train_x = torch.linspace(0.0, 1.0, 8, dtype=torch.double).unsqueeze(-1)
    train_y = (train_x - 0.5).square()
    model = SingleTaskGP(train_x, train_y)
    model.eval()
    return model


def _optimize(acquisition) -> tuple[torch.Tensor, torch.Tensor]:
    return optimize_acqf_botorch(
        acquisition,
        torch.tensor([[0.0], [1.0]], dtype=torch.double),
        q=1,
        num_restarts=3,
        raw_samples=32,
    )


def test_posterior_variance_active_learning_runs_through_optimizer() -> None:
    candidate, value = _optimize(PosteriorVariance(_model()))

    assert candidate.shape == torch.Size([1, 1])
    assert value.numel() == 1
    assert torch.isfinite(candidate).all()
    assert torch.isfinite(value).all()


def test_straddle_level_set_learning_runs_through_optimizer() -> None:
    candidate, value = _optimize(Straddle(_model(), target=0.1))

    assert candidate.shape == torch.Size([1, 1])
    assert value.numel() == 1
    assert torch.isfinite(candidate).all()
    assert torch.isfinite(value).all()
