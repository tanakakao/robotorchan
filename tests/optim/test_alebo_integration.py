"""End-to-end tests for the ALEBO model and search strategy."""

import torch
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.sampling.normal import SobolQMCNormalSampler

from robotorchan.models import ALEBOGP
from robotorchan.optim import ALEBOStrategy


def test_alebo_model_and_strategy_complete_one_bo_step() -> None:
    bounds = torch.stack([torch.zeros(6, dtype=torch.double), torch.ones(6, dtype=torch.double)])
    strategy = ALEBOStrategy(bounds, embedding_dim=2, seed=7, num_restarts=3, raw_samples=32)
    train_Z = torch.tensor([[0.0, 0.0], [0.25, -0.1], [-0.2, 0.2], [0.1, 0.3]], dtype=torch.double)
    train_X = strategy.project(train_Z)
    train_Y = -((train_X - 0.5) ** 2).sum(dim=-1, keepdim=True)

    model = ALEBOGP(train_Z, train_Y, torch.full_like(train_Y, 1e-6))
    model.eval()
    covariance = torch.eye(model.metric_parameter_vector().numel(), dtype=torch.double) * 0.01
    acquisition_model = model.acquisition_model(
        n_metric_samples=3,
        covariance=covariance,
        generator=torch.Generator().manual_seed(13),
    )
    acquisition = LogExpectedImprovement(model=acquisition_model, best_f=train_Y.max())

    result = strategy.optimize(acquisition, q=1)

    assert result.candidates.shape == (1, 6)
    assert torch.all(result.candidates >= bounds[0])
    assert torch.all(result.candidates <= bounds[1])
    embedded = result.metadata["embedded_candidates"]
    assert bool(strategy.is_feasible(embedded).all())
    assert result.acquisition_value is not None
    assert torch.isfinite(result.acquisition_value)


def test_alebo_metric_marginal_model_supports_batch_qlogei() -> None:
    bounds = torch.stack([torch.zeros(6, dtype=torch.double), torch.ones(6, dtype=torch.double)])
    strategy = ALEBOStrategy(
        bounds,
        embedding_dim=2,
        seed=11,
        num_restarts=2,
        raw_samples=16,
        sequential=False,
    )
    train_Z = torch.tensor(
        [[0.0, 0.0], [0.2, -0.1], [-0.15, 0.2], [0.1, 0.25]],
        dtype=torch.double,
    )
    train_X = strategy.project(train_Z)
    train_Y = -((train_X - 0.5) ** 2).sum(dim=-1, keepdim=True)
    model = ALEBOGP(train_Z, train_Y, torch.full_like(train_Y, 1e-6))
    covariance = torch.eye(model.metric_parameter_vector().numel(), dtype=torch.double) * 0.01
    acquisition_model = model.acquisition_model(
        n_metric_samples=3,
        covariance=covariance,
        generator=torch.Generator().manual_seed(37),
    )
    acquisition = qLogExpectedImprovement(
        model=acquisition_model,
        best_f=train_Y.max(),
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([32])),
    )

    result = strategy.optimize(acquisition, q=2)

    assert result.candidates.shape == (2, 6)
    embedded = result.metadata["embedded_candidates"]
    assert embedded.shape == (2, 2)
    assert bool(strategy.is_feasible(embedded).all())
    assert torch.all(result.candidates >= bounds[0])
    assert torch.all(result.candidates <= bounds[1])
    assert result.acquisition_value is not None
    assert torch.isfinite(result.acquisition_value)
