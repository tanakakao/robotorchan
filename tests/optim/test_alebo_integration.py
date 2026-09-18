"""End-to-end tests for the ALEBO model and search strategy."""

import torch
from botorch.acquisition.analytic import ExpectedImprovement

from robotorchan.models import ALEBOGP
from robotorchan.optim import ALEBOStrategy


def test_alebo_model_and_strategy_complete_one_bo_step() -> None:
    bounds = torch.stack([torch.zeros(6, dtype=torch.double), torch.ones(6, dtype=torch.double)])
    strategy = ALEBOStrategy(bounds, embedding_dim=2, seed=7, num_restarts=3, raw_samples=32)
    train_Z = torch.tensor([[0.0, 0.0], [0.25, -0.1], [-0.2, 0.2], [0.1, 0.3]], dtype=torch.double)
    train_X = strategy.project(train_Z)
    train_Y = -((train_X - 0.5) ** 2).sum(dim=-1, keepdim=True)

    model = ALEBOGP(train_X, train_Y)
    model.eval()
    acquisition = ExpectedImprovement(model=model, best_f=train_Y.max())

    result = strategy.optimize(acquisition, q=1)

    assert result.candidates.shape == (1, 6)
    assert torch.all(result.candidates >= bounds[0])
    assert torch.all(result.candidates <= bounds[1])
    embedded = result.metadata["embedded_candidates"]
    assert bool(strategy.is_feasible(embedded).all())
    assert result.acquisition_value is not None
    assert torch.isfinite(result.acquisition_value)
