"""Integration regressions for TuRBO public-space state handling."""

import pytest
import torch
from botorch.acquisition.analytic import PosteriorMean

from robotorchan.models import PCAGP
from robotorchan.optim import TuRBOState, TuRBOStrategy, update_turbo_state


def test_first_observation_initializes_turbo_state_as_success() -> None:
    state = update_turbo_state(TuRBOState(), torch.tensor([0.5]))

    assert state.best_value == pytest.approx(0.5)
    assert state.success_counter == 1
    assert state.failure_counter == 0


def test_reduced_gp_uses_explicit_public_space_incumbent() -> None:
    torch.manual_seed(23)
    input_dim = 6
    train_X = torch.rand(12, input_dim, dtype=torch.double)
    train_Y = -((train_X[:, :2] - 0.65) ** 2).sum(dim=-1, keepdim=True)
    bounds = torch.stack(
        [
            torch.zeros(input_dim, dtype=torch.double),
            torch.ones(input_dim, dtype=torch.double),
        ]
    )
    model = PCAGP(train_X, train_Y, n_components=3)
    acquisition = PosteriorMean(model)
    incumbent = train_X[train_Y.squeeze(-1).argmax()]
    strategy = TuRBOStrategy(
        bounds,
        center=incumbent,
        num_restarts=2,
        raw_samples=16,
    )

    result = strategy.optimize(acquisition)

    torch.testing.assert_close(result.metadata["trust_region_center"], incumbent)
    assert result.candidates.shape == (1, input_dim)


def test_update_state_moves_incumbent_when_candidate_improves() -> None:
    bounds = torch.stack([torch.zeros(3), torch.ones(3)])
    initial = torch.tensor([0.2, 0.3, 0.4])
    improved = torch.tensor([[0.8, 0.7, 0.6]])
    strategy = TuRBOStrategy(
        bounds,
        center=initial,
        state=TuRBOState(best_value=0.0),
    )

    strategy.update_state(torch.tensor([1.0]), candidates=improved)

    torch.testing.assert_close(strategy.center, improved[0])
