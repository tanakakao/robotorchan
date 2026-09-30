"""Integration regressions for TuRBO public-space state handling."""

import pytest
import torch
from botorch.acquisition.analytic import ExpectedImprovement, PosteriorMean

from robotorchan.models import PCAGP, SingleTaskGP
from robotorchan.optim import TuRBOState, TuRBOStrategy, update_turbo_state


def test_first_observation_initializes_turbo_state_as_success() -> None:
    state = update_turbo_state(TuRBOState(dim=3), torch.tensor([0.5]))

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
        state=TuRBOState(dim=3, best_value=0.0),
    )

    strategy.update_state(torch.tensor([1.0]), candidates=improved)

    torch.testing.assert_close(strategy.center, improved[0])


def test_baseline_turbo1_runs_multiple_local_bo_iterations() -> None:
    torch.manual_seed(31)
    input_dim = 3
    bounds = torch.stack(
        [
            torch.zeros(input_dim, dtype=torch.double),
            torch.ones(input_dim, dtype=torch.double),
        ]
    )

    def objective(x: torch.Tensor) -> torch.Tensor:
        return -((x - 0.72) ** 2).sum(dim=-1, keepdim=True)

    train_X = torch.rand(8, input_dim, dtype=torch.double)
    train_Y = objective(train_X)
    best_index = train_Y.squeeze(-1).argmax()
    strategy = TuRBOStrategy(
        bounds,
        center=train_X[best_index],
        state=TuRBOState(
            dim=input_dim,
            best_value=float(train_Y[best_index].item()),
        ),
        num_restarts=2,
        raw_samples=32,
    )

    initial_best = strategy.state.best_value
    for _ in range(2):
        model = SingleTaskGP(train_X, train_Y)
        acquisition = ExpectedImprovement(
            model,
            best_f=float(train_Y.max().item()),
        )
        result = strategy.optimize(acquisition)
        new_X = result.candidates.detach()
        new_Y = objective(new_X)

        strategy.update_state(new_Y, candidates=new_X)
        train_X = torch.cat([train_X, new_X], dim=0)
        train_Y = torch.cat([train_Y, new_Y], dim=0)

        trust_bounds = result.metadata["trust_region_bounds"]
        assert torch.all(new_X >= trust_bounds[0])
        assert torch.all(new_X <= trust_bounds[1])

    assert train_X.shape == (10, input_dim)
    assert strategy.state.best_value >= initial_best
    assert not strategy.state.restart_triggered
