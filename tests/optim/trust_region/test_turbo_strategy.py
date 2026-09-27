"""Tests for the stateful TuRBO acquisition search strategy."""

import pytest
import torch
from botorch.acquisition.analytic import PosteriorMean

from robotorchan.models import SingleTaskGP
from robotorchan.optim import TuRBOState, TuRBOStrategy, update_turbo_state


def _problem(input_dim: int = 4):
    torch.manual_seed(19)
    train_X = torch.rand(10, input_dim, dtype=torch.double)
    train_Y = -((train_X[:, :2] - 0.7) ** 2).sum(dim=-1, keepdim=True)
    bounds = torch.stack(
        [
            torch.zeros(input_dim, dtype=torch.double),
            torch.ones(input_dim, dtype=torch.double),
        ]
    )
    return train_X, train_Y, bounds


def _center(train_X: torch.Tensor, train_Y: torch.Tensor) -> torch.Tensor:
    return train_X[train_Y.squeeze(-1).argmax()]


def test_state_expands_after_success_tolerance() -> None:
    state = TuRBOState(length=0.4, success_tolerance=2, best_value=0.0)
    state = update_turbo_state(state, torch.tensor([1.0]))
    assert state.length == pytest.approx(0.4)
    assert state.success_counter == 1

    state = update_turbo_state(state, torch.tensor([2.0]))
    assert state.length == pytest.approx(0.8)
    assert state.success_counter == 0
    assert state.failure_counter == 0
    assert state.best_value == pytest.approx(2.0)


def test_state_shrinks_and_triggers_restart() -> None:
    state = TuRBOState(length=0.2, length_min=0.15, failure_tolerance=2, best_value=1.0)
    state = update_turbo_state(state, torch.tensor([0.0]))
    assert state.failure_counter == 1

    state = update_turbo_state(state, torch.tensor([0.0]))
    assert state.length == pytest.approx(0.1)
    assert state.failure_counter == 0
    assert state.restart_triggered


def test_terminal_state_below_minimum_requires_restart_flag() -> None:
    state = TuRBOState(length=0.1, length_min=0.15, restart_triggered=True)
    assert state.restart_triggered
    with pytest.raises(ValueError, match="requires restart_triggered"):
        TuRBOState(length=0.1, length_min=0.15)


def test_trust_region_is_clipped_to_public_bounds() -> None:
    _, _, bounds = _problem()
    center = torch.tensor([0.1, 0.9, 0.5, 0.5], dtype=torch.double)
    strategy = TuRBOStrategy(bounds, center=center, state=TuRBOState(length=0.8))

    trust_bounds = strategy.trust_region_bounds()

    assert torch.all(bounds[0] <= trust_bounds[0])
    assert torch.all(bounds[1] >= trust_bounds[1])
    torch.testing.assert_close(trust_bounds[0, :2], torch.tensor([0.0, 0.5], dtype=torch.double))
    torch.testing.assert_close(trust_bounds[1, :2], torch.tensor([0.5, 1.0], dtype=torch.double))


def test_optimize_uses_explicit_incumbent_as_center() -> None:
    train_X, train_Y, bounds = _problem()
    model = SingleTaskGP(train_X, train_Y)
    acquisition = PosteriorMean(model)
    center = _center(train_X, train_Y)
    strategy = TuRBOStrategy(bounds, center=center, num_restarts=2, raw_samples=16)

    result = strategy.optimize(acquisition, q=1)

    torch.testing.assert_close(result.metadata["trust_region_center"], center)
    trust_bounds = result.metadata["trust_region_bounds"]
    assert torch.all(trust_bounds[0] <= result.candidates)
    assert torch.all(trust_bounds[1] >= result.candidates)
    assert result.acquisition_value is not None


def test_strategy_update_state_persists_state() -> None:
    _, _, bounds = _problem()
    strategy = TuRBOStrategy(
        bounds,
        center=bounds.mean(dim=0),
        state=TuRBOState(success_tolerance=1, best_value=0.0),
    )

    state = strategy.update_state(torch.tensor([1.0]))

    assert strategy.state is state
    assert state.length == pytest.approx(1.6)
    assert state.best_value == pytest.approx(1.0)


def test_validates_arguments_and_restart_state() -> None:
    train_X, train_Y, bounds = _problem()
    center = _center(train_X, train_Y)
    with pytest.raises(ValueError, match="num_restarts"):
        TuRBOStrategy(bounds, center=center, num_restarts=0)
    with pytest.raises(ValueError, match="raw_samples"):
        TuRBOStrategy(bounds, center=center, raw_samples=0)
    with pytest.raises(ValueError, match="center must have shape"):
        TuRBOStrategy(bounds, center=torch.zeros(3, dtype=torch.double))
    with pytest.raises(ValueError, match="values must contain"):
        update_turbo_state(TuRBOState(), torch.tensor([]))

    state = TuRBOState(length=0.1, length_min=0.1, restart_triggered=True)
    strategy = TuRBOStrategy(bounds, center=center, state=state)
    acquisition = PosteriorMean(SingleTaskGP(train_X, train_Y))
    with pytest.raises(RuntimeError, match="restart is required"):
        strategy.optimize(acquisition)
