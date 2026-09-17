"""Tests for adaptive BAxUS acquisition search."""

import pytest
import torch
from botorch.acquisition.analytic import PosteriorMean

from robotorchan.models import SingleTaskGP
from robotorchan.optim import BAxUSState, BAxUSStrategy, update_baxus_state


def _problem(input_dim: int = 6):
    torch.manual_seed(29)
    train_X = torch.rand(12, input_dim, dtype=torch.double)
    train_Y = -((train_X[:, :2] - 0.65) ** 2).sum(dim=-1, keepdim=True)
    bounds = torch.stack(
        [torch.zeros(input_dim, dtype=torch.double), torch.ones(input_dim, dtype=torch.double)]
    )
    return train_X, train_Y, bounds


def test_seed_reproduces_initial_embedding() -> None:
    _, _, bounds = _problem()
    first = BAxUSStrategy(bounds, initial_target_dim=2, seed=7)
    second = BAxUSStrategy(bounds, initial_target_dim=2, seed=7)
    torch.testing.assert_close(first.embedding, second.embedding)


def test_project_supports_arbitrary_leading_dimensions() -> None:
    _, _, bounds = _problem()
    strategy = BAxUSStrategy(bounds, initial_target_dim=2, seed=3)
    Z = torch.randn(2, 3, 2, dtype=torch.double)
    X = strategy.project(Z)
    assert X.shape == (2, 3, 6)
    assert torch.all(bounds[0] <= X)
    assert torch.all(bounds[1] >= X)


def test_state_collapse_triggers_expansion() -> None:
    _, _, bounds = _problem()
    state = BAxUSState(
        target_dim=1,
        length=0.2,
        length_min=0.15,
        failure_tolerance=1,
        best_value=1.0,
    )
    strategy = BAxUSStrategy(bounds, initial_target_dim=1, new_dimensions=2, seed=4, state=state)
    old_embedding = strategy.embedding.clone()

    state = strategy.update_state(torch.tensor([0.0]))
    assert state.restart_triggered
    assert strategy.expand_subspace()
    assert strategy.target_dim == 3
    torch.testing.assert_close(strategy.embedding[:, :1], old_embedding)
    assert not strategy.state.restart_triggered
    assert strategy.state.best_value == pytest.approx(1.0)


def test_full_dimension_cannot_expand_further() -> None:
    _, _, bounds = _problem(input_dim=3)
    state = BAxUSState(target_dim=3, length=0.1, length_min=0.15, restart_triggered=True)
    strategy = BAxUSStrategy(bounds, initial_target_dim=3, state=state)
    assert not strategy.expand_subspace()


def test_optimize_returns_original_space_candidates() -> None:
    train_X, train_Y, bounds = _problem()
    model = SingleTaskGP(train_X, train_Y)
    acquisition = PosteriorMean(model)
    strategy = BAxUSStrategy(
        bounds,
        initial_target_dim=2,
        seed=9,
        num_restarts=2,
        raw_samples=16,
    )

    result = strategy.optimize(acquisition)

    assert result.candidates.shape == (1, bounds.shape[-1])
    assert torch.all(bounds[0] <= result.candidates)
    assert torch.all(bounds[1] >= result.candidates)
    assert result.metadata["target_dim"] == 2
    with torch.no_grad():
        expected = acquisition(result.candidates)
    torch.testing.assert_close(result.acquisition_value, expected)


def test_update_function_expands_and_shrinks_length() -> None:
    state = BAxUSState(target_dim=2, length=0.4, success_tolerance=1, best_value=0.0)
    state = update_baxus_state(state, torch.tensor([1.0]))
    assert state.length == pytest.approx(0.8)

    state = BAxUSState(target_dim=2, length=0.4, failure_tolerance=1, best_value=1.0)
    state = update_baxus_state(state, torch.tensor([0.0]))
    assert state.length == pytest.approx(0.2)


def test_validates_state_and_expansion_contract() -> None:
    _, _, bounds = _problem()
    with pytest.raises(ValueError, match="initial_target_dim"):
        BAxUSStrategy(bounds, initial_target_dim=0)
    with pytest.raises(ValueError, match=r"state\.target_dim"):
        BAxUSStrategy(bounds, initial_target_dim=2, state=BAxUSState(target_dim=1))
    with pytest.raises(RuntimeError, match="requires restart_triggered"):
        BAxUSStrategy(bounds).expand_subspace()
