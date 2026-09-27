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


def _state(input_dim: int, **kwargs) -> BAxUSState:
    return BAxUSState(dim=input_dim, eval_budget=100, **kwargs)


def test_state_derives_initial_dimension_and_split_schedule() -> None:
    state = BAxUSState(dim=500, eval_budget=500)

    assert state.n_splits == 4
    assert state.initial_target_dim == 2
    assert state.target_dim == 2
    assert state.split_budget > 0
    assert 1 <= state.failure_tolerance <= state.target_dim


def test_failure_tolerance_depends_on_target_dimension() -> None:
    initial = BAxUSState(dim=500, eval_budget=500)
    expanded = BAxUSState(dim=500, eval_budget=500, target_dim=8)
    full = BAxUSState(dim=500, eval_budget=500, target_dim=500)

    assert expanded.split_budget > initial.split_budget
    assert expanded.failure_tolerance >= initial.failure_tolerance
    assert full.failure_tolerance == 500


def test_seed_reproduces_initial_embedding() -> None:
    _, _, bounds = _problem()
    first = BAxUSStrategy(bounds, state=_state(6), seed=7)
    second = BAxUSStrategy(bounds, state=_state(6), seed=7)
    torch.testing.assert_close(first.embedding, second.embedding)


def test_initial_embedding_is_sparse_signed_and_balanced() -> None:
    _, _, bounds = _problem(input_dim=11)
    strategy = BAxUSStrategy(bounds, state=_state(11, target_dim=3), seed=7)
    embedding = strategy.embedding

    assert embedding.shape == (11, 3)
    assert torch.all(embedding.ne(0).sum(dim=1) == 1)
    assert set(embedding[embedding != 0].tolist()) == {-1.0, 1.0}
    occupancy = embedding.ne(0).sum(dim=0)
    assert int(occupancy.max() - occupancy.min()) <= 1


def test_project_supports_arbitrary_leading_dimensions() -> None:
    _, _, bounds = _problem()
    strategy = BAxUSStrategy(bounds, state=_state(6, target_dim=2), seed=3)
    Z = torch.randn(2, 3, 2, dtype=torch.double)
    X = strategy.project(Z)
    assert X.shape == (2, 3, 6)
    assert torch.all(bounds[0] <= X)
    assert torch.all(bounds[1] >= X)


def test_state_collapse_splits_existing_embedding_bins() -> None:
    _, _, bounds = _problem()
    state = BAxUSState(
        dim=6,
        eval_budget=100,
        new_bins_on_split=2,
        target_dim=1,
        length=0.1,
        length_min=0.15,
        best_value=1.0,
        restart_triggered=True,
    )
    strategy = BAxUSStrategy(bounds, state=state, seed=4)
    old_signs = strategy.embedding.sum(dim=1).clone()

    assert strategy.expand_subspace()
    assert strategy.target_dim == 3
    assert strategy.state.target_dim == 3
    assert strategy.state.length == pytest.approx(strategy.state.length_init)
    assert torch.all(strategy.embedding.ne(0).sum(dim=1) == 1)
    torch.testing.assert_close(strategy.embedding.sum(dim=1), old_signs)
    assert torch.all(strategy.embedding.ne(0).sum(dim=0) > 0)
    assert not strategy.state.restart_triggered
    assert strategy.state.best_value == pytest.approx(1.0)


def test_expansion_splits_every_splittable_parent_bin() -> None:
    _, _, bounds = _problem(input_dim=12)
    state = BAxUSState(
        dim=12,
        eval_budget=100,
        new_bins_on_split=2,
        target_dim=2,
        length=0.1,
        length_min=0.15,
        restart_triggered=True,
    )
    strategy = BAxUSStrategy(bounds, state=state, seed=8)
    parent_assignment = strategy.embedding.ne(0).to(torch.int64).argmax(dim=1)

    assert strategy.expand_subspace()
    assert strategy.target_dim == 6
    assert torch.all(strategy.embedding.ne(0).sum(dim=1) == 1)
    assert torch.all(strategy.embedding.ne(0).sum(dim=0) > 0)
    for parent in range(2):
        parent_embedding = strategy.embedding[parent_assignment == parent]
        child_bins = parent_embedding.ne(0).to(torch.int64).argmax(dim=1)
        assert torch.unique(child_bins).numel() == 3


def test_repeated_expansion_reaches_full_dimension_without_empty_bins() -> None:
    _, _, bounds = _problem(input_dim=7)
    state = BAxUSState(
        dim=7,
        eval_budget=100,
        target_dim=1,
        length=0.1,
        length_min=0.15,
        restart_triggered=True,
    )
    strategy = BAxUSStrategy(bounds, state=state, seed=5)

    assert strategy.expand_subspace()
    assert strategy.target_dim == 4
    strategy.state = replace_state_for_restart(strategy.state)
    assert strategy.expand_subspace()
    assert strategy.target_dim == 7
    assert torch.all(strategy.embedding.ne(0).sum(dim=0) == 1)


def replace_state_for_restart(state: BAxUSState) -> BAxUSState:
    return BAxUSState(
        dim=state.dim,
        eval_budget=state.eval_budget,
        new_bins_on_split=state.new_bins_on_split,
        target_dim=state.target_dim,
        length=state.length_min / 2,
        length_init=state.length_init,
        length_min=state.length_min,
        length_max=state.length_max,
        success_tolerance=state.success_tolerance,
        best_value=state.best_value,
        restart_triggered=True,
    )


def test_full_dimension_cannot_expand_further() -> None:
    _, _, bounds = _problem(input_dim=3)
    state = _state(3, target_dim=3, restart_triggered=True)
    strategy = BAxUSStrategy(bounds, state=state)
    assert not strategy.expand_subspace()


def test_optimize_returns_original_space_candidates() -> None:
    train_X, train_Y, bounds = _problem()
    model = SingleTaskGP(train_X, train_Y)
    acquisition = PosteriorMean(model)
    strategy = BAxUSStrategy(
        bounds,
        state=_state(6, target_dim=2),
        seed=9,
        num_restarts=2,
        raw_samples=16,
    )

    result = strategy.optimize(acquisition)

    assert result.candidates.shape == (1, bounds.shape[-1])
    assert torch.all(bounds[0] <= result.candidates)
    assert torch.all(bounds[1] >= result.candidates)
    assert result.metadata["target_dim"] == 2
    assert result.metadata["failure_tolerance"] == strategy.state.failure_tolerance
    assert result.metadata["split_budget"] == strategy.state.split_budget
    with torch.no_grad():
        expected = acquisition(result.candidates).reshape(())
    torch.testing.assert_close(result.acquisition_value, expected)


def test_first_finite_observation_is_success() -> None:
    state = _state(6, target_dim=2)

    state = update_baxus_state(state, torch.tensor([-0.5]))

    assert state.best_value == pytest.approx(-0.5)
    assert state.success_counter == 1
    assert state.failure_counter == 0


def test_update_function_expands_and_shrinks_length() -> None:
    state = _state(6, target_dim=2, length=0.4, success_tolerance=1, best_value=0.0)
    state = update_baxus_state(state, torch.tensor([1.0]))
    assert state.length == pytest.approx(0.8)

    state = _state(6, target_dim=2, length=0.4, best_value=1.0)
    for _ in range(state.failure_tolerance):
        state = update_baxus_state(state, torch.tensor([0.0]))
    assert state.length == pytest.approx(0.2)


def test_validates_state_and_strategy_contract() -> None:
    _, _, bounds = _problem()
    with pytest.raises(ValueError, match="dim"):
        BAxUSState(dim=0, eval_budget=10)
    with pytest.raises(ValueError, match="eval_budget"):
        BAxUSState(dim=6, eval_budget=0)
    with pytest.raises(ValueError, match="new_bins_on_split"):
        BAxUSState(dim=6, eval_budget=10, new_bins_on_split=0)
    with pytest.raises(ValueError, match=r"state\.dim"):
        BAxUSStrategy(bounds, state=BAxUSState(dim=5, eval_budget=10))
    strategy = BAxUSStrategy(bounds, state=_state(6))
    with pytest.raises(RuntimeError, match="requires restart_triggered"):
        strategy.expand_subspace()
