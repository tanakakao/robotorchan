"""Tests for BAxUS Thompson-sampling candidate generation."""

import pytest
import torch
from botorch.acquisition.analytic import PosteriorMean

from robotorchan.models import SingleTaskGP
from robotorchan.optim import BAxUSState, BAxUSThompsonSamplingStrategy


def _problem(input_dim: int = 24):
    torch.manual_seed(41)
    train_X = torch.rand(16, input_dim, dtype=torch.double)
    train_Y = -((train_X[:, :2] - 0.6) ** 2).sum(dim=-1, keepdim=True)
    bounds = torch.stack(
        [torch.zeros(input_dim, dtype=torch.double), torch.ones(input_dim, dtype=torch.double)]
    )
    model = SingleTaskGP(train_X, train_Y)
    return bounds, PosteriorMean(model)


def test_default_candidate_budget_tracks_current_target_dimension() -> None:
    bounds, _ = _problem(input_dim=30)
    small_target = BAxUSThompsonSamplingStrategy(
        bounds,
        state=BAxUSState(dim=30, eval_budget=40, target_dim=2),
    )
    medium_target = BAxUSThompsonSamplingStrategy(
        bounds,
        state=BAxUSState(dim=30, eval_budget=40, target_dim=15),
    )
    full_target = BAxUSThompsonSamplingStrategy(
        bounds,
        state=BAxUSState(dim=30, eval_budget=40, target_dim=30),
    )

    assert small_target.n_candidates == 2000
    assert medium_target.n_candidates == 3000
    assert full_target.n_candidates == 5000


def test_default_candidate_budget_updates_after_subspace_expansion() -> None:
    bounds, _ = _problem(input_dim=30)
    strategy = BAxUSThompsonSamplingStrategy(
        bounds,
        state=BAxUSState(dim=30, eval_budget=40, target_dim=2, restart_triggered=True),
        seed=3,
    )

    assert strategy.n_candidates == 2000
    strategy.expand_subspace()

    assert strategy.target_dim == 8
    assert strategy.n_candidates == 2000


def test_explicit_candidate_budget_stays_fixed_across_expansion() -> None:
    bounds, _ = _problem(input_dim=30)
    strategy = BAxUSThompsonSamplingStrategy(
        bounds,
        state=BAxUSState(dim=30, eval_budget=40, target_dim=2, restart_triggered=True),
        seed=3,
        n_candidates=64,
    )

    strategy.expand_subspace()

    assert strategy.n_candidates == 64


def test_candidate_pool_uses_sparse_trust_region_perturbations() -> None:
    bounds, acquisition = _problem()
    strategy = BAxUSThompsonSamplingStrategy(
        bounds,
        state=BAxUSState(dim=24, eval_budget=40, target_dim=24),
        seed=3,
        n_candidates=128,
    )
    weights = strategy._target_lengthscale_weights(acquisition)
    target_bounds = strategy._target_bounds(weights)

    target_pool, input_pool = strategy._candidate_pool(target_bounds)

    assert target_pool.shape == (128, 24)
    assert input_pool.shape == (128, 24)
    assert torch.all(target_pool >= target_bounds[0])
    assert torch.all(target_pool <= target_bounds[1])
    changed = target_pool != strategy.target_center
    assert torch.all(changed.sum(dim=-1) >= 1)
    assert torch.any(changed.sum(dim=-1) < strategy.target_dim)


def test_optimize_returns_matching_target_candidate_and_metadata() -> None:
    bounds, acquisition = _problem(input_dim=8)
    strategy = BAxUSThompsonSamplingStrategy(
        bounds,
        state=BAxUSState(dim=8, eval_budget=20, target_dim=2),
        seed=7,
        n_candidates=64,
    )

    result = strategy.optimize(acquisition)

    assert result.candidates.shape == (1, 8)
    assert result.metadata["target_candidates"].shape == (1, 2)
    projected = strategy.project(result.metadata["target_candidates"])
    torch.testing.assert_close(projected, result.candidates)
    assert result.metadata["candidate_generation"] == "thompson_sampling"
    assert result.metadata["n_candidates"] == 64


def test_thompson_strategy_validates_candidate_count_and_q() -> None:
    bounds, acquisition = _problem(input_dim=6)
    state = BAxUSState(dim=6, eval_budget=20, target_dim=2)

    with pytest.raises(ValueError, match="n_candidates"):
        BAxUSThompsonSamplingStrategy(bounds, state=state, n_candidates=0)

    strategy = BAxUSThompsonSamplingStrategy(bounds, state=state, n_candidates=2)
    with pytest.raises(ValueError, match="q must not exceed"):
        strategy.optimize(acquisition, q=3)
