"""Regression tests for BAxUS lengthscale-weighted target trust regions."""

import torch
from botorch.acquisition.analytic import PosteriorMean

from robotorchan.models import SingleTaskGP
from robotorchan.optim import BAxUSState, BAxUSStrategy


def _problem(input_dim: int = 6):
    torch.manual_seed(41)
    train_X = torch.rand(12, input_dim, dtype=torch.double)
    train_Y = -((train_X[:, :2] - 0.6) ** 2).sum(dim=-1, keepdim=True)
    bounds = torch.stack(
        [torch.zeros(input_dim, dtype=torch.double), torch.ones(input_dim, dtype=torch.double)]
    )
    model = SingleTaskGP(train_X, train_Y)
    return model, bounds


def test_target_weights_are_induced_by_original_ard_metric() -> None:
    model, bounds = _problem()
    lengthscales = torch.tensor([0.2, 0.4, 0.8, 1.6, 3.2, 6.4], dtype=torch.double)
    model.covar_module.lengthscale = lengthscales
    acquisition = PosteriorMean(model)
    strategy = BAxUSStrategy(
        bounds,
        state=BAxUSState(dim=6, eval_budget=40, target_dim=2),
        seed=7,
    )

    weights = strategy._target_lengthscale_weights(acquisition)

    precision = strategy.embedding.abs().transpose(-2, -1) @ lengthscales.pow(-2)
    effective = precision.rsqrt()
    expected = effective / effective.mean()
    expected = expected / torch.prod(expected.pow(1.0 / strategy.target_dim))
    torch.testing.assert_close(weights, expected)
    torch.testing.assert_close(weights.prod(), torch.ones((), dtype=torch.double))


def test_weighted_bounds_are_centered_on_best_target_observation() -> None:
    model, bounds = _problem()
    model.covar_module.lengthscale = torch.tensor(
        [0.2, 0.4, 0.8, 1.6, 3.2, 6.4], dtype=torch.double
    )
    acquisition = PosteriorMean(model)
    strategy = BAxUSStrategy(
        bounds,
        state=BAxUSState(dim=6, eval_budget=40, target_dim=2, length=0.3),
        seed=8,
    )
    target_X = torch.tensor([[-0.5, 0.2], [0.4, -0.3]], dtype=torch.double)
    target_Y = torch.tensor([-1.0, 0.5], dtype=torch.double)
    strategy.update_state(target_Y, target_candidates=target_X)
    weights = strategy._target_lengthscale_weights(acquisition)

    weighted_bounds = strategy._target_bounds(weights)

    center = target_X[1]
    expected = torch.stack(
        [
            (center - weights * strategy.state.length).clamp_min(-1.0),
            (center + weights * strategy.state.length).clamp_max(1.0),
        ]
    )
    torch.testing.assert_close(weighted_bounds, expected)


def test_optimize_reports_weighted_target_geometry() -> None:
    model, bounds = _problem()
    model.covar_module.lengthscale = torch.tensor(
        [0.2, 0.4, 0.8, 1.6, 3.2, 6.4], dtype=torch.double
    )
    acquisition = PosteriorMean(model)
    strategy = BAxUSStrategy(
        bounds,
        state=BAxUSState(dim=6, eval_budget=40, target_dim=2),
        seed=9,
        num_restarts=2,
        raw_samples=16,
    )

    result = strategy.optimize(acquisition)

    weights = result.metadata["target_lengthscale_weights"]
    target_bounds = result.metadata["target_bounds"]
    assert weights.shape == (strategy.target_dim,)
    assert target_bounds.shape == (2, strategy.target_dim)
    torch.testing.assert_close(target_bounds, strategy._target_bounds(weights))
    assert torch.all(result.metadata["target_candidates"] >= target_bounds[0])
    assert torch.all(result.metadata["target_candidates"] <= target_bounds[1])
