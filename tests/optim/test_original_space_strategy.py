"""Tests for direct acquisition optimization in original input space."""

import pytest
import torch
from botorch.acquisition.analytic import PosteriorMean

from robotorchan.models.high_dimensional.reduced import PCAGP
from robotorchan.models.standard.single_task import SingleTaskGP
from robotorchan.optim import CandidateConstraints, OriginalSpaceStrategy


def _training_data() -> tuple[torch.Tensor, torch.Tensor]:
    train_X = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.5, 0.25],
            [0.25, 0.75],
        ],
        dtype=torch.double,
    )
    train_Y = (train_X[:, :1] - 0.5 * train_X[:, 1:2]).sin()
    return train_X, train_Y


def test_original_space_strategy_validates_optimizer_configuration() -> None:
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]])

    with pytest.raises(ValueError, match="num_restarts"):
        OriginalSpaceStrategy(bounds, num_restarts=0)
    with pytest.raises(ValueError, match="raw_samples"):
        OriginalSpaceStrategy(bounds, raw_samples=0)


def test_original_space_strategy_rejects_invalid_q() -> None:
    train_X, train_Y = _training_data()
    model = SingleTaskGP(train_X, train_Y)
    acquisition = PosteriorMean(model)
    strategy = OriginalSpaceStrategy(
        torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double),
        num_restarts=2,
        raw_samples=8,
    )

    with pytest.raises(ValueError, match="q"):
        strategy.optimize(acquisition, q=0)


def test_original_space_strategy_optimizes_single_task_gp() -> None:
    train_X, train_Y = _training_data()
    model = SingleTaskGP(train_X, train_Y)
    acquisition = PosteriorMean(model)
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
    strategy = OriginalSpaceStrategy(bounds, num_restarts=2, raw_samples=16)

    result = strategy.optimize(acquisition)

    assert result.candidates.shape == torch.Size([1, 2])
    assert torch.all(result.candidates >= bounds[0])
    assert torch.all(result.candidates <= bounds[1])
    assert result.acquisition_value is not None


def test_original_space_strategy_preserves_reduced_gp_public_space() -> None:
    train_X, train_Y = _training_data()
    model = PCAGP(train_X, train_Y, n_components=1)
    acquisition = PosteriorMean(model)
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
    strategy = OriginalSpaceStrategy(bounds, num_restarts=2, raw_samples=16)

    result = strategy.optimize(acquisition)

    assert result.candidates.shape == torch.Size([1, 2])
    assert model.raw_train_X.shape[-1] == 2
    assert torch.all(result.candidates >= bounds[0])
    assert torch.all(result.candidates <= bounds[1])


def test_original_space_strategy_satisfies_linear_inequality_constraint() -> None:
    train_X, train_Y = _training_data()
    model = SingleTaskGP(train_X, train_Y)
    acquisition = PosteriorMean(model)
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
    constraints = CandidateConstraints(
        inequality_constraints=(
            (
                torch.tensor([0, 1]),
                torch.tensor([-1.0, -1.0], dtype=torch.double),
                -0.5,
            ),
        ),
    )
    strategy = OriginalSpaceStrategy(
        bounds,
        num_restarts=4,
        raw_samples=64,
        constraints=constraints,
    )

    result = strategy.optimize(acquisition)

    assert result.candidates.shape == torch.Size([1, 2])
    assert result.candidates.sum() <= 0.5 + 1e-6


def test_original_space_strategy_satisfies_linear_equality_constraint() -> None:
    train_X, train_Y = _training_data()
    model = SingleTaskGP(train_X, train_Y)
    acquisition = PosteriorMean(model)
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
    constraints = CandidateConstraints(
        equality_constraints=(
            (
                torch.tensor([0, 1]),
                torch.tensor([1.0, 1.0], dtype=torch.double),
                1.0,
            ),
        ),
    )
    strategy = OriginalSpaceStrategy(
        bounds,
        num_restarts=4,
        raw_samples=64,
        constraints=constraints,
    )

    result = strategy.optimize(acquisition)

    assert result.candidates.shape == torch.Size([1, 2])
    assert torch.allclose(
        result.candidates.sum(),
        torch.tensor(1.0, dtype=torch.double),
        atol=1e-6,
    )
