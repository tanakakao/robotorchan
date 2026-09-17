"""Tests for the REMBO acquisition search strategy."""

import pytest
import torch
from botorch.acquisition.analytic import PosteriorMean

from robotorchan.models import SingleTaskGP
from robotorchan.optim import REMBOStrategy


def _problem(input_dim: int = 6):
    torch.manual_seed(11)
    train_X = torch.rand(12, input_dim, dtype=torch.double)
    train_Y = -((train_X[:, :2] - 0.7) ** 2).sum(dim=-1, keepdim=True)
    bounds = torch.stack(
        [
            torch.zeros(input_dim, dtype=torch.double),
            torch.ones(input_dim, dtype=torch.double),
        ]
    )
    return train_X, train_Y, bounds


def test_embedding_is_reproducible_and_normalized() -> None:
    _, _, bounds = _problem()
    first = REMBOStrategy(bounds, embedding_dim=2, seed=5)
    second = REMBOStrategy(bounds, embedding_dim=2, seed=5)

    torch.testing.assert_close(first.embedding, second.embedding)
    torch.testing.assert_close(
        first.embedding.norm(dim=0),
        torch.ones(2, dtype=torch.double),
    )
    assert first.embedded_bounds.shape == (2, 2)


def test_projection_returns_original_box_points() -> None:
    _, _, bounds = _problem()
    strategy = REMBOStrategy(bounds, embedding_dim=2, embedded_bound=3.0, seed=3)
    Z = torch.tensor([[3.0, -3.0], [-3.0, 3.0]], dtype=torch.double)

    X = strategy.project(Z)

    assert X.shape == (2, 6)
    assert torch.all(bounds[0] <= X)
    assert torch.all(bounds[1] >= X)


def test_optimize_preserves_original_acquisition_contract() -> None:
    train_X, train_Y, bounds = _problem()
    model = SingleTaskGP(train_X, train_Y)
    acquisition = PosteriorMean(model)
    strategy = REMBOStrategy(
        bounds,
        embedding_dim=2,
        seed=7,
        num_restarts=2,
        raw_samples=16,
    )

    result = strategy.optimize(acquisition, q=1)

    assert result.candidates.shape == (1, 6)
    assert torch.all(bounds[0] <= result.candidates)
    assert torch.all(bounds[1] >= result.candidates)
    assert result.acquisition_value is not None
    torch.testing.assert_close(result.acquisition_value, acquisition(result.candidates))
    assert result.metadata["embedded_candidates"].shape == (1, 2)
    assert result.metadata["clipping_distance"].shape == (1,)
    assert result.metadata["embedding"].shape == (6, 2)


def test_validates_arguments() -> None:
    _, _, bounds = _problem()
    with pytest.raises(ValueError, match="embedding_dim"):
        REMBOStrategy(bounds, embedding_dim=0)
    with pytest.raises(ValueError, match="embedding_dim"):
        REMBOStrategy(bounds, embedding_dim=7)
    with pytest.raises(ValueError, match="embedded_bound"):
        REMBOStrategy(bounds, embedding_dim=2, embedded_bound=0.0)
    with pytest.raises(ValueError, match="num_restarts"):
        REMBOStrategy(bounds, embedding_dim=2, num_restarts=0)
    with pytest.raises(ValueError, match="raw_samples"):
        REMBOStrategy(bounds, embedding_dim=2, raw_samples=0)

    strategy = REMBOStrategy(bounds, embedding_dim=2)
    with pytest.raises(ValueError, match="Z last dimension"):
        strategy.project(torch.zeros(1, 3, dtype=torch.double))
    with pytest.raises(ValueError, match="q must be at least 1"):
        strategy.optimize(PosteriorMean(SingleTaskGP(*_problem()[:2])), q=0)
