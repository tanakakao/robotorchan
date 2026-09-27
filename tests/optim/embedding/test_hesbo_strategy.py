"""Tests for the HeSBO acquisition search strategy."""

import pytest
import torch
from botorch.acquisition.analytic import PosteriorMean

from robotorchan.models import SingleTaskGP
from robotorchan.optim import HeSBOStrategy


def _problem(input_dim: int = 8):
    torch.manual_seed(17)
    train_X = torch.rand(12, input_dim, dtype=torch.double)
    train_Y = -((train_X[:, :2] - 0.4) ** 2).sum(dim=-1, keepdim=True)
    bounds = torch.stack(
        [torch.zeros(input_dim, dtype=torch.double), torch.ones(input_dim, dtype=torch.double)]
    )
    model = SingleTaskGP(train_X, train_Y)
    return bounds, PosteriorMean(model)


def test_hash_embedding_has_one_signed_assignment_per_input_dimension() -> None:
    bounds, _ = _problem(input_dim=12)
    strategy = HeSBOStrategy(bounds, embedding_dim=4, seed=3)

    assert strategy.embedding.shape == (12, 4)
    assert torch.all((strategy.embedding != 0).sum(dim=-1) == 1)
    assert set(strategy.embedding[strategy.embedding != 0].tolist()) <= {-1.0, 1.0}


def test_seed_reproduces_hash_embedding() -> None:
    bounds, _ = _problem(input_dim=10)

    first = HeSBOStrategy(bounds, embedding_dim=3, seed=11)
    second = HeSBOStrategy(bounds, embedding_dim=3, seed=11)

    torch.testing.assert_close(first.embedding, second.embedding)


def test_project_returns_original_space_candidates() -> None:
    bounds, _ = _problem(input_dim=6)
    strategy = HeSBOStrategy(bounds, embedding_dim=2, seed=5)
    Z = torch.tensor([[0.25, -0.5]], dtype=torch.double)

    X = strategy.project(Z)

    assert X.shape == (1, 6)
    assert torch.all(bounds[0] <= X)
    assert torch.all(bounds[1] >= X)


def test_optimize_returns_public_space_candidate_and_embedding_metadata() -> None:
    bounds, acquisition = _problem(input_dim=6)
    strategy = HeSBOStrategy(
        bounds,
        embedding_dim=2,
        seed=7,
        num_restarts=2,
        raw_samples=16,
    )

    result = strategy.optimize(acquisition)

    assert result.candidates.shape == (1, 6)
    assert result.metadata["embedded_candidates"].shape == (1, 2)
    assert result.metadata["embedding"].shape == (6, 2)
    torch.testing.assert_close(
        strategy.project(result.metadata["embedded_candidates"]),
        result.candidates,
    )


def test_hesbo_validates_configuration_and_q() -> None:
    bounds, acquisition = _problem(input_dim=5)

    with pytest.raises(ValueError, match="embedding_dim"):
        HeSBOStrategy(bounds, embedding_dim=0)
    with pytest.raises(ValueError, match="embedding_dim"):
        HeSBOStrategy(bounds, embedding_dim=6)
    with pytest.raises(ValueError, match="embedded_bound"):
        HeSBOStrategy(bounds, embedding_dim=2, embedded_bound=0.0)

    strategy = HeSBOStrategy(bounds, embedding_dim=2)
    with pytest.raises(ValueError, match="q must be at least 1"):
        strategy.optimize(acquisition, q=0)
