"""Tests for native NSGA-II vector-objective optimization."""

import pytest
import torch
from torch import Tensor

from robotorchan.optim.backends import optimize_vector_nsga2


def _tradeoff_objective(X: Tensor) -> Tensor:
    first = -(X[:, 0] ** 2)
    second = -((X[:, 0] - 1.0) ** 2)
    return torch.stack([first, second], dim=-1)


def test_nsga2_returns_nondominated_pareto_approximation() -> None:
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)

    candidates, values = optimize_vector_nsga2(
        _tradeoff_objective,
        bounds,
        population_size=64,
        generations=30,
        seed=7,
    )

    assert candidates.ndim == 2
    assert values.shape == (candidates.shape[0], 2)
    assert candidates.shape[0] > 10
    assert float(candidates.min()) >= 0.0
    assert float(candidates.max()) <= 1.0
    dominates = (
        (values.unsqueeze(1) >= values.unsqueeze(0)).all(dim=-1)
        & (values.unsqueeze(1) > values.unsqueeze(0)).any(dim=-1)
    )
    assert not torch.any(dominates)


def test_nsga2_is_reproducible_with_local_seed() -> None:
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)

    first = optimize_vector_nsga2(
        _tradeoff_objective, bounds, population_size=32, generations=8, seed=4
    )
    second = optimize_vector_nsga2(
        _tradeoff_objective, bounds, population_size=32, generations=8, seed=4
    )

    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])


def test_nsga2_requires_vector_objective() -> None:
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)

    with pytest.raises(ValueError, match="at least two"):
        optimize_vector_nsga2(
            lambda X: X[:, :1],
            bounds,
            population_size=8,
            generations=1,
        )


def test_nsga2_preserves_dtype_and_device() -> None:
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)

    candidates, values = optimize_vector_nsga2(
        lambda X: torch.stack([X[:, 0], X[:, 1]], dim=-1),
        bounds,
        population_size=16,
        generations=3,
        seed=2,
    )

    assert candidates.dtype == bounds.dtype
    assert values.dtype == bounds.dtype
    assert candidates.device == bounds.device
