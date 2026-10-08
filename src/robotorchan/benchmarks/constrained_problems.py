"""Deterministic constrained optimization problems for benchmark evaluation."""

from __future__ import annotations

import torch
from torch import Tensor

from robotorchan.benchmarks.problem import BenchmarkProblem
from robotorchan.benchmarks.registry import BenchmarkProblemRegistry


def _constrained_quadratic_objective(X: Tensor) -> Tensor:
    return -((X[..., 0] - 0.8).square() + (X[..., 1] - 0.8).square()).unsqueeze(-1)


def _constrained_quadratic_constraints(X: Tensor) -> Tensor:
    return (1.0 - X[..., 0] - X[..., 1]).unsqueeze(-1)


def constrained_quadratic() -> BenchmarkProblem:
    """Maximize a quadratic on [0, 1]^2 with x0 + x1 <= 1.

    The unconstrained maximizer (0.8, 0.8) is infeasible.
    The constrained maximizer is (0.5, 0.5) with value -0.18.
    """
    return BenchmarkProblem(
        name="constrained_quadratic",
        bounds=torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double),
        objective=_constrained_quadratic_objective,
        directions=("maximize",),
        variable_types=("continuous", "continuous"),
        constraints=_constrained_quadratic_constraints,
        n_constraints=1,
        optimal_value=torch.tensor([-0.18], dtype=torch.double),
    )


def _annulus_objective(X: Tensor) -> Tensor:
    return X.square().sum(dim=-1, keepdim=True)


def _annulus_constraints(X: Tensor) -> Tensor:
    radius_squared = X.square().sum(dim=-1)
    return torch.stack((radius_squared - 0.25, 1.0 - radius_squared), dim=-1)


def constrained_annulus() -> BenchmarkProblem:
    """Minimize squared radius on the annulus 0.5 <= ||x|| <= 1."""
    return BenchmarkProblem(
        name="constrained_annulus",
        bounds=torch.tensor([[-1.0, -1.0], [1.0, 1.0]], dtype=torch.double),
        objective=_annulus_objective,
        directions=("minimize",),
        variable_types=("continuous", "continuous"),
        constraints=_annulus_constraints,
        n_constraints=2,
        optimal_value=torch.tensor([0.25], dtype=torch.double),
    )


def register_constrained_problems(registry: BenchmarkProblemRegistry) -> None:
    """Register Phase 7 constrained problems in the supplied registry."""
    for name, factory in (
        ("constrained_annulus", constrained_annulus),
        ("constrained_quadratic", constrained_quadratic),
    ):
        registry.register(name, factory)
