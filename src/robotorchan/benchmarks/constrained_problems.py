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


def _disconnected_objective(X: Tensor) -> Tensor:
    return -((X[..., 0] - 0.75).square() + X[..., 1].square()).unsqueeze(-1)


def _disconnected_constraints(X: Tensor) -> Tensor:
    radius_squared = (X[..., 0].abs() - 0.75).square() + X[..., 1].square()
    return (0.25**2 - radius_squared).unsqueeze(-1)


def constrained_disconnected() -> BenchmarkProblem:
    """Maximize a quadratic over two disconnected feasible disks.

    The disks have radius 0.25 and centers (-0.75, 0) and (0.75, 0).
    The global constrained maximizer is (0.75, 0).
    """
    return BenchmarkProblem(
        name="constrained_disconnected",
        bounds=torch.tensor([[-1.0, -1.0], [1.0, 1.0]], dtype=torch.double),
        objective=_disconnected_objective,
        directions=("maximize",),
        variable_types=("continuous", "continuous"),
        constraints=_disconnected_constraints,
        n_constraints=1,
        optimal_value=torch.tensor([0.0], dtype=torch.double),
    )


def _narrow_band_objective(X: Tensor) -> Tensor:
    return -((X[..., 0] - 0.8).square() + (X[..., 1] - 0.8).square()).unsqueeze(-1)


def _narrow_band_constraints(X: Tensor) -> Tensor:
    total = X[..., 0] + X[..., 1]
    return torch.stack((total - 0.95, 1.05 - total), dim=-1)


def constrained_narrow_band() -> BenchmarkProblem:
    """Maximize a quadratic within 0.95 <= x0+x1 <= 1.05.

    The constrained maximizer is (0.525, 0.525), value -0.15125.
    """
    return BenchmarkProblem(
        name="constrained_narrow_band",
        bounds=torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double),
        objective=_narrow_band_objective,
        directions=("maximize",),
        variable_types=("continuous", "continuous"),
        constraints=_narrow_band_constraints,
        n_constraints=2,
        optimal_value=torch.tensor([-0.15125], dtype=torch.double),
    )


def register_constrained_problems(registry: BenchmarkProblemRegistry) -> None:
    """Register constrained problems in the supplied registry."""
    for name, factory in (
        ("constrained_annulus", constrained_annulus),
        ("constrained_disconnected", constrained_disconnected),
        ("constrained_narrow_band", constrained_narrow_band),
        ("constrained_quadratic", constrained_quadratic),
    ):
        registry.register(name, factory)
