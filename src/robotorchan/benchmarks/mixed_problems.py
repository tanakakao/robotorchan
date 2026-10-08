"""Mixed continuous, integer, and categorical optimization benchmarks."""

from __future__ import annotations

import torch
from torch import Tensor

from robotorchan.benchmarks.problem import BenchmarkProblem
from robotorchan.benchmarks.registry import BenchmarkProblemRegistry


def _mixed_quadratic(X: Tensor) -> Tensor:
    """A minimization landscape with non-ordinal category-specific optima."""
    continuous = X[..., 0]
    integer = X[..., 1]
    category = X[..., 2].long()
    category_center = X.new_tensor([0.2, 0.8, 0.45])[category]
    category_offset = X.new_tensor([0.4, 0.0, 0.2])[category]
    value = (continuous - category_center).square()
    value = value + 0.25 * (integer - 2.0).square() + category_offset
    return value.unsqueeze(-1)


def mixed_quadratic() -> BenchmarkProblem:
    """Minimize over x in [0,1], integer n in {0,...,4}, category c in {0,1,2}.

    Category codes are labels rather than ordinal distances. The optimum is
    (0.8, 2, 1), where the objective is zero.
    """
    return BenchmarkProblem(
        name="mixed_quadratic",
        bounds=torch.tensor([[0.0, 0.0, 0.0], [1.0, 4.0, 2.0]], dtype=torch.double),
        objective=_mixed_quadratic,
        directions=("minimize",),
        variable_types=("continuous", "integer", "categorical"),
        optimal_value=torch.tensor([0.0], dtype=torch.double),
    )


def _categorical_switch(X: Tensor) -> Tensor:
    """Three category-specific response curves with a known best category."""
    x = X[..., 0]
    category = X[..., 1].long()
    centers = X.new_tensor([0.1, 0.7, 0.4])[category]
    peaks = X.new_tensor([0.6, 1.0, 0.8])[category]
    return (peaks - (x - centers).square()).unsqueeze(-1)


def categorical_switch() -> BenchmarkProblem:
    """Maximize a category-conditioned response on a continuous input."""
    return BenchmarkProblem(
        name="categorical_switch",
        bounds=torch.tensor([[0.0, 0.0], [1.0, 2.0]], dtype=torch.double),
        objective=_categorical_switch,
        directions=("maximize",),
        variable_types=("continuous", "categorical"),
        optimal_value=torch.tensor([1.0], dtype=torch.double),
    )


def register_mixed_problems(registry: BenchmarkProblemRegistry) -> None:
    """Register Phase 9 mixed-variable benchmarks explicitly."""
    for name, factory in (
        ("categorical_switch", categorical_switch),
        ("mixed_quadratic", mixed_quadratic),
    ):
        registry.register(name, factory)
