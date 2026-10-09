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


def _mixed_process_yield(X: Tensor) -> Tensor:
    temperature = X[..., 0]
    cycles = X[..., 1]
    recipe = X[..., 2].long()
    centers = X.new_tensor([0.25, 0.7, 0.5])[recipe]
    penalties = X.new_tensor([0.1, 0.0, 0.15])[recipe]
    response = 1.0 - (temperature - centers).square()
    response = response - 0.04 * (cycles - 3.0).square() - penalties
    return response.unsqueeze(-1)


def _mixed_process_constraints(X: Tensor) -> Tensor:
    temperature = X[..., 0]
    cycles = X[..., 1]
    recipe = X[..., 2].long()
    upper_limits = X.new_tensor([0.55, 0.85, 0.65])[recipe]
    return torch.stack((upper_limits - temperature, 4.0 - cycles), dim=-1)


def mixed_process_yield() -> BenchmarkProblem:
    """Maximize process yield with integer cycles and nominal recipe choices.

    Recipe labels are non-ordinal. The feasible optimum is (0.7, 3, 1)
    with value 1.0.
    """
    return BenchmarkProblem(
        name="mixed_process_yield",
        bounds=torch.tensor([[0.0, 1.0, 0.0], [1.0, 5.0, 2.0]], dtype=torch.double),
        objective=_mixed_process_yield,
        directions=("maximize",),
        variable_types=("continuous", "integer", "categorical"),
        constraints=_mixed_process_constraints,
        n_constraints=2,
        optimal_value=torch.tensor([1.0], dtype=torch.double),
    )


def _mixed_category_interaction(X: Tensor) -> Tensor:
    x = X[..., 0]
    first = X[..., 1].long()
    second = X[..., 2].long()
    centers = X.new_tensor([[0.1, 0.3], [0.6, 0.85], [0.4, 0.7]])
    offsets = X.new_tensor([[0.3, 0.2], [0.1, 0.0], [0.25, 0.15]])
    center = centers[first, second]
    offset = offsets[first, second]
    return ((x - center).square() + offset).unsqueeze(-1)


def mixed_category_interaction() -> BenchmarkProblem:
    """Minimize a response with interacting nominal categorical choices.

    The global optimum is (0.85, 1, 1) with value zero.
    """
    return BenchmarkProblem(
        name="mixed_category_interaction",
        bounds=torch.tensor([[0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=torch.double),
        objective=_mixed_category_interaction,
        directions=("minimize",),
        variable_types=("continuous", "categorical", "categorical"),
        optimal_value=torch.tensor([0.0], dtype=torch.double),
    )


def register_mixed_problems(registry: BenchmarkProblemRegistry) -> None:
    """Register mixed-variable benchmarks explicitly."""
    for name, factory in (
        ("categorical_switch", categorical_switch),
        ("mixed_category_interaction", mixed_category_interaction),
        ("mixed_process_yield", mixed_process_yield),
        ("mixed_quadratic", mixed_quadratic),
    ):
        registry.register(name, factory)
