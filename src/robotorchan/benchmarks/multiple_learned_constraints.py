"""Benchmarks with multiple learned binary and continuous outcome constraints."""

from __future__ import annotations

import torch
from torch import Tensor

from robotorchan.benchmarks.problem import BenchmarkProblem
from robotorchan.benchmarks.registry import BenchmarkProblemRegistry


def _yield(X: Tensor) -> Tensor:
    return (1.0 - (X[..., 0] - 0.8).square() - (X[..., 1] - 0.2).square()).unsqueeze(-1)


def _pass_margin(X: Tensor) -> Tensor:
    return (X[..., 1] - 0.2 - 0.4 * X[..., 0]).unsqueeze(-1)


def _reliability_margin(X: Tensor) -> Tensor:
    return (0.4 - X[..., 0]).unsqueeze(-1)


def _temperature_margin(X: Tensor) -> Tensor:
    return (0.8 - X[..., 1]).unsqueeze(-1)


def binary_constraint_labels(X: Tensor) -> Tensor:
    """Return two canonical 0/1 labels for the binary constraints."""
    margins = torch.cat((_pass_margin(X), _reliability_margin(X)), dim=-1)
    return (margins >= 0).to(dtype=X.dtype)


def _binary_constraints(X: Tensor) -> Tensor:
    return torch.cat((_pass_margin(X), _reliability_margin(X)), dim=-1)


def _mixed_constraints(X: Tensor) -> Tensor:
    return torch.cat((_pass_margin(X), _temperature_margin(X)), dim=-1)


def yield_two_binary_constraints() -> BenchmarkProblem:
    """Maximize Yield subject to two binary feasibility conditions."""
    return BenchmarkProblem(
        name="yield_two_binary_constraints",
        bounds=torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double),
        objective=_yield,
        directions=("maximize",),
        variable_types=("continuous", "continuous"),
        constraints=_binary_constraints,
        n_constraints=2,
        optimal_value=torch.tensor([0.8144], dtype=torch.double),
    )


def yield_binary_continuous_constraints() -> BenchmarkProblem:
    """Maximize Yield with binary Pass and continuous temperature limits."""
    return BenchmarkProblem(
        name="yield_binary_continuous_constraints",
        bounds=torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double),
        objective=_yield,
        directions=("maximize",),
        variable_types=("continuous", "continuous"),
        constraints=_mixed_constraints,
        n_constraints=2,
        optimal_value=torch.tensor([1.0 - 0.64 * 0.16 / 1.16], dtype=torch.double),
    )


def register_multiple_learned_constraint_problems(
    registry: BenchmarkProblemRegistry,
) -> None:
    """Register multiple learned-outcome-constraint benchmark variants."""
    registry.register("yield_binary_continuous_constraints", yield_binary_continuous_constraints)
    registry.register("yield_two_binary_constraints", yield_two_binary_constraints)
