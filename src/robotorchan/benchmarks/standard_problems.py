"""Standard analytic benchmark problems with documented native minimization conventions."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from robotorchan.benchmarks.problem import BenchmarkProblem
from robotorchan.benchmarks.registry import BenchmarkProblemRegistry

_HARTMANN_ALPHA = (1.0, 1.2, 3.0, 3.2)
_HARTMANN_A = (
    (10.0, 3.0, 17.0, 3.5, 1.7, 8.0),
    (0.05, 10.0, 17.0, 0.1, 8.0, 14.0),
    (3.0, 3.5, 1.7, 10.0, 17.0, 8.0),
    (17.0, 8.0, 0.05, 10.0, 0.1, 14.0),
)
_HARTMANN_P = (
    (0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886),
    (0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991),
    (0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650),
    (0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381),
)


def _branin(X: Tensor) -> Tensor:
    x1, x2 = X[..., 0], X[..., 1]
    b = 5.1 / (4.0 * math.pi**2)
    c = 5.0 / math.pi
    t = 1.0 / (8.0 * math.pi)
    value = (x2 - b * x1.square() + c * x1 - 6.0).square()
    value = value + 10.0 * (1.0 - t) * torch.cos(x1) + 10.0
    return value.unsqueeze(-1)


def _hartmann6(X: Tensor) -> Tensor:
    alpha = X.new_tensor(_HARTMANN_ALPHA)
    A = X.new_tensor(_HARTMANN_A)
    P = X.new_tensor(_HARTMANN_P) * 1e-4
    inner = (A * (X.unsqueeze(-2) - P).square()).sum(dim=-1)
    return -(alpha * torch.exp(-inner)).sum(dim=-1, keepdim=True)


def _sphere(X: Tensor) -> Tensor:
    return X.square().sum(dim=-1, keepdim=True)


def branin() -> BenchmarkProblem:
    """Two-dimensional Branin minimization with three global minima."""
    return BenchmarkProblem(
        name="branin",
        bounds=torch.tensor([[-5.0, 0.0], [10.0, 15.0]], dtype=torch.double),
        objective=_branin,
        directions=("minimize",),
        variable_types=("continuous", "continuous"),
        optimal_value=torch.tensor([0.39788735772973816], dtype=torch.double),
    )


def hartmann6() -> BenchmarkProblem:
    """Six-dimensional Hartmann minimization on the unit hypercube."""
    return BenchmarkProblem(
        name="hartmann6",
        bounds=torch.tensor([[0.0] * 6, [1.0] * 6], dtype=torch.double),
        objective=_hartmann6,
        directions=("minimize",),
        variable_types=("continuous",) * 6,
        optimal_value=torch.tensor([-3.322368011415515], dtype=torch.double),
    )


def sphere3() -> BenchmarkProblem:
    """Three-dimensional sphere minimization with optimum at the origin."""
    return BenchmarkProblem(
        name="sphere3",
        bounds=torch.tensor([[-5.0] * 3, [5.0] * 3], dtype=torch.double),
        objective=_sphere,
        directions=("minimize",),
        variable_types=("continuous",) * 3,
        optimal_value=torch.tensor([0.0], dtype=torch.double),
    )


def register_standard_problems(registry: BenchmarkProblemRegistry) -> None:
    """Register all Phase 6 problems in an explicit registry."""
    for name, factory in (
        ("branin", branin),
        ("hartmann6", hartmann6),
        ("sphere3", sphere3),
    ):
        registry.register(name, factory)
