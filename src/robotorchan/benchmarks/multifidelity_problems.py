"""Cost-aware multi-fidelity benchmark problems.

The last input coordinate is the continuous fidelity parameter s in [0, 1].
At s=1 the response equals the high-fidelity target. Lower fidelities
introduce deterministic approximation bias, with cheaper evaluations.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from robotorchan.benchmarks.problem import BenchmarkProblem
from robotorchan.benchmarks.registry import BenchmarkProblemRegistry


def _quadratic_response(X: Tensor) -> Tensor:
    x = X[..., 0]
    fidelity = X[..., 1]
    target = (x - 0.7).square()
    bias = (1.0 - fidelity) * (0.25 + 0.2 * (x - 0.2).square())
    return (target + bias).unsqueeze(-1)


def _oscillatory_response(X: Tensor) -> Tensor:
    x = X[..., 0]
    fidelity = X[..., 1]
    target = (x - 0.65).square() + 0.05 * torch.sin(6.0 * math.pi * x).square()
    bias = (1.0 - fidelity) * (0.1 + 0.3 * (x - 0.1).square())
    return (target + bias).unsqueeze(-1)


def _evaluation_cost(X: Tensor) -> Tensor:
    fidelity = X[..., -1:]
    return 0.1 + 0.9 * fidelity.square()


def multifidelity_quadratic() -> BenchmarkProblem:
    """Minimize a fidelity-biased quadratic, with fidelity-dependent cost."""
    return BenchmarkProblem(
        name="multifidelity_quadratic",
        bounds=torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double),
        objective=_quadratic_response,
        directions=("minimize",),
        variable_types=("continuous", "continuous"),
        cost=_evaluation_cost,
        optimal_value=torch.tensor([0.0], dtype=torch.double),
    )


def multifidelity_oscillatory() -> BenchmarkProblem:
    """Minimize a nonlinear fidelity-biased response with variable cost."""
    return BenchmarkProblem(
        name="multifidelity_oscillatory",
        bounds=torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double),
        objective=_oscillatory_response,
        directions=("minimize",),
        variable_types=("continuous", "continuous"),
        cost=_evaluation_cost,
    )


def register_multifidelity_problems(registry: BenchmarkProblemRegistry) -> None:
    """Register multi-fidelity benchmark factories."""
    for name, factory in (
        ("multifidelity_oscillatory", multifidelity_oscillatory),
        ("multifidelity_quadratic", multifidelity_quadratic),
    ):
        registry.register(name, factory)
