"""Multi-objective regression benchmarks with binary outcome feasibility."""

from __future__ import annotations

import torch
from torch import Tensor

from robotorchan.benchmarks.problem import BenchmarkProblem
from robotorchan.benchmarks.registry import BenchmarkProblemRegistry


def _responses(X: Tensor) -> Tensor:
    strength = 1.0 - (X[..., 0] - 0.8).square() - 0.25 * (X[..., 1] - 0.3).square()
    conductivity = 1.0 - (X[..., 0] - 0.2).square() - 0.25 * (X[..., 1] - 0.7).square()
    return torch.stack((strength, conductivity), dim=-1)


def _pass_margin(X: Tensor) -> Tensor:
    return (X[..., 1] - 0.25 - 0.5 * X[..., 0]).unsqueeze(-1)


def _reliability_margin(X: Tensor) -> Tensor:
    return (0.8 - (X[..., 0] - 0.5).square() - (X[..., 1] - 0.5).square()).unsqueeze(-1)


def _two_margins(X: Tensor) -> Tensor:
    return torch.cat((_pass_margin(X), _reliability_margin(X)), dim=-1)


def multiobjective_pass_labels(X: Tensor) -> Tensor:
    """Canonical binary Pass label; boundary points are feasible."""
    return (_pass_margin(X) >= 0).to(dtype=X.dtype)


def multiobjective_two_labels(X: Tensor) -> Tensor:
    """Canonical binary labels for both learned feasibility constraints."""
    return (_two_margins(X) >= 0).to(dtype=X.dtype)


def strength_conductivity_pass_tradeoff() -> BenchmarkProblem:
    """Maximize Strength and Conductivity under binary Pass feasibility."""
    return BenchmarkProblem(
        name="strength_conductivity_pass_tradeoff",
        bounds=torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double),
        objective=_responses,
        directions=("maximize", "maximize"),
        variable_types=("continuous", "continuous"),
        constraints=_pass_margin,
        n_constraints=1,
        reference_point=torch.tensor([0.0, 0.0], dtype=torch.double),
    )


def strength_conductivity_two_pass() -> BenchmarkProblem:
    """Maximize both responses subject to two binary feasibility outcomes."""
    return BenchmarkProblem(
        name="strength_conductivity_two_pass",
        bounds=torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double),
        objective=_responses,
        directions=("maximize", "maximize"),
        variable_types=("continuous", "continuous"),
        constraints=_two_margins,
        n_constraints=2,
        reference_point=torch.tensor([0.0, 0.0], dtype=torch.double),
    )


def register_multiobjective_heterogeneous_problems(
    registry: BenchmarkProblemRegistry,
) -> None:
    """Register both heterogeneous multi-objective benchmark variants."""
    registry.register("strength_conductivity_pass_tradeoff", strength_conductivity_pass_tradeoff)
    registry.register("strength_conductivity_two_pass", strength_conductivity_two_pass)
