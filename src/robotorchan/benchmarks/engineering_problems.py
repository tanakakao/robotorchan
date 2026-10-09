"""Engineering-inspired design benchmarks with physical inequality constraints."""

from __future__ import annotations

import torch
from torch import Tensor

from robotorchan.benchmarks.problem import BenchmarkProblem
from robotorchan.benchmarks.registry import BenchmarkProblemRegistry


def _cantilever_mass(X: Tensor) -> Tensor:
    width = X[..., 0]
    height = X[..., 1]
    return (width * height).unsqueeze(-1)


def _cantilever_constraints(X: Tensor) -> Tensor:
    width = X[..., 0]
    height = X[..., 1]
    # Nondimensionalized bending stress and tip deflection constraints.
    stress = 1.0 - 0.2 / (width * height.square())
    deflection = 1.0 - 0.1 / (width * height**3)
    return torch.stack((stress, deflection), dim=-1)


def cantilever_beam() -> BenchmarkProblem:
    """Minimize rectangular beam mass subject to bending and deflection limits.

    Width and height are positive nondimensional design variables.
    Residuals are nonnegative for feasible designs.
    """
    return BenchmarkProblem(
        name="cantilever_beam",
        bounds=torch.tensor([[0.2, 0.2], [2.0, 2.0]], dtype=torch.double),
        objective=_cantilever_mass,
        directions=("minimize",),
        variable_types=("continuous", "continuous"),
        constraints=_cantilever_constraints,
        n_constraints=2,
    )


def _thermal_cost(X: Tensor) -> Tensor:
    insulation = X[..., 0]
    cooling = X[..., 1]
    return (insulation + 2.0 * cooling.square()).unsqueeze(-1)


def _thermal_constraints(X: Tensor) -> Tensor:
    insulation = X[..., 0]
    cooling = X[..., 1]
    temperature_margin = insulation + cooling - 1.0
    cooling_capacity = 0.8 - cooling
    return torch.stack((temperature_margin, cooling_capacity), dim=-1)


def thermal_management() -> BenchmarkProblem:
    """Minimize insulation and cooling cost under thermal/capacity limits."""
    return BenchmarkProblem(
        name="thermal_management",
        bounds=torch.tensor([[0.0, 0.0], [2.0, 1.0]], dtype=torch.double),
        objective=_thermal_cost,
        directions=("minimize",),
        variable_types=("continuous", "continuous"),
        constraints=_thermal_constraints,
        n_constraints=2,
    )


def register_engineering_problems(registry: BenchmarkProblemRegistry) -> None:
    """Register engineering-inspired design problems."""
    for name, factory in (
        ("cantilever_beam", cantilever_beam),
        ("thermal_management", thermal_management),
    ):
        registry.register(name, factory)
