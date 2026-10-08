"""Deterministic multiobjective benchmarks with analytic Pareto fronts."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from robotorchan.benchmarks.problem import BenchmarkProblem
from robotorchan.benchmarks.registry import BenchmarkProblemRegistry


def _biobjective_linear(X: Tensor) -> Tensor:
    return torch.cat((X, 1.0 - X), dim=-1)


def biobjective_linear() -> BenchmarkProblem:
    """Maximize (x, 1-x) on [0,1], with a complete linear Pareto front."""
    front_x = torch.linspace(0.0, 1.0, 101, dtype=torch.double).unsqueeze(-1)
    return BenchmarkProblem(
        name="biobjective_linear",
        bounds=torch.tensor([[0.0], [1.0]], dtype=torch.double),
        objective=_biobjective_linear,
        directions=("maximize", "maximize"),
        variable_types=("continuous",),
        reference_point=torch.tensor([-0.1, -0.1], dtype=torch.double),
        reference_front=_biobjective_linear(front_x),
    )


def _zdt1(X: Tensor) -> Tensor:
    f1 = X[..., :1]
    g = 1.0 + 9.0 * X[..., 1:].mean(dim=-1, keepdim=True)
    f2 = g * (1.0 - torch.sqrt(f1 / g))
    return torch.cat((f1, f2), dim=-1)


def zdt1() -> BenchmarkProblem:
    """Minimize ZDT1 with six decision variables and a convex Pareto front."""
    f1 = torch.linspace(0.0, 1.0, 101, dtype=torch.double).unsqueeze(-1)
    front = torch.cat((f1, 1.0 - torch.sqrt(f1)), dim=-1)
    return BenchmarkProblem(
        name="zdt1",
        bounds=torch.tensor([[0.0] * 6, [1.0] * 6], dtype=torch.double),
        objective=_zdt1,
        directions=("minimize", "minimize"),
        variable_types=("continuous",) * 6,
        reference_point=torch.tensor([1.1, 1.1], dtype=torch.double),
        reference_front=front,
    )


def _dtlz2(X: Tensor) -> Tensor:
    g = (X[..., 2:] - 0.5).square().sum(dim=-1, keepdim=True)
    angle0 = X[..., :1] * (math.pi / 2.0)
    angle1 = X[..., 1:2] * (math.pi / 2.0)
    f1 = (1.0 + g) * torch.cos(angle0) * torch.cos(angle1)
    f2 = (1.0 + g) * torch.cos(angle0) * torch.sin(angle1)
    f3 = (1.0 + g) * torch.sin(angle0)
    return torch.cat((f1, f2, f3), dim=-1)


def dtlz2() -> BenchmarkProblem:
    """Minimize three-objective DTLZ2 with seven decision variables."""
    angles = torch.linspace(0.0, math.pi / 2.0, 21, dtype=torch.double)
    angle0, angle1 = torch.meshgrid(angles, angles, indexing="ij")
    front = torch.stack(
        (
            torch.cos(angle0) * torch.cos(angle1),
            torch.cos(angle0) * torch.sin(angle1),
            torch.sin(angle0),
        ),
        dim=-1,
    ).reshape(-1, 3)
    return BenchmarkProblem(
        name="dtlz2",
        bounds=torch.tensor([[0.0] * 7, [1.0] * 7], dtype=torch.double),
        objective=_dtlz2,
        directions=("minimize",) * 3,
        variable_types=("continuous",) * 7,
        reference_point=torch.tensor([1.1] * 3, dtype=torch.double),
        reference_front=front,
    )


def register_multiobjective_problems(registry: BenchmarkProblemRegistry) -> None:
    """Register Phase 8 problems explicitly in a benchmark registry."""
    for name, factory in (
        ("biobjective_linear", biobjective_linear),
        ("dtlz2", dtlz2),
        ("zdt1", zdt1),
    ):
        registry.register(name, factory)
