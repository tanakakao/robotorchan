"""High-dimensional optimization benchmarks with controlled effective dimension."""

from __future__ import annotations

import torch
from torch import Tensor

from robotorchan.benchmarks.problem import BenchmarkProblem
from robotorchan.benchmarks.registry import BenchmarkProblemRegistry


def _sparse_sphere(X: Tensor) -> Tensor:
    return (X[..., :4] - 0.3).square().sum(dim=-1, keepdim=True)


def sparse_sphere_50() -> BenchmarkProblem:
    """Minimize a 50D sphere with four active coordinates and 46 inactive ones."""
    return BenchmarkProblem(
        name="sparse_sphere_50",
        bounds=torch.tensor([[0.0] * 50, [1.0] * 50], dtype=torch.double),
        objective=_sparse_sphere,
        directions=("minimize",),
        variable_types=("continuous",) * 50,
        optimal_value=torch.tensor([0.0], dtype=torch.double),
    )


def _rotated_subspace(X: Tensor) -> Tensor:
    # Orthogonal projections couple every input coordinate to two active directions.
    dimension = X.shape[-1]
    first = X.new_ones(dimension) / dimension**0.5
    second = X.new_tensor([1.0] * (dimension // 2) + [-1.0] * (dimension // 2))
    second = second / dimension**0.5
    centered = X - 0.5
    projected = torch.stack((centered @ first, centered @ second), dim=-1)
    return projected.square().sum(dim=-1, keepdim=True)


def rotated_subspace_40() -> BenchmarkProblem:
    """Minimize a 40D problem with two dense orthogonal active directions."""
    return BenchmarkProblem(
        name="rotated_subspace_40",
        bounds=torch.tensor([[0.0] * 40, [1.0] * 40], dtype=torch.double),
        objective=_rotated_subspace,
        directions=("minimize",),
        variable_types=("continuous",) * 40,
        optimal_value=torch.tensor([0.0], dtype=torch.double),
    )


def _interaction_chain(X: Tensor) -> Tensor:
    active = X[..., :10]
    centered = active - 0.4
    pairwise = (centered[..., 1:] - centered[..., :-1]).square()
    return (centered.square().sum(dim=-1) + pairwise.sum(dim=-1)).unsqueeze(-1)


def interaction_chain_30() -> BenchmarkProblem:
    """Minimize a 30D landscape with ten active, interacting coordinates."""
    return BenchmarkProblem(
        name="interaction_chain_30",
        bounds=torch.tensor([[0.0] * 30, [1.0] * 30], dtype=torch.double),
        objective=_interaction_chain,
        directions=("minimize",),
        variable_types=("continuous",) * 30,
        optimal_value=torch.tensor([0.0], dtype=torch.double),
    )


def register_high_dimensional_problems(registry: BenchmarkProblemRegistry) -> None:
    """Register sparse, rotated and interacting high-dimensional problems."""
    for name, factory in (
        ("interaction_chain_30", interaction_chain_30),
        ("rotated_subspace_40", rotated_subspace_40),
        ("sparse_sphere_50", sparse_sphere_50),
    ):
        registry.register(name, factory)
