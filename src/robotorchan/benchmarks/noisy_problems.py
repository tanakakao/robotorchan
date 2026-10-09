"""Noisy benchmark problems with independent observation and truth evaluators."""

from __future__ import annotations

import torch
from torch import Tensor

from robotorchan.benchmarks.problem import BenchmarkProblem
from robotorchan.benchmarks.registry import BenchmarkProblemRegistry


def _quadratic(X: Tensor) -> Tensor:
    return -((X - 0.35).square().sum(dim=-1, keepdim=True))


def _heteroscedastic_std(X: Tensor) -> Tensor:
    return 0.02 + 0.18 * X[..., :1]


def _noisy_quadratic(name: str, *, heteroscedastic: bool) -> BenchmarkProblem:
    generator = torch.Generator(device="cpu").manual_seed(0)

    def observe(X: Tensor) -> Tensor:
        truth = _quadratic(X)
        std = _heteroscedastic_std(X) if heteroscedastic else torch.full_like(truth, 0.1)
        noise = torch.randn(truth.shape, generator=generator, dtype=truth.dtype)
        return truth + std * noise.to(device=truth.device)

    return BenchmarkProblem(
        name=name,
        bounds=torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double),
        objective=_quadratic,
        observe=observe,
        directions=("maximize",),
        variable_types=("continuous", "continuous"),
        optimal_value=torch.tensor([0.0], dtype=torch.double),
    )


def noisy_quadratic() -> BenchmarkProblem:
    """Quadratic maximization with independent Gaussian noise (std 0.1)."""
    return _noisy_quadratic("noisy_quadratic", heteroscedastic=False)


def heteroscedastic_quadratic() -> BenchmarkProblem:
    """Quadratic maximization with input-dependent Gaussian noise."""
    return _noisy_quadratic("heteroscedastic_quadratic", heteroscedastic=True)


def register_noisy_problems(registry: BenchmarkProblemRegistry) -> None:
    """Register noisy benchmark factories."""
    for name, factory in (
        ("heteroscedastic_quadratic", heteroscedastic_quadratic),
        ("noisy_quadratic", noisy_quadratic),
    ):
        registry.register(name, factory)
