"""Probability-valued objectives with binary observations for BO benchmarks."""

from __future__ import annotations

import torch
from torch import Tensor

from robotorchan.benchmarks.problem import BenchmarkProblem
from robotorchan.benchmarks.registry import BenchmarkProblemRegistry


def _pass_probability(X: Tensor) -> Tensor:
    """Return analytic Bernoulli success probability in [0, 1]."""
    distance = (X[..., 0] - 0.7).square() + (X[..., 1] - 0.3).square()
    return (0.1 + 0.8 * torch.exp(-8.0 * distance)).unsqueeze(-1)


def _tradeoff_objectives(X: Tensor) -> Tensor:
    yield_value = 1.0 - (X[..., 0] - 0.2).square() - (X[..., 1] - 0.7).square()
    return torch.cat((yield_value.unsqueeze(-1), _pass_probability(X)), dim=-1)


def pass_probability_objective() -> BenchmarkProblem:
    """Maximize an analytic Pass probability; optimum is 0.9 at (0.7, 0.3)."""
    return BenchmarkProblem(
        name="pass_probability_objective",
        bounds=torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double),
        objective=_pass_probability,
        directions=("maximize",),
        variable_types=("continuous", "continuous"),
        optimal_value=torch.tensor([0.9], dtype=torch.double),
    )


def yield_probability_tradeoff() -> BenchmarkProblem:
    """Maximize deterministic yield and analytic Pass probability together."""
    return BenchmarkProblem(
        name="yield_probability_tradeoff",
        bounds=torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double),
        objective=_tradeoff_objectives,
        directions=("maximize", "maximize"),
        variable_types=("continuous", "continuous"),
        reference_point=torch.tensor([0.0, 0.0], dtype=torch.double),
    )


def sample_pass_labels(X: Tensor, generator: torch.Generator) -> Tensor:
    """Draw Bernoulli 0/1 labels separately from probability-valued truth.

    The supplied generator controls reproducibility. It must use the
    same device as X for torch.rand.
    """
    probability = _pass_probability(X)
    draws = torch.rand(
        probability.shape,
        dtype=probability.dtype,
        device=probability.device,
        generator=generator,
    )
    return (draws < probability).to(dtype=probability.dtype)


def register_probability_objective_problems(registry: BenchmarkProblemRegistry) -> None:
    """Register single- and multi-objective probability benchmarks."""
    registry.register("pass_probability_objective", pass_probability_objective)
    registry.register("yield_probability_tradeoff", yield_probability_tradeoff)
