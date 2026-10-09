"""Regression objectives and binary feasibility labels for heterogeneous BO."""

from __future__ import annotations

import torch
from torch import Tensor

from robotorchan.benchmarks.problem import BenchmarkProblem
from robotorchan.benchmarks.registry import BenchmarkProblemRegistry


def _strength_conductivity(X: Tensor) -> Tensor:
    """Two competing continuous responses on a two-dimensional process domain."""
    strength = 1.0 - (X[..., 0] - 0.8).square() - 0.25 * (X[..., 1] - 0.3).square()
    conductivity = 1.0 - (X[..., 0] - 0.2).square() - 0.25 * (X[..., 1] - 0.7).square()
    return torch.stack((strength, conductivity), dim=-1)


def _pass_margin(X: Tensor) -> Tensor:
    """Signed feasibility margin: positive means Pass."""
    return (X[..., 1] - 0.25 - 0.5 * X[..., 0]).unsqueeze(-1)


def strength_conductivity_pass_labels(X: Tensor) -> Tensor:
    """Return canonical 0/1 Pass labels with shape (..., q, 1).

    Boundary points are feasible and assigned label 1.
    """
    return (_pass_margin(X) >= 0).to(dtype=X.dtype)


def strength_conductivity_pass() -> BenchmarkProblem:
    """Two regression objectives (maximize) subject to binary Pass feasibility.

    The BenchmarkProblem constraint is the *signed deterministic margin*, not
    the binary label. Models should train a classifier on the separate labels
    and use its predictive Pass probability for acquisition constraints.
    """
    return BenchmarkProblem(
        name="strength_conductivity_pass",
        bounds=torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double),
        objective=_strength_conductivity,
        directions=("maximize", "maximize"),
        variable_types=("continuous", "continuous"),
        constraints=_pass_margin,
        n_constraints=1,
        reference_point=torch.tensor([0.0, 0.0], dtype=torch.double),
    )


def register_heterogeneous_problems(registry: BenchmarkProblemRegistry) -> None:
    """Register heterogeneous benchmark problems explicitly."""
    registry.register("strength_conductivity_pass", strength_conductivity_pass)
