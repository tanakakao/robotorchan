"""Candidate-feasibility and trust-region benchmark scenarios.

Candidate constraints restrict where the optimizer may propose X; they
are intentionally separate from observed outcome constraints.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

from robotorchan.benchmarks.problem import BenchmarkProblem
from robotorchan.benchmarks.registry import BenchmarkProblemRegistry

CandidateConstraint = Callable[[Tensor], Tensor]


@dataclass(frozen=True)
class CandidateConstraintScenario:
    """A benchmark and its deterministic candidate-domain feasibility rule."""

    problem: BenchmarkProblem
    feasible: CandidateConstraint

    def validate(self, X: Tensor) -> None:
        """Reject candidates outside the domain or violating feasibility."""
        self.problem._validate_X(X)
        mask = self.feasible(X)
        if not isinstance(mask, Tensor) or mask.shape != X.shape[:-1] or mask.dtype != torch.bool:
            raise ValueError("Candidate feasibility must return boolean shape (..., q).")
        if not mask.all():
            raise ValueError("Candidate constraints violated.")


def _sphere_objective(X: Tensor) -> Tensor:
    return (X - 0.75).square().sum(dim=-1, keepdim=True)


def candidate_linear_region() -> CandidateConstraintScenario:
    """Restrict a 2D sphere to the half-space x0+x1 <= 1."""

    problem = BenchmarkProblem(
        name="candidate_linear_region",
        bounds=torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double),
        objective=_sphere_objective,
        directions=("minimize",),
        variable_types=("continuous", "continuous"),
    )
    return CandidateConstraintScenario(
        problem=problem,
        feasible=lambda X: X.sum(dim=-1) <= 1.0,
    )


def candidate_nonlinear_region() -> CandidateConstraintScenario:
    """Restrict a 2D sphere to a disk centered at (0.5, 0.5)."""

    problem = BenchmarkProblem(
        name="candidate_nonlinear_region",
        bounds=torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double),
        objective=_sphere_objective,
        directions=("minimize",),
        variable_types=("continuous", "continuous"),
    )
    return CandidateConstraintScenario(
        problem=problem,
        feasible=lambda X: ((X - 0.5).square().sum(dim=-1) <= 0.2**2),
    )


def register_candidate_constraint_problems(registry: BenchmarkProblemRegistry) -> None:
    """Register the objective contracts; feasibility remains a separate layer."""
    for scenario in (candidate_linear_region(), candidate_nonlinear_region()):
        registry.register(scenario.problem.name, lambda scenario=scenario: scenario.problem)


@dataclass
class TrustRegionState:
    """Minimal TuRBO-style region state for deterministic benchmark diagnostics."""

    center: Tensor
    length: float = 0.8
    min_length: float = 0.05
    max_length: float = 1.6
    success_tolerance: int = 3
    failure_tolerance: int = 3
    successes: int = 0
    failures: int = 0
    restart_required: bool = False

    def __post_init__(self) -> None:
        if self.center.ndim != 1 or not self.center.is_floating_point():
            raise ValueError("center must be a floating vector.")
        if not torch.isfinite(self.center).all():
            raise ValueError("center must be finite.")
        if not 0 < self.min_length <= self.length <= self.max_length:
            raise ValueError("Require 0 < min_length <= length <= max_length.")
        if self.success_tolerance < 1 or self.failure_tolerance < 1:
            raise ValueError("Tolerances must be positive.")

    def bounds(self, domain_bounds: Tensor) -> Tensor:
        """Return a clipped axis-aligned trust region."""
        if domain_bounds.shape != (2, self.center.numel()):
            raise ValueError("Domain bounds must have shape (2, d).")
        center = self.center.to(dtype=domain_bounds.dtype, device=domain_bounds.device)
        half = self.length / 2.0
        return torch.stack((
            torch.maximum(domain_bounds[0], center - half),
            torch.minimum(domain_bounds[1], center + half),
        ))

    def update(self, improved: bool) -> None:
        """Expand on repeated success; shrink and request restart on failure."""
        if improved:
            self.successes += 1
            self.failures = 0
            if self.successes >= self.success_tolerance:
                self.length = min(self.max_length, 2.0 * self.length)
                self.successes = 0
        else:
            self.failures += 1
            self.successes = 0
            if self.failures >= self.failure_tolerance:
                self.length /= 2.0
                self.failures = 0
                self.restart_required = self.length < self.min_length
