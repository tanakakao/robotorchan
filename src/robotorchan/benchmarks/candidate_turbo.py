"""Candidate-space constraints and trust-region benchmark baselines.

Candidate constraints are separate from observed outcome constraints.
This module supplies a transparent local-search baseline, not TuRBO's
adaptive success/failure state machine.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

from robotorchan.benchmarks.problem import BenchmarkProblem
from robotorchan.benchmarks.registry import BenchmarkProblemRegistry

CandidateConstraint = Callable[[Tensor], Tensor]


def _objective(X: Tensor) -> Tensor:
    return -((X[..., 0] - 0.8).square() + (X[..., 1] - 0.8).square()).unsqueeze(-1)


def _linear_candidate_constraint(X: Tensor) -> Tensor:
    return (1.0 - X[..., 0] - X[..., 1]).unsqueeze(-1)


def _nonlinear_candidate_constraint(X: Tensor) -> Tensor:
    return (0.16 - (X[..., 0] - 0.5).square() - (X[..., 1] - 0.5).square()).unsqueeze(-1)


def candidate_linear_region() -> BenchmarkProblem:
    """Quadratic objective with a separately specified linear candidate region."""
    return BenchmarkProblem(
        name="candidate_linear_region",
        bounds=torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double),
        objective=_objective,
        directions=("maximize",),
        variable_types=("continuous", "continuous"),
        optimal_value=torch.tensor([-0.18], dtype=torch.double),
    )


def candidate_nonlinear_region() -> BenchmarkProblem:
    """Quadratic objective with a separately specified nonlinear candidate region."""
    return BenchmarkProblem(
        name="candidate_nonlinear_region",
        bounds=torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double),
        objective=_objective,
        directions=("maximize",),
        variable_types=("continuous", "continuous"),
    )


def candidate_constraint(problem_name: str) -> CandidateConstraint:
    """Return a g(X)>=0 candidate feasibility rule for a benchmark name."""
    rules = {
        "candidate_linear_region": _linear_candidate_constraint,
        "candidate_nonlinear_region": _nonlinear_candidate_constraint,
    }
    try:
        return rules[problem_name]
    except KeyError as error:
        raise ValueError(f"Unknown candidate constraint benchmark: {problem_name}") from error


@dataclass(frozen=True)
class TrustRegionCandidateGenerator:
    """Sample feasible points within a fixed box trust region around a center.

    The region is clipped to global bounds. Rejection sampling is capped,
    and failure is explicit rather than returning infeasible candidates.
    """

    center: Tensor
    length: float
    constraint: CandidateConstraint
    max_draws: int = 4096

    def __post_init__(self) -> None:
        if self.center.ndim != 1 or not torch.isfinite(self.center).all():
            raise ValueError("center must be a finite one-dimensional tensor.")
        if not 0 < self.length <= 2:
            raise ValueError("length must lie in (0, 2].")
        if type(self.max_draws) is not int or self.max_draws < 1:
            raise ValueError("max_draws must be a positive integer.")

    def __call__(
        self,
        problem: BenchmarkProblem,
        X: Tensor,
        Y: Tensor,
        q: int,
        generator: torch.Generator,
    ) -> Tensor:
        """Return q feasible candidates in the fixed trust region."""
        del Y
        if self.center.numel() != problem.dimension:
            raise ValueError("center dimension does not match problem.")
        if type(q) is not int or q < 1:
            raise ValueError("q must be a positive integer.")
        bounds = problem.bounds.to(dtype=X.dtype, device=X.device)
        center = self.center.to(dtype=X.dtype, device=X.device)
        if ((center < bounds[0]) | (center > bounds[1])).any():
            raise ValueError("center must lie within problem bounds.")
        radius = self.length * (bounds[1] - bounds[0]) / 2
        lower = torch.maximum(bounds[0], center - radius)
        upper = torch.minimum(bounds[1], center + radius)
        accepted: list[Tensor] = []
        for _ in range(self.max_draws):
            point = lower + torch.rand(
                (1, problem.dimension), dtype=X.dtype, device=X.device, generator=generator
            ) * (upper - lower)
            for index, kind in enumerate(problem.variable_types):
                if kind != "continuous":
                    point[:, index] = point[:, index].round()
            residual = self.constraint(point)
            if residual.ndim != 2 or residual.shape[0] != 1:
                raise ValueError("Candidate constraint must return shape (q, c).")
            if not torch.isfinite(residual).all():
                raise ValueError("Candidate constraint must be finite.")
            if (residual >= 0).all():
                accepted.append(point.squeeze(0))
                if len(accepted) == q:
                    return torch.stack(accepted)
        raise RuntimeError("Unable to sample feasible trust-region candidates.")


def register_candidate_region_problems(registry: BenchmarkProblemRegistry) -> None:
    """Register candidate-space constrained benchmark objectives."""
    registry.register("candidate_linear_region", candidate_linear_region)
    registry.register("candidate_nonlinear_region", candidate_nonlinear_region)
