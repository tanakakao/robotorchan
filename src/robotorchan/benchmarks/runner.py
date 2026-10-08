"""Seeded closed-loop benchmark runner with truth and observation histories."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

from robotorchan.benchmarks.config import BenchmarkExperimentConfig
from robotorchan.benchmarks.problem import BenchmarkProblem
from robotorchan.benchmarks.registry import BenchmarkProblemRegistry

CandidateGenerator = Callable[[BenchmarkProblem, Tensor, Tensor, int, torch.Generator], Tensor]


@dataclass(frozen=True)
class BenchmarkTrajectory:
    """One seeded experiment; initial evaluations precede proposed evaluations."""

    seed: int
    X: Tensor
    Y_observed: Tensor
    Y_truth: Tensor
    constraints: Tensor
    costs: Tensor
    initial_points: int

    @property
    def evaluation_count(self) -> int:
        """Number of candidate evaluations excluding the initial design."""
        return self.X.shape[0] - self.initial_points

    @property
    def cumulative_cost(self) -> Tensor:
        """Cumulative evaluation costs including the initial design."""
        return self.costs.squeeze(-1).cumsum(dim=0)


def sobol_initial_design(
    problem: BenchmarkProblem,
    n: int,
    seed: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    """Generate reproducible continuous or mixed-variable initial designs."""
    engine = torch.quasirandom.SobolEngine(problem.dimension, scramble=True, seed=seed)
    unit = engine.draw(n).to(dtype=dtype, device=device)
    bounds = problem.bounds.to(dtype=dtype, device=device)
    X = bounds[0] + unit * (bounds[1] - bounds[0])
    for index, kind in enumerate(problem.variable_types):
        if kind != "continuous":
            X[:, index] = X[:, index].round()
    return X


def random_candidates(
    problem: BenchmarkProblem,
    X: Tensor,
    Y: Tensor,
    q: int,
    generator: torch.Generator,
) -> Tensor:
    """Random-search baseline using the run-local generator."""
    del Y
    unit = torch.rand(
        (q, problem.dimension),
        dtype=X.dtype,
        device=X.device,
        generator=generator,
    )
    bounds = problem.bounds.to(dtype=X.dtype, device=X.device)
    candidates = bounds[0] + unit * (bounds[1] - bounds[0])
    for index, kind in enumerate(problem.variable_types):
        if kind != "continuous":
            candidates[:, index] = candidates[:, index].round()
    return candidates


def run_benchmark(
    config: BenchmarkExperimentConfig,
    candidate_generator: CandidateGenerator,
    *,
    registry: BenchmarkProblemRegistry | None = None,
) -> tuple[BenchmarkTrajectory, ...]:
    """Run independent seeds with a strict evaluation budget.

    The candidate callback receives the problem, evaluated X, observed Y,
    requested batch size, and a run-local generator. It must return (q, d).
    """
    problem = config.resolve_problem(registry)
    device = config.torch_device
    dtype = config.torch_dtype
    trajectories = []
    for seed in config.seeds:
        generator = torch.Generator(device=device).manual_seed(seed)
        X = sobol_initial_design(
            problem, config.initial_points, seed, dtype=dtype, device=device
        )
        observed = problem.evaluate_observation(X)
        truth = problem.evaluate_truth(X)
        constraints = problem.evaluate_constraints(X)
        costs = problem.evaluate_cost(X)
        remaining = config.evaluation_budget
        while remaining:
            batch_size = min(config.q, remaining)
            candidates = candidate_generator(
                problem, X.clone(), observed.clone(), batch_size, generator
            )
            if not isinstance(candidates, Tensor) or candidates.shape != (
                batch_size,
                problem.dimension,
            ):
                raise ValueError("Candidate generator must return a tensor of shape (q, d).")
            problem._validate_X(candidates)
            new_observed = problem.evaluate_observation(candidates)
            new_truth = problem.evaluate_truth(candidates)
            new_constraints = problem.evaluate_constraints(candidates)
            new_costs = problem.evaluate_cost(candidates)
            X = torch.cat((X, candidates), dim=0)
            observed = torch.cat((observed, new_observed), dim=0)
            truth = torch.cat((truth, new_truth), dim=0)
            constraints = torch.cat((constraints, new_constraints), dim=0)
            costs = torch.cat((costs, new_costs), dim=0)
            remaining -= batch_size
        trajectories.append(
            BenchmarkTrajectory(
                seed=seed,
                X=X,
                Y_observed=observed,
                Y_truth=truth,
                constraints=constraints,
                costs=costs,
                initial_points=config.initial_points,
            )
        )
    return tuple(trajectories)
