"""Seeded closed-loop benchmark runner with truth and observation histories."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from time import perf_counter

import torch
from torch import Tensor

from robotorchan.benchmarks.candidate_view import CandidateProblemView
from robotorchan.benchmarks.config import BenchmarkExperimentConfig
from robotorchan.benchmarks.observation_noise import SeededObservationNoise
from robotorchan.benchmarks.problem import BenchmarkProblem
from robotorchan.benchmarks.random_streams import derive_stream_seed, make_stream_generator
from robotorchan.benchmarks.registry import BenchmarkProblemRegistry
from robotorchan.benchmarks.strategy_rng import strategy_rng_scope

CandidateGenerator = Callable[
    [BenchmarkProblem | CandidateProblemView, Tensor, Tensor, int, torch.Generator], Tensor
]


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
    completed_batches: tuple[int, ...] = ()
    candidate_seconds: tuple[float, ...] = ()
    evaluation_seconds: tuple[float, ...] = ()
    initial_evaluation_seconds: float | None = None

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
    restrict_candidate_problem: bool = False,
    independent_rng_streams: bool = False,
    observation_noise: SeededObservationNoise | None = None,
    isolate_strategy_rng: bool = False,
) -> tuple[BenchmarkTrajectory, ...]:
    """Run independent seeds with a strict evaluation budget.

    The candidate callback receives the problem, evaluated X, observed Y,
    requested batch size, and a run-local generator. It must return (q, d).
    Set restrict_candidate_problem=True for oracle-free strategy-facing metadata.
    Set independent_rng_streams=True to isolate initial-design and candidate seeds.
    Seeded observation_noise overrides problem.observe and uses a separate RNG stream.
    Set isolate_strategy_rng=True to seed PyTorch global RNG per candidate batch.
    """
    device = config.torch_device
    dtype = config.torch_dtype
    trajectories = []
    for seed in config.seeds:
        problem = config.resolve_problem(registry)
        generator = (
            make_stream_generator(seed, "candidate", device)
            if independent_rng_streams
            else torch.Generator(device=device).manual_seed(seed)
        )
        noise_generator = (
            observation_noise.generator(seed, config.strategy, device)
            if observation_noise is not None
            else None
        )
        initial_seed = (
            derive_stream_seed(seed, "initial_design") if independent_rng_streams else seed
        )
        candidate_problem = (
            CandidateProblemView.from_problem(problem) if restrict_candidate_problem else problem
        )
        X = sobol_initial_design(
            problem, config.initial_points, initial_seed, dtype=dtype, device=device
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        initial_start = perf_counter()
        truth = problem.evaluate_truth(X)
        observed = (
            problem.evaluate_observation(X)
            if observation_noise is None
            else observation_noise.observe(truth, noise_generator)
        )
        constraints = problem.evaluate_constraints(X)
        costs = problem.evaluate_cost(X)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        initial_evaluation_seconds = perf_counter() - initial_start
        completed_batches = []
        candidate_seconds = []
        evaluation_seconds = []
        remaining = config.evaluation_budget
        while remaining:
            batch_size = min(config.q, remaining)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            candidate_start = perf_counter()
            scope = (
                strategy_rng_scope(seed, len(completed_batches), device)
                if isolate_strategy_rng
                else nullcontext()
            )
            with scope:
                candidates = candidate_generator(
                    candidate_problem, X.clone(), observed.clone(), batch_size, generator
                )
            if not isinstance(candidates, Tensor) or candidates.shape != (
                batch_size,
                problem.dimension,
            ):
                raise ValueError("Candidate generator must return a tensor of shape (q, d).")
            problem._validate_X(candidates)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            candidate_seconds.append(perf_counter() - candidate_start)
            evaluation_start = perf_counter()
            new_truth = problem.evaluate_truth(candidates)
            new_observed = (
                problem.evaluate_observation(candidates)
                if observation_noise is None
                else observation_noise.observe(new_truth, noise_generator)
            )
            new_constraints = problem.evaluate_constraints(candidates)
            new_costs = problem.evaluate_cost(candidates)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            evaluation_seconds.append(perf_counter() - evaluation_start)
            X = torch.cat((X, candidates), dim=0)
            observed = torch.cat((observed, new_observed), dim=0)
            truth = torch.cat((truth, new_truth), dim=0)
            constraints = torch.cat((constraints, new_constraints), dim=0)
            costs = torch.cat((costs, new_costs), dim=0)
            remaining -= batch_size
            completed_batches.append(config.evaluation_budget - remaining)
        trajectories.append(
            BenchmarkTrajectory(
                seed=seed,
                X=X,
                Y_observed=observed,
                Y_truth=truth,
                constraints=constraints,
                costs=costs,
                initial_points=config.initial_points,
                completed_batches=tuple(completed_batches),
                candidate_seconds=tuple(candidate_seconds),
                evaluation_seconds=tuple(evaluation_seconds),
                initial_evaluation_seconds=initial_evaluation_seconds,
            )
        )
    return tuple(trajectories)
