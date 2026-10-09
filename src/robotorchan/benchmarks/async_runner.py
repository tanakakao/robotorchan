"""Deterministic batch and asynchronous benchmark simulation.

Durations are simulated logical time, not measured wall-clock latency.
"""

from __future__ import annotations

import heapq
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

from robotorchan.benchmarks.config import BenchmarkExperimentConfig
from robotorchan.benchmarks.problem import BenchmarkProblem
from robotorchan.benchmarks.registry import BenchmarkProblemRegistry
from robotorchan.benchmarks.runner import sobol_initial_design

AsyncCandidateGenerator = Callable[
    [BenchmarkProblem, Tensor, Tensor, Tensor, int, torch.Generator], Tensor
]
DurationModel = Callable[[Tensor], float]


@dataclass(frozen=True)
class AsyncEvaluation:
    """One candidate evaluation in submission order."""

    candidate: Tensor
    submitted_at: float
    completed_at: float
    submission_index: int
    completion_index: int


@dataclass(frozen=True)
class AsyncBenchmarkResult:
    """Completed observations and the corresponding asynchronous event history."""

    seed: int
    X: Tensor
    Y_observed: Tensor
    Y_truth: Tensor
    constraints: Tensor
    costs: Tensor
    initial_points: int
    evaluations: tuple[AsyncEvaluation, ...]
    max_concurrency: int
    simulated_makespan: float

    @property
    def completion_order(self) -> tuple[int, ...]:
        """Return submission indices ordered by completion."""
        return tuple(
            event.submission_index
            for event in sorted(self.evaluations, key=lambda event: event.completion_index)
        )


def run_async_benchmark(
    config: BenchmarkExperimentConfig,
    candidate_generator: AsyncCandidateGenerator,
    duration_model: DurationModel,
    *,
    max_concurrency: int,
    registry: BenchmarkProblemRegistry | None = None,
) -> tuple[AsyncBenchmarkResult, ...]:
    """Simulate a capacity-limited asynchronous experiment for each seed.

    The generator receives completed X/Y, pending X, requested q, and a local RNG.
    Each call fills up to min(q, available workers, remaining budget) slots.
    Durations must be finite and strictly positive. Ties complete in submission order.
    Initial observations are available at simulated time zero and excluded from budget.
    """
    if type(max_concurrency) is not int or max_concurrency < 1:
        raise ValueError("max_concurrency must be a positive integer.")
    results = []
    for seed in config.seeds:
        problem = config.resolve_problem(registry)
        generator = torch.Generator(device=config.torch_device).manual_seed(seed)
        X = sobol_initial_design(
            problem,
            config.initial_points,
            seed,
            dtype=config.torch_dtype,
            device=config.torch_device,
        )
        observed = problem.evaluate_observation(X)
        truth = problem.evaluate_truth(X)
        constraints = problem.evaluate_constraints(X)
        costs = problem.evaluate_cost(X)
        pending: list[tuple[float, int, Tensor]] = []
        events: dict[int, AsyncEvaluation] = {}
        submitted = 0
        completed = 0
        clock = 0.0
        peak = 0
        while completed < config.evaluation_budget:
            available = max_concurrency - len(pending)
            remaining = config.evaluation_budget - submitted
            if available > 0 and remaining > 0:
                count = min(config.q, available, remaining)
                pending_X = (
                    torch.stack([entry[2] for entry in pending])
                    if pending
                    else X.new_empty((0, problem.dimension))
                )
                candidates = candidate_generator(
                    problem, X.clone(), observed.clone(), pending_X.clone(), count, generator
                )
                if not isinstance(candidates, Tensor) or candidates.shape != (
                    count,
                    problem.dimension,
                ):
                    raise ValueError("Candidate generator must return a tensor of shape (q, d).")
                problem._validate_X(candidates)
                for candidate in candidates:
                    duration = float(duration_model(candidate.clone()))
                    if not torch.isfinite(torch.tensor(duration)) or duration <= 0:
                        raise ValueError("Simulated durations must be positive and finite.")
                    finish = clock + duration
                    heapq.heappush(pending, (finish, submitted, candidate.clone()))
                    events[submitted] = AsyncEvaluation(
                        candidate=candidate.clone(),
                        submitted_at=clock,
                        completed_at=finish,
                        submission_index=submitted,
                        completion_index=-1,
                    )
                    submitted += 1
                peak = max(peak, len(pending))
            if not pending:
                raise RuntimeError("No pending evaluations while the budget remains.")
            finish, index, candidate = heapq.heappop(pending)
            clock = finish
            point = candidate.unsqueeze(0)
            X = torch.cat((X, point), dim=0)
            observed = torch.cat((observed, problem.evaluate_observation(point)), dim=0)
            truth = torch.cat((truth, problem.evaluate_truth(point)), dim=0)
            constraints = torch.cat((constraints, problem.evaluate_constraints(point)), dim=0)
            costs = torch.cat((costs, problem.evaluate_cost(point)), dim=0)
            old = events[index]
            events[index] = AsyncEvaluation(
                candidate=old.candidate,
                submitted_at=old.submitted_at,
                completed_at=old.completed_at,
                submission_index=index,
                completion_index=completed,
            )
            completed += 1
        results.append(
            AsyncBenchmarkResult(
                seed=seed,
                X=X,
                Y_observed=observed,
                Y_truth=truth,
                constraints=constraints,
                costs=costs,
                initial_points=config.initial_points,
                evaluations=tuple(events[index] for index in range(submitted)),
                max_concurrency=peak,
                simulated_makespan=clock,
            )
        )
    return tuple(results)
