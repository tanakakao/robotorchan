"""Oracle-isolation contracts for empirical candidate callbacks."""

from __future__ import annotations

import torch
from torch import Tensor

from robotorchan.benchmarks.candidate_view import CandidateProblemView
from robotorchan.benchmarks.config import BenchmarkExperimentConfig
from robotorchan.benchmarks.registry import BenchmarkProblemRegistry
from robotorchan.benchmarks.runner import random_candidates, run_benchmark
from robotorchan.benchmarks.standard_problems import register_standard_problems


def _registry() -> BenchmarkProblemRegistry:
    registry = BenchmarkProblemRegistry()
    register_standard_problems(registry)
    return registry


def test_restricted_random_baseline_runs_with_complete_budget() -> None:
    """Random candidates need only public search-space metadata."""
    config = BenchmarkExperimentConfig(
        problem="branin",
        strategy="random",
        seeds=(0,),
        initial_points=4,
        evaluation_budget=5,
        q=3,
    )
    trajectories = run_benchmark(
        config,
        random_candidates,
        restrict_candidate_problem=True,
        registry=_registry(),
    )
    assert len(trajectories) == 1
    assert trajectories[0].evaluation_count == 5
    assert trajectories[0].completed_batches == (3, 5)


def test_restricted_callback_cannot_access_oracle() -> None:
    """An acquisition callback must not receive objective or truth evaluators."""
    config = BenchmarkExperimentConfig(
        problem="branin",
        strategy="random",
        initial_points=4,
        evaluation_budget=1,
    )

    def inspect_callback(
        problem: CandidateProblemView,
        X: Tensor,
        Y: Tensor,
        q: int,
        generator: torch.Generator,
    ) -> Tensor:
        assert isinstance(problem, CandidateProblemView)
        for attribute in (
            "objective",
            "observe",
            "constraints",
            "optimal_value",
            "reference_front",
            "evaluate_truth",
            "evaluate_observation",
            "evaluate_constraints",
        ):
            assert not hasattr(problem, attribute)
        assert problem.dimension == X.shape[-1]
        return random_candidates(problem, X, Y, q, generator)

    run_benchmark(
        config, inspect_callback, registry=_registry(), restrict_candidate_problem=True
    )


def test_restricted_view_bounds_are_defensively_copied() -> None:
    """Callback-side bounds mutation cannot change the runner's true problem."""
    config = BenchmarkExperimentConfig(
        problem="branin",
        strategy="random",
        initial_points=4,
        evaluation_budget=1,
    )

    def mutate_callback(
        problem: CandidateProblemView,
        X: Tensor,
        Y: Tensor,
        q: int,
        generator: torch.Generator,
    ) -> Tensor:
        problem.bounds.fill_(0)
        return X[:q].clone()

    trajectory = run_benchmark(
        config, mutate_callback, registry=_registry(), restrict_candidate_problem=True
    )[0]
    assert torch.equal(trajectory.X[-1:], trajectory.X[:1])
