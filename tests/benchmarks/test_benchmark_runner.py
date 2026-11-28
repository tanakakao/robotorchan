"""Closed-loop benchmark runner regression tests."""

import pytest
import torch

from robotorchan.benchmarks.config import BenchmarkExperimentConfig
from robotorchan.benchmarks.problem import BenchmarkProblem
from robotorchan.benchmarks.registry import BenchmarkProblemRegistry
from robotorchan.benchmarks.runner import random_candidates, run_benchmark


def _registry() -> BenchmarkProblemRegistry:
    registry = BenchmarkProblemRegistry()

    def make_problem() -> BenchmarkProblem:
        return BenchmarkProblem(
            name="quadratic",
            bounds=torch.tensor([[0.0], [1.0]], dtype=torch.double),
            objective=lambda X: -(X - 0.5).square(),
            observe=lambda X: -(X - 0.5).square() + 0.1,
            directions=("maximize",),
            variable_types=("continuous",),
            constraints=lambda X: X - 0.2,
            n_constraints=1,
            cost=lambda X: X + 1,
            optimal_value=torch.tensor([0.0], dtype=torch.double),
        )

    registry.register("quadratic", make_problem)
    return registry


def test_multiseed_partial_batch_and_histories() -> None:
    config = BenchmarkExperimentConfig(
        problem="quadratic",
        strategy="random",
        seeds=(3, 7),
        initial_points=4,
        evaluation_budget=5,
        q=2,
    )
    trajectories = run_benchmark(config, random_candidates, registry=_registry())
    assert len(trajectories) == 2
    for trajectory in trajectories:
        assert trajectory.X.shape == (9, 1)
        assert trajectory.Y_observed.shape == (9, 1)
        assert trajectory.Y_truth.shape == (9, 1)
        assert trajectory.constraints.shape == (9, 1)
        assert trajectory.costs.shape == (9, 1)
        assert trajectory.evaluation_count == 5
        torch.testing.assert_close(
            trajectory.Y_observed - trajectory.Y_truth,
            torch.full((9, 1), 0.1, dtype=torch.double),
        )
        assert trajectory.cumulative_cost.shape == (9,)
    repeated = run_benchmark(config, random_candidates, registry=_registry())
    for first, second in zip(trajectories, repeated, strict=True):
        torch.testing.assert_close(first.X, second.X)


def test_candidate_shape_is_validated() -> None:
    config = BenchmarkExperimentConfig(
        problem="quadratic", strategy="invalid", initial_points=2, evaluation_budget=1
    )
    with pytest.raises(ValueError, match="shape"):
        run_benchmark(
            config,
            lambda problem, X, Y, q, generator: X,
            registry=_registry(),
        )


def test_explicit_design_delivers_only_observations_with_defensive_copies() -> None:
    from robotorchan.benchmarks.candidate_view import CandidateProblemView
    from robotorchan.benchmarks.constrained_problems import register_constrained_problems

    registry = BenchmarkProblemRegistry()
    register_constrained_problems(registry)
    config = BenchmarkExperimentConfig(
        problem="constrained_annulus",
        strategy="external",
        initial_points=2,
        evaluation_budget=3,
        q=2,
    )
    design = torch.tensor([[0.0, 0.0], [0.75, 0.0]], dtype=torch.float64)
    observed_batches = []

    def observe(x, y, constraints, costs, initial):
        observed_batches.append((x.clone(), y.clone(), constraints.clone(), costs.clone(), initial))
        x.fill_(99)
        y.fill_(99)
        constraints.fill_(99)
        costs.fill_(99)

    def propose(view, x, y, q, rng):
        assert isinstance(view, CandidateProblemView)
        assert len(x) == sum(len(batch[0]) for batch in observed_batches)
        return x.new_full((q, view.dimension), 0.5)

    (trajectory,) = run_benchmark(
        config,
        propose,
        registry=registry,
        restrict_candidate_problem=True,
        initial_design=design,
        observation_consumer=observe,
    )
    assert trajectory.completed_batches == (2, 3)
    assert [batch[4] for batch in observed_batches] == [True, False, False]
    assert torch.equal(trajectory.X[:2], design)
    assert torch.equal(trajectory.constraints, torch.cat([batch[2] for batch in observed_batches]))
    assert torch.equal(trajectory.costs, torch.cat([batch[3] for batch in observed_batches]))
