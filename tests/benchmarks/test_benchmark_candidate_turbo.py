"""Test candidate-space feasibility and fixed trust-region search."""

import pytest
import torch

from robotorchan.benchmarks.candidate_turbo import (
    TrustRegionCandidateGenerator,
    candidate_constraint,
    register_candidate_region_problems,
)
from robotorchan.benchmarks.config import BenchmarkExperimentConfig
from robotorchan.benchmarks.registry import BenchmarkProblemRegistry
from robotorchan.benchmarks.runner import run_benchmark


@pytest.mark.parametrize("name", ["candidate_linear_region", "candidate_nonlinear_region"])
def test_candidate_region_runner(name) -> None:
    registry = BenchmarkProblemRegistry()
    register_candidate_region_problems(registry)
    constraint = candidate_constraint(name)
    strategy = TrustRegionCandidateGenerator(
        center=torch.tensor([0.5, 0.5], dtype=torch.double),
        length=0.6,
        constraint=constraint,
    )
    config = BenchmarkExperimentConfig(
        problem=name,
        strategy="fixed_trust_region",
        seeds=(2, 4),
        initial_points=4,
        evaluation_budget=5,
        q=2,
    )
    first = run_benchmark(config, strategy, registry=registry)
    second = run_benchmark(config, strategy, registry=registry)
    for a, b in zip(first, second, strict=True):
        torch.testing.assert_close(a.X, b.X)
        proposed = a.X[config.initial_points :]
        assert proposed.shape == (5, 2)
        assert (constraint(proposed) >= 0).all()
        assert ((proposed >= 0.2) & (proposed <= 0.8)).all()
        assert a.constraints.shape == (9, 0)


def test_linear_region_rejects_infeasible_candidates() -> None:
    constraint = candidate_constraint("candidate_linear_region")
    X = torch.tensor([[0.8, 0.8], [0.5, 0.5]], dtype=torch.double)
    assert constraint(X)[0, 0] < 0
    assert constraint(X)[1, 0] >= 0


def test_trust_region_fails_explicitly_when_no_feasible_points() -> None:
    registry = BenchmarkProblemRegistry()
    register_candidate_region_problems(registry)
    problem = registry.create("candidate_linear_region")
    strategy = TrustRegionCandidateGenerator(
        center=torch.tensor([0.9, 0.9], dtype=torch.double),
        length=0.1,
        constraint=candidate_constraint("candidate_linear_region"),
        max_draws=8,
    )
    with pytest.raises(RuntimeError, match="Unable to sample feasible"):
        strategy(problem, torch.zeros(1, 2), torch.zeros(1, 1), 1, torch.Generator())


def test_trust_region_validates_configuration() -> None:
    with pytest.raises(ValueError, match="length"):
        TrustRegionCandidateGenerator(
            center=torch.tensor([0.5, 0.5]),
            length=0,
            constraint=candidate_constraint("candidate_linear_region"),
        )
    with pytest.raises(ValueError, match="Unknown candidate"):
        candidate_constraint("unknown")
