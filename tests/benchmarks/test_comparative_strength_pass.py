"""Phase 14 Strength/Pass benchmark contract tests."""

import pytest
import torch

from robotorchan.benchmarks.comparative_matrix import ComparativeExperimentCell
from robotorchan.benchmarks.comparative_strength_pass import run_strength_pass
from robotorchan.benchmarks.config import BenchmarkExperimentConfig
from robotorchan.benchmarks.regression_binary_problems import strength_pass_labels
from robotorchan.benchmarks.runner import random_candidates


def _cell(strategy: str = "random") -> ComparativeExperimentCell:
    return ComparativeExperimentCell(
        tier="smoke",
        config=BenchmarkExperimentConfig(
            problem="strength_pass",
            strategy=strategy,
            seeds=(0,),
            initial_points=4,
            evaluation_budget=4,
            q=1,
        ),
    )


@pytest.mark.parametrize("strategy", ["random", "sobol"])
def test_strength_pass_smoke(strategy: str) -> None:
    result = run_strength_pass(_cell(strategy))
    run = result.trajectories[0]
    assert result.evaluations.tolist() == [0, 1, 2, 3, 4]
    assert run.evaluation_count == 4
    assert result.feasible_regret.shape == (1, 5)
    assert result.feasibility_rate.shape == (1, 5)
    assert result.pass_labels[0].shape == (8, 1)
    assert set(result.pass_labels[0].unique().tolist()).issubset({0.0, 1.0})
    torch.testing.assert_close(result.pass_labels[0], strength_pass_labels(run.X))
    assert (result.feasible_regret[:, 1:] <= result.feasible_regret[:, :-1]).all()


def test_binary_candidate_receives_only_labels_not_margin() -> None:
    seen = []

    def propose(problem, X, Y, labels, q, generator):
        torch.testing.assert_close(labels, strength_pass_labels(X))
        assert labels.shape == (X.shape[0], 1)
        assert not hasattr(problem, "constraints")
        assert not hasattr(problem, "objective")
        assert not hasattr(problem, "optimal_value")
        assert not hasattr(problem, "evaluate_constraints")
        assert problem.dimension == X.shape[1]
        assert set(labels.unique().tolist()).issubset({0.0, 1.0})
        seen.append(labels.clone())
        return random_candidates(problem, X, Y, q, generator)

    result = run_strength_pass(_cell(), candidate_generator=propose)
    assert len(seen) == 4
    assert result.trajectories[0].evaluation_count == 4


def test_strength_pass_reproducible() -> None:
    first = run_strength_pass(_cell())
    second = run_strength_pass(_cell())
    torch.testing.assert_close(first.trajectories[0].X, second.trajectories[0].X)
    torch.testing.assert_close(first.feasibility_rate, second.feasibility_rate)
