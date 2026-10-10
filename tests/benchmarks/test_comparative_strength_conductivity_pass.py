"""Phase 15 multiobjective binary feasibility benchmark tests."""

import pytest
import torch

from robotorchan.benchmarks.comparative_matrix import ComparativeExperimentCell
from robotorchan.benchmarks.comparative_strength_conductivity_pass import (
    run_strength_conductivity_pass,
)
from robotorchan.benchmarks.config import BenchmarkExperimentConfig
from robotorchan.benchmarks.heterogeneous_problems import (
    strength_conductivity_pass_labels,
)
from robotorchan.benchmarks.runner import random_candidates


def _cell(strategy: str = "random") -> ComparativeExperimentCell:
    return ComparativeExperimentCell(
        tier="smoke",
        config=BenchmarkExperimentConfig(
            problem="strength_conductivity_pass",
            strategy=strategy,
            seeds=(0,),
            initial_points=4,
            evaluation_budget=4,
            q=1,
        ),
    )


@pytest.mark.parametrize("strategy", ["random", "sobol"])
def test_heterogeneous_baseline(strategy: str) -> None:
    result = run_strength_conductivity_pass(_cell(strategy))
    run = result.trajectories[0]
    assert result.evaluations.tolist() == [0, 1, 2, 3, 4]
    assert run.Y_observed.shape == (8, 2)
    assert result.pass_labels[0].shape == (8, 1)
    assert result.feasible_hypervolume.shape == (1, 5)
    assert result.feasibility_rate.shape == (1, 5)
    torch.testing.assert_close(result.pass_labels[0], strength_conductivity_pass_labels(run.X))
    assert (result.feasible_hypervolume[:, 1:] >= result.feasible_hypervolume[:, :-1]).all()


def test_candidate_receives_binary_labels_without_truth() -> None:
    seen = []

    def propose(problem, X, Y, labels, q, generator):
        assert Y.shape == (X.shape[0], 2)
        assert labels.shape == (X.shape[0], 1)
        assert not hasattr(problem, "constraints")
        assert not hasattr(problem, "objective")
        assert not hasattr(problem, "evaluate_constraints")
        seen.append(labels.clone())
        return random_candidates(problem, X, Y, q, generator)

    run_strength_conductivity_pass(_cell(), candidate_generator=propose)
    assert len(seen) == 4


def test_heterogeneous_reproducible() -> None:
    first = run_strength_conductivity_pass(_cell("sobol"))
    second = run_strength_conductivity_pass(_cell("sobol"))
    torch.testing.assert_close(first.trajectories[0].X, second.trajectories[0].X)
    torch.testing.assert_close(first.feasible_hypervolume, second.feasible_hypervolume)


def test_binary_strategy_required() -> None:
    with pytest.raises(ValueError, match="binary-aware"):
        run_strength_conductivity_pass(_cell("qEHVI"))
