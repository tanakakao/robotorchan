"""End-to-end reproducibility checks for comparative baseline strategies."""

import pytest
import torch

from robotorchan.benchmarks.comparative_baselines import (
    run_comparative_baseline,
    sobol_candidates,
)
from robotorchan.benchmarks.comparative_matrix import ComparativeExperimentCell
from robotorchan.benchmarks.config import BenchmarkExperimentConfig
from robotorchan.benchmarks.standard_problems import branin


def _cell(strategy: str, problem: str = "branin") -> ComparativeExperimentCell:
    return ComparativeExperimentCell(
        tier="smoke",
        config=BenchmarkExperimentConfig(
            problem=problem,
            strategy=strategy,
            seeds=(0,),
            initial_points=4,
            evaluation_budget=4,
            q=1,
        ),
    )


@pytest.mark.parametrize("strategy", ["random", "sobol"])
def test_baseline_reproducible_and_budget_exact(strategy: str) -> None:
    cell = _cell(strategy)
    first = run_comparative_baseline(cell)[0]
    second = run_comparative_baseline(cell)[0]
    assert first.evaluation_count == 4
    assert first.X.shape[0] == 8
    torch.testing.assert_close(first.X, second.X)
    torch.testing.assert_close(first.Y_truth, second.Y_truth)


def test_random_and_sobol_share_initial_design() -> None:
    random = run_comparative_baseline(_cell("random"))[0]
    sobol = run_comparative_baseline(_cell("sobol"))[0]
    torch.testing.assert_close(random.X[:4], sobol.X[:4])
    assert not torch.equal(random.X[4:], sobol.X[4:])


@pytest.mark.parametrize("problem", ["strength_pass", "strength_conductivity_pass"])
def test_constrained_baselines_can_run_without_classification_labels(problem: str) -> None:
    trajectory = run_comparative_baseline(_cell("sobol", problem))[0]
    assert trajectory.evaluation_count == 4
    assert trajectory.constraints.shape[0] == 8


def test_baseline_runner_rejects_bo_strategy() -> None:
    cell = _cell("qEI")
    with pytest.raises(ValueError, match="Random and Sobol"):
        run_comparative_baseline(cell)


def test_sobol_sequence_is_independent_of_batch_partition() -> None:
    problem = branin()
    X = torch.zeros((4, problem.dimension), dtype=torch.double)
    Y = torch.zeros((4, 1), dtype=torch.double)
    generator = torch.Generator().manual_seed(17)
    combined = sobol_candidates(problem, X, Y, 5, generator, initial_points=4)
    first = sobol_candidates(problem, X, Y, 3, generator, initial_points=4)
    extended_X = torch.cat((X, first))
    second = sobol_candidates(problem, extended_X, Y, 2, generator, initial_points=4)
    torch.testing.assert_close(combined, torch.cat((first, second)))


def test_sobol_candidate_stream_does_not_match_next_run_initial_design() -> None:
    from robotorchan.benchmarks.runner import sobol_initial_design

    problem = branin()
    X = sobol_initial_design(problem, 4, 0, dtype=torch.double, device=torch.device("cpu"))
    Y = torch.zeros((4, 1), dtype=torch.double)
    candidate = sobol_candidates(
        problem, X, Y, 4, torch.Generator().manual_seed(0), initial_points=4
    )
    next_initial = sobol_initial_design(
        problem, 4, 1, dtype=torch.double, device=torch.device("cpu")
    )
    assert not torch.equal(candidate, next_initial)
