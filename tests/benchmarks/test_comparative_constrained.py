"""Phase 16 constrained benchmark comparison tests."""

from dataclasses import replace

import pytest
import torch

from robotorchan.benchmarks.comparative_constrained import compare_constrained
from robotorchan.benchmarks.comparative_matrix import ComparativeExperimentCell
from robotorchan.benchmarks.comparative_strength_conductivity_pass import (
    run_strength_conductivity_pass,
)
from robotorchan.benchmarks.comparative_strength_pass import run_strength_pass
from robotorchan.benchmarks.config import BenchmarkExperimentConfig


def _cell(problem: str, strategy: str) -> ComparativeExperimentCell:
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


@pytest.mark.parametrize(
    ("problem", "runner", "metric"),
    [
        ("strength_pass", run_strength_pass, "feasible_regret"),
        (
            "strength_conductivity_pass",
            run_strength_conductivity_pass,
            "feasible_hypervolume",
        ),
    ],
)
def test_paired_constrained_baselines(problem, runner, metric) -> None:
    random = runner(_cell(problem, "random"))
    sobol = runner(_cell(problem, "sobol"))
    comparison = compare_constrained({"random": random, "sobol": sobol})
    assert comparison.metric == metric
    assert comparison.seeds == (0,)
    assert comparison.evaluations.tolist() == [0, 1, 2, 3, 4]
    assert set(comparison.scores_by_method) == {"random", "sobol"}
    assert comparison.scores_by_method["random"].shape == (1, 5)
    torch.testing.assert_close(
        comparison.mean_score_by_method["random"],
        comparison.scores_by_method["random"][0],
    )
    assert torch.count_nonzero(comparison.standard_error_by_method["random"]) == 0


def test_reject_incompatible_metrics() -> None:
    regression = run_strength_pass(_cell("strength_pass", "random"))
    multiobjective = run_strength_conductivity_pass(_cell("strength_conductivity_pass", "random"))
    with pytest.raises(ValueError, match="regret and hypervolume"):
        compare_constrained({"regression": regression, "multiobjective": multiobjective})


def test_reject_unpaired_initial_design() -> None:
    baseline = run_strength_pass(_cell("strength_pass", "random"))
    run = baseline.trajectories[0]
    modified = replace(run, X=run.X.clone())
    modified.X[0, 0] += 0.01
    mismatched = replace(baseline, trajectories=(modified,))
    with pytest.raises(ValueError, match="Initial designs"):
        compare_constrained({"original": baseline, "modified": mismatched})


def test_reject_empty_comparison() -> None:
    with pytest.raises(ValueError, match="At least one"):
        compare_constrained({})


def test_reject_different_problem_with_matching_initial_design() -> None:
    baseline = run_strength_pass(_cell("strength_pass", "random"))
    original = baseline.trajectories[0]
    other = replace(
        original,
        Y_truth=original.Y_truth.clone() + 1.0,
        constraints=original.constraints.clone() + 1.0,
    )
    incompatible = replace(baseline, trajectories=(other,))
    with pytest.raises(ValueError, match="Initial designs, histories"):
        compare_constrained({"original": baseline, "different_problem": incompatible})
