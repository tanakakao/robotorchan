"""Test constrained eligibility and classification evidence contracts."""

import pytest
import torch

from robotorchan.benchmarks.constrained_evidence import (
    constrained_eligibility,
    observed_pass_labels,
    validate_constrained_trajectory,
)
from robotorchan.benchmarks.regression_binary_problems import strength_pass
from robotorchan.benchmarks.runner import BenchmarkTrajectory


@pytest.mark.parametrize(
    ("problem", "strategy"),
    [
        ("strength_pass", "qEI"),
        ("strength_pass", "qNEI"),
        ("strength_conductivity_pass", "qEHVI"),
        ("strength_conductivity_pass", "qNEHVI"),
    ],
)
def test_classification_acquisitions_remain_blocked(problem, strategy):
    assert constrained_eligibility(problem, strategy).eligibility == "blocked_classification"


@pytest.mark.parametrize("problem", ["strength_pass", "strength_conductivity_pass"])
@pytest.mark.parametrize("strategy", ["random", "sobol"])
def test_baselines_are_eligible(problem, strategy):
    assert constrained_eligibility(problem, strategy).eligibility == "eligible_baseline"


def test_unknown_constrained_strategy_rejected():
    with pytest.raises(ValueError):
        constrained_eligibility("strength_pass", "qEHVI")


def test_binary_labels_agree_with_evaluation_margin():
    problem = strength_pass()
    X = torch.tensor([[0.0, 0.0], [0.0, 1.0]], dtype=torch.double)
    labels = observed_pass_labels("strength_pass", X)
    assert labels.squeeze(-1).tolist() == [0.0, 1.0]
    assert torch.equal(labels.bool(), problem.evaluate_constraints(X) >= 0)


def test_baseline_trajectory_reports_feasibility():
    problem = strength_pass()
    X = torch.tensor([[0.0, 0.0], [0.0, 1.0]], dtype=torch.double).repeat(35, 1)
    truth = problem.evaluate_truth(X)
    constraints = problem.evaluate_constraints(X)
    run = BenchmarkTrajectory(
        seed=0,
        X=X,
        Y_observed=truth,
        Y_truth=truth,
        constraints=constraints,
        costs=torch.ones(70, 1, dtype=torch.double),
        initial_points=20,
        completed_batches=(50,),
    )
    result = validate_constrained_trajectory("strength_pass", "random", run)
    assert result["feasible_count"] == 35
    assert result["feasible_rate"] == 0.5
    assert result["zero_feasible"] is False
    assert result["label_counts"] == {"pass": 35, "fail": 35}


def test_blocked_acquisition_cannot_certify_trajectory():
    with pytest.raises(ValueError, match="Blocked"):
        validate_constrained_trajectory("strength_pass", "qEI", None)
