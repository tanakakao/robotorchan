"""Budget and evaluation-cost fairness regression tests."""

from __future__ import annotations

from dataclasses import replace

import pytest

from robotorchan.benchmarks.budget_fairness import (
    expected_completed_batches,
    validate_budget_fairness,
)
from robotorchan.benchmarks.config import BenchmarkExperimentConfig
from robotorchan.benchmarks.registry import BenchmarkProblemRegistry
from robotorchan.benchmarks.runner import random_candidates, run_benchmark
from robotorchan.benchmarks.standard_problems import register_standard_problems


def _registry() -> BenchmarkProblemRegistry:
    registry = BenchmarkProblemRegistry()
    register_standard_problems(registry)
    return registry


def _experiment(strategy: str):
    config = BenchmarkExperimentConfig(
        problem="branin",
        strategy=strategy,
        seeds=(0, 1),
        initial_points=4,
        evaluation_budget=5,
        q=3,
    )
    runs = run_benchmark(
        config,
        random_candidates,
        registry=_registry(),
        restrict_candidate_problem=True,
        independent_rng_streams=True,
    )
    return config, runs


def test_short_final_batch_schedule() -> None:
    assert expected_completed_batches(5, 3) == (3, 5)
    assert expected_completed_batches(6, 3) == (3, 6)


def test_identical_evaluation_budgets_pass() -> None:
    result = validate_budget_fairness(
        {"random": _experiment("random"), "sobol": _experiment("sobol")},
        require_cost_pairing=True,
    )
    assert result.completed_batches == (3, 5)
    assert result.cost_paired


def test_different_budget_is_rejected() -> None:
    first = _experiment("random")
    second_config, second_runs = _experiment("sobol")
    changed_config = replace(second_config, evaluation_budget=6)
    with pytest.raises(ValueError, match="mismatched experiment settings"):
        validate_budget_fairness({"random": first, "sobol": (changed_config, second_runs)})


def test_missing_batch_is_rejected() -> None:
    first = _experiment("random")
    second_config, second_runs = _experiment("sobol")
    changed_runs = (replace(second_runs[0], completed_batches=(3,)), second_runs[1])
    with pytest.raises(ValueError, match="completed-batch schedule"):
        validate_budget_fairness({"random": first, "sobol": (second_config, changed_runs)})


def test_mismatched_costs_are_optional() -> None:
    first = _experiment("random")
    second_config, second_runs = _experiment("sobol")
    changed_costs = second_runs[0].costs.clone()
    changed_costs[0, 0] += 1
    changed_runs = (replace(second_runs[0], costs=changed_costs), second_runs[1])
    experiments = {"random": first, "sobol": (second_config, changed_runs)}
    assert not validate_budget_fairness(experiments).cost_paired
    with pytest.raises(ValueError, match="Evaluation costs differ"):
        validate_budget_fairness(experiments, require_cost_pairing=True)
