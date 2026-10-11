"""Regression tests for the aggregate fairness report."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from robotorchan.benchmarks.config import BenchmarkExperimentConfig
from robotorchan.benchmarks.fairness_report import build_fairness_report
from robotorchan.benchmarks.registry import BenchmarkProblemRegistry
from robotorchan.benchmarks.runner import random_candidates, run_benchmark
from robotorchan.benchmarks.standard_problems import register_standard_problems


def _experiment(strategy: str):
    registry = BenchmarkProblemRegistry()
    register_standard_problems(registry)
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
        registry=registry,
        restrict_candidate_problem=True,
        independent_rng_streams=True,
        isolate_strategy_rng=True,
    )
    return config, runs


def test_fairness_report_is_serializable() -> None:
    report = build_fairness_report(
        {"random": _experiment("random"), "sobol": _experiment("sobol")},
        require_cost_pairing=True,
    )
    data = json.loads(json.dumps(report.to_dict()))
    assert data["initial"]["design_paired"]
    assert data["initial"]["truth_paired"]
    assert data["initial"]["observation_paired"]
    assert data["budget"]["completed_batches"] == [3, 5]
    assert data["budget"]["cost_paired"]


def test_report_rejects_mismatched_initial_design() -> None:
    first = _experiment("random")
    second_config, second_runs = _experiment("sobol")
    changed_x = second_runs[0].X.clone()
    changed_x[0, 0] += 0.01
    changed = (replace(second_runs[0], X=changed_x), second_runs[1])
    with pytest.raises(ValueError, match="Initial designs"):
        build_fairness_report({"random": first, "sobol": (second_config, changed)})


def test_report_rejects_mismatched_budget() -> None:
    first = _experiment("random")
    second_config, second_runs = _experiment("sobol")
    with pytest.raises(ValueError, match="mismatched experiment settings"):
        build_fairness_report(
            {"random": first, "sobol": (replace(second_config, q=1), second_runs)}
        )
