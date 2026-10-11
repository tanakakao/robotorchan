"""End-to-end comparison report serialization tests."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from robotorchan.benchmarks.comparison_report import build_comparison_report
from robotorchan.benchmarks.config import BenchmarkExperimentConfig
from robotorchan.benchmarks.registry import BenchmarkProblemRegistry
from robotorchan.benchmarks.runner import random_candidates, run_benchmark
from robotorchan.benchmarks.standard_problems import register_standard_problems


def _experiments():
    registry = BenchmarkProblemRegistry()
    register_standard_problems(registry)
    experiments = {}
    for strategy in ("random", "sobol"):
        config = BenchmarkExperimentConfig(
            problem="branin",
            strategy=strategy,
            seeds=(0, 1, 2),
            initial_points=4,
            evaluation_budget=5,
            q=3,
        )
        runs = run_benchmark(
            config,
            random_candidates,
            registry=registry,
            independent_rng_streams=True,
            restrict_candidate_problem=True,
        )
        experiments[strategy] = (config, runs)
    return registry.create("branin"), experiments


def test_report_is_strict_json_serializable() -> None:
    problem, experiments = _experiments()
    report = build_comparison_report(
        problem,
        experiments,
        metric="simple_regret",
        reference="random",
        bootstrap_samples=100,
    )
    data = json.loads(json.dumps(report.to_dict(), allow_nan=False))
    assert data["metric"] == "simple_regret"
    assert data["reference"] == "random"
    assert data["fairness"]["initial"]["design_paired"]
    assert data["strategies"]["random"]["n_seeds"] == 3
    assert len(data["comparisons"]) == 1
    assert len(data["inference"]) == 1


def test_report_rejects_unfair_budget() -> None:
    problem, experiments = _experiments()
    config, runs = experiments["sobol"]
    experiments["sobol"] = (replace(config, q=1), runs)
    with pytest.raises(ValueError, match="mismatched experiment settings"):
        build_comparison_report(
            problem,
            experiments,
            metric="simple_regret",
            reference="random",
            bootstrap_samples=100,
        )


def test_report_rejects_missing_reference() -> None:
    problem, experiments = _experiments()
    with pytest.raises(ValueError, match="Reference strategy"):
        build_comparison_report(
            problem,
            experiments,
            metric="simple_regret",
            reference="missing",
            bootstrap_samples=100,
        )
