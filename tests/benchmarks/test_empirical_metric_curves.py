"""Tests for empirical metric-curve alignment."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from robotorchan.benchmarks.config import BenchmarkExperimentConfig
from robotorchan.benchmarks.metric_curves import align_metric_curves
from robotorchan.benchmarks.registry import BenchmarkProblemRegistry
from robotorchan.benchmarks.runner import random_candidates, run_benchmark
from robotorchan.benchmarks.standard_problems import register_standard_problems


def _experiment():
    registry = BenchmarkProblemRegistry()
    register_standard_problems(registry)
    config = BenchmarkExperimentConfig(
        problem="branin",
        strategy="random",
        seeds=(0, 1),
        initial_points=4,
        evaluation_budget=5,
        q=3,
    )
    return registry.create("branin"), run_benchmark(
        config,
        random_candidates,
        registry=registry,
        independent_rng_streams=True,
    )


def test_metric_curves_include_initial_design_at_zero() -> None:
    problem, runs = _experiment()
    curves = align_metric_curves(problem, runs, "simple_regret")
    assert curves.evaluations == (0, 1, 2, 3, 4, 5)
    assert curves.values.shape == (2, 6)
    assert torch.equal(curves.final_values, curves.values[:, -1])
    assert torch.equal(curves.mean, curves.values.mean(dim=0))
    assert torch.all(curves.values[:, 1:] <= curves.values[:, :-1])


def test_metric_curves_can_exclude_initial_design() -> None:
    problem, runs = _experiment()
    curves = align_metric_curves(problem, runs, "feasibility_rate", include_initial=False)
    assert curves.evaluations == (1, 2, 3, 4, 5)
    assert curves.values.shape == (2, 5)
    assert torch.all(curves.values == 1)


def test_mismatched_evaluation_budgets_are_rejected() -> None:
    problem, runs = _experiment()
    changed = replace(
        runs[1],
        X=runs[1].X[:-1],
        Y_truth=runs[1].Y_truth[:-1],
        Y_observed=runs[1].Y_observed[:-1],
        constraints=runs[1].constraints[:-1],
        costs=runs[1].costs[:-1],
    )
    with pytest.raises(ValueError, match="share initial points and evaluation budget"):
        align_metric_curves(problem, (runs[0], changed), "simple_regret")


def test_duplicate_seeds_are_rejected() -> None:
    problem, runs = _experiment()
    with pytest.raises(ValueError, match="seeds must be unique"):
        align_metric_curves(problem, (runs[0], runs[0]), "simple_regret")
