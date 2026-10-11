"""Smoke-test actual acquisition paths with restricted benchmark metadata."""

from __future__ import annotations

import pytest
import torch

from robotorchan.benchmarks.comparative_qehvi import make_comparative_qehvi_strategy
from robotorchan.benchmarks.comparative_qei import make_comparative_qei_strategy
from robotorchan.benchmarks.comparative_qnehvi import make_comparative_qnehvi_strategy
from robotorchan.benchmarks.comparative_qnei import make_comparative_qnei_strategy
from robotorchan.benchmarks.config import BenchmarkExperimentConfig
from robotorchan.benchmarks.multiobjective_problems import register_multiobjective_problems
from robotorchan.benchmarks.registry import BenchmarkProblemRegistry
from robotorchan.benchmarks.runner import run_benchmark
from robotorchan.benchmarks.standard_problems import register_standard_problems


@pytest.mark.parametrize(
    ("strategy", "factory", "problem_name"),
    [
        ("qEI", make_comparative_qei_strategy, "branin"),
        ("qNEI", make_comparative_qnei_strategy, "branin"),
        ("qEHVI", make_comparative_qehvi_strategy, "branin_currin"),
        ("qNEHVI", make_comparative_qnehvi_strategy, "branin_currin"),
    ],
)
def test_real_candidate_smoke(strategy, factory, problem_name):
    """Fit real GPs and propose a candidate without access to truth evaluators."""
    registry = BenchmarkProblemRegistry()
    if problem_name == "branin":
        register_standard_problems(registry)
        callback = factory(num_restarts=1, raw_samples=8)
        n_objectives = 1
    else:
        register_multiobjective_problems(registry)
        callback = factory(num_restarts=1, raw_samples=8, mc_samples=8)
        n_objectives = 2
    config = BenchmarkExperimentConfig(
        problem=problem_name,
        strategy=strategy,
        seeds=(0,),
        initial_points=4,
        evaluation_budget=1,
        q=1,
    )
    runs = run_benchmark(
        config,
        callback,
        registry=registry,
        restrict_candidate_problem=True,
        independent_rng_streams=True,
        isolate_strategy_rng=True,
    )
    run = runs[0]
    assert run.X.shape == (5, 2)
    assert run.Y_observed.shape == (5, n_objectives)
    assert run.completed_batches == (1,)
    assert torch.isfinite(run.Y_truth).all()
