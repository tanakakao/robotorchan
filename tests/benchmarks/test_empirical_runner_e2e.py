"""End-to-end benchmark execution and persistence tests with real evaluations."""

import json

import pytest
import torch

from robotorchan.benchmarks.config import BenchmarkExperimentConfig
from robotorchan.benchmarks.metrics import simple_regret_curve
from robotorchan.benchmarks.persistence import load_trajectory, save_trajectory
from robotorchan.benchmarks.registry import BenchmarkProblemRegistry
from robotorchan.benchmarks.runner import random_candidates, run_benchmark
from robotorchan.benchmarks.standard_problems import register_standard_problems


@pytest.mark.parametrize("q, expected_batches", [(1, 5), (3, 2)])
def test_real_branin_round_trip(tmp_path, q, expected_batches):
    """Evaluate actual Branin points, persist and reload every measured field."""
    registry = BenchmarkProblemRegistry()
    register_standard_problems(registry)
    config = BenchmarkExperimentConfig(
        problem="branin",
        strategy="random",
        seeds=(0, 1),
        initial_points=4,
        evaluation_budget=5,
        q=q,
    )
    runs = run_benchmark(
        config,
        random_candidates,
        registry=registry,
        independent_rng_streams=True,
        isolate_strategy_rng=True,
    )
    problem = registry.create("branin")
    assert len(runs) == 2
    for run in runs:
        path = tmp_path / f"seed{run.seed}.json"
        save_trajectory(run, path)
        saved = json.loads(path.read_text(encoding="utf-8"))
        assert saved["schema_version"] == 2
        restored = load_trajectory(path)
        assert restored.seed == run.seed
        assert restored.initial_points == 4
        assert restored.evaluation_count == 5
        assert restored.X.shape == (9, problem.dimension)
        assert len(restored.completed_batches) == expected_batches
        assert restored.completed_batches[-1] == 5
        if q == 3:
            assert restored.completed_batches == (3, 5)
        assert len(restored.candidate_seconds) == expected_batches
        assert len(restored.evaluation_seconds) == expected_batches
        assert restored.initial_evaluation_seconds >= 0
        for field in ("X", "Y_truth", "Y_observed", "constraints", "costs"):
            torch.testing.assert_close(getattr(restored, field), getattr(run, field))
        torch.testing.assert_close(
            restored.Y_truth, problem.evaluate_truth(restored.X)
        )
        curve = simple_regret_curve(problem, restored)
        assert curve.shape == (9,)
        assert torch.isfinite(curve).all()
        assert torch.all(curve[1:] <= curve[:-1] + 1e-12)
