"""End-to-end tests for reconstructing reports from saved histories."""

from __future__ import annotations

from dataclasses import replace

import pytest

from robotorchan.benchmarks.comparison_report import build_comparison_report
from robotorchan.benchmarks.config import BenchmarkExperimentConfig
from robotorchan.benchmarks.registry import BenchmarkProblemRegistry
from robotorchan.benchmarks.report_reconstruction import (
    reconstruct_comparison_from_files,
    reconstruct_comparison_report,
)
from robotorchan.benchmarks.runner import random_candidates, run_benchmark
from robotorchan.benchmarks.standard_problems import register_standard_problems
from robotorchan.benchmarks.trajectory_persistence import (
    TrajectoryArtifact,
    save_trajectory_artifact,
)


def _experiments():
    registry = BenchmarkProblemRegistry()
    register_standard_problems(registry)
    artifacts = {}
    for name in ("random", "sobol"):
        config = BenchmarkExperimentConfig(
            problem="branin",
            strategy=name,
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
        artifacts[name] = TrajectoryArtifact(config=config, trajectories=runs)
    return registry, artifacts


def test_saved_reports_match_in_memory_comparison(tmp_path) -> None:
    registry, artifacts = _experiments()
    problem = registry.create("branin")
    expected = build_comparison_report(
        problem,
        {name: (item.config, item.trajectories) for name, item in artifacts.items()},
        metric="simple_regret",
        reference="random",
        bootstrap_samples=100,
    )
    paths = {}
    for name, artifact in artifacts.items():
        path = tmp_path / f"{name}.json"
        save_trajectory_artifact(artifact, path)
        paths[name] = path
    actual = reconstruct_comparison_from_files(
        paths,
        metric="simple_regret",
        reference="random",
        registry=registry,
        bootstrap_samples=100,
    )
    assert actual.to_dict() == expected.to_dict()


def test_reconstruction_rejects_mismatched_strategy() -> None:
    registry, artifacts = _experiments()
    artifact = artifacts["sobol"]
    artifacts["sobol"] = replace(artifact, config=replace(artifact.config, strategy="other"))
    with pytest.raises(ValueError, match="strategy"):
        reconstruct_comparison_report(
            artifacts,
            metric="simple_regret",
            reference="random",
            registry=registry,
            bootstrap_samples=100,
        )


def test_reconstruction_rejects_missing_reference() -> None:
    registry, artifacts = _experiments()
    with pytest.raises(ValueError, match="Reference strategy"):
        reconstruct_comparison_report(
            artifacts,
            metric="simple_regret",
            reference="missing",
            registry=registry,
            bootstrap_samples=100,
        )
