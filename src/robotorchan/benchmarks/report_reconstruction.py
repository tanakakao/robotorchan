"""Reconstruct empirical comparisons from persisted trajectory artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from robotorchan.benchmarks.comparison_report import (
    EmpiricalComparisonReport,
    build_comparison_report,
)
from robotorchan.benchmarks.metric_curves import MetricName
from robotorchan.benchmarks.problem import BenchmarkProblem
from robotorchan.benchmarks.registry import BenchmarkProblemRegistry
from robotorchan.benchmarks.trajectory_persistence import (
    TrajectoryArtifact,
    load_trajectory_artifact,
)


def reconstruct_comparison_report(
    artifacts: Mapping[str, TrajectoryArtifact],
    *,
    metric: MetricName,
    reference: str,
    registry: BenchmarkProblemRegistry | None = None,
    confidence_level: float = 0.95,
    bootstrap_samples: int = 2000,
    bootstrap_seed: int = 0,
    alpha: float = 0.05,
    require_observation_pairing: bool = True,
    require_cost_pairing: bool = False,
) -> EmpiricalComparisonReport:
    """Recompute a fairness-gated report without rerunning candidate generators."""
    if not artifacts:
        raise ValueError("At least one trajectory artifact is required.")
    if reference not in artifacts:
        raise ValueError("Reference strategy must exist in artifacts.")
    experiments = {}
    problem: BenchmarkProblem | None = None
    for name, artifact in artifacts.items():
        if name != artifact.config.strategy:
            raise ValueError("Artifact strategy does not match its mapping key.")
        if problem is None:
            problem = artifact.config.resolve_problem(registry)
        elif artifact.config.problem != problem.name:
            raise ValueError("All artifacts must refer to the same benchmark problem.")
        experiments[name] = (artifact.config, artifact.trajectories)
    assert problem is not None
    return build_comparison_report(
        problem,
        experiments,
        metric=metric,
        reference=reference,
        confidence_level=confidence_level,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
        alpha=alpha,
        require_observation_pairing=require_observation_pairing,
        require_cost_pairing=require_cost_pairing,
    )


def reconstruct_comparison_from_files(
    paths: Mapping[str, str | Path],
    *,
    metric: MetricName,
    reference: str,
    registry: BenchmarkProblemRegistry | None = None,
    confidence_level: float = 0.95,
    bootstrap_samples: int = 2000,
    bootstrap_seed: int = 0,
    alpha: float = 0.05,
    require_observation_pairing: bool = True,
    require_cost_pairing: bool = False,
) -> EmpiricalComparisonReport:
    """Load each saved experiment and rebuild its empirical comparison report."""
    artifacts = {name: load_trajectory_artifact(path) for name, path in paths.items()}
    return reconstruct_comparison_report(
        artifacts,
        metric=metric,
        reference=reference,
        registry=registry,
        confidence_level=confidence_level,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
        alpha=alpha,
        require_observation_pairing=require_observation_pairing,
        require_cost_pairing=require_cost_pairing,
    )
