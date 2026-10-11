"""Serializable, fairness-gated empirical benchmark comparison reports."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor

from robotorchan.benchmarks.config import BenchmarkExperimentConfig
from robotorchan.benchmarks.curve_statistics import summarize_metric_curves
from robotorchan.benchmarks.fairness_report import FairnessReport, build_fairness_report
from robotorchan.benchmarks.metric_curves import MetricName, align_metric_curves
from robotorchan.benchmarks.paired_comparison import compare_paired_strategies
from robotorchan.benchmarks.paired_inference import infer_paired_strategies
from robotorchan.benchmarks.problem import BenchmarkProblem
from robotorchan.benchmarks.runner import BenchmarkTrajectory


def _json_value(value: Any) -> Any:
    """Convert tensors and nested dataclass payloads to JSON-compatible values."""
    if isinstance(value, Tensor):
        return _json_value(value.detach().cpu().tolist())
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not torch.isfinite(torch.tensor(value)):
        raise ValueError("Nonfinite values cannot be serialized to strict JSON.")
    return value


@dataclass(frozen=True)
class EmpiricalComparisonReport:
    """Validated experiment configuration and statistical comparison results."""

    metric: MetricName
    reference: str
    fairness: FairnessReport
    strategies: dict[str, dict[str, Any]]
    comparisons: tuple[dict[str, Any], ...]
    inference: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a strict JSON-compatible report payload."""
        return _json_value(asdict(self))


def build_comparison_report(
    problem: BenchmarkProblem,
    experiments: Mapping[str, tuple[BenchmarkExperimentConfig, tuple[BenchmarkTrajectory, ...]]],
    *,
    metric: MetricName,
    reference: str,
    confidence_level: float = 0.95,
    bootstrap_samples: int = 2000,
    bootstrap_seed: int = 0,
    alpha: float = 0.05,
    require_observation_pairing: bool = True,
    require_cost_pairing: bool = False,
) -> EmpiricalComparisonReport:
    """Build a comparison only after enforcing the requested fairness gates."""
    if reference not in experiments:
        raise ValueError("Reference strategy must exist in experiments.")
    fairness = build_fairness_report(
        experiments,
        require_observation_pairing=require_observation_pairing,
        require_cost_pairing=require_cost_pairing,
    )
    if any(config.problem != problem.name for config, _ in experiments.values()):
        raise ValueError("Provided problem does not match experiment configurations.")
    curves = {
        name: align_metric_curves(problem, runs, metric) for name, (_, runs) in experiments.items()
    }
    summaries = {
        name: _json_value(
            asdict(
                summarize_metric_curves(
                    curve,
                    confidence_level=confidence_level,
                    bootstrap_samples=bootstrap_samples,
                    bootstrap_seed=bootstrap_seed,
                )
            )
        )
        for name, curve in curves.items()
    }
    comparisons = compare_paired_strategies(
        curves,
        reference=reference,
        confidence_level=confidence_level,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    inference = infer_paired_strategies(curves, reference=reference, alpha=alpha)
    return EmpiricalComparisonReport(
        metric=metric,
        reference=reference,
        fairness=fairness,
        strategies=summaries,
        comparisons=tuple(_json_value(asdict(item)) for item in comparisons),
        inference=tuple(_json_value(asdict(item)) for item in inference),
    )
