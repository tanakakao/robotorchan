"""Statistical summary tests for seeded benchmark trajectories."""

from __future__ import annotations

import pytest
import torch

from robotorchan.benchmarks.curve_statistics import (
    paired_metric_difference,
    summarize_metric_curves,
)
from robotorchan.benchmarks.metric_curves import MetricCurves


def _curves(seeds: tuple[int, ...] = (0, 1, 2)) -> MetricCurves:
    return MetricCurves(
        metric="simple_regret",
        seeds=seeds,
        evaluations=(0, 1),
        values=torch.tensor([[3.0, 2.0], [2.0, 1.0], [1.0, 0.0]]),
    )


def test_bootstrap_statistics_are_repeatable() -> None:
    first = summarize_metric_curves(_curves(), bootstrap_samples=200, bootstrap_seed=7)
    second = summarize_metric_curves(_curves(), bootstrap_samples=200, bootstrap_seed=7)
    assert torch.equal(first.mean, torch.tensor([2.0, 1.0]))
    assert torch.equal(first.std, torch.tensor([1.0, 1.0]))
    assert torch.equal(first.sem, first.std / 3**0.5)
    assert torch.equal(first.ci_lower, second.ci_lower)
    assert torch.equal(first.ci_upper, second.ci_upper)
    assert torch.all(first.ci_lower <= first.mean)
    assert torch.all(first.ci_upper >= first.mean)


def test_paired_difference_reorders_by_seed() -> None:
    first = _curves()
    second = MetricCurves(
        metric="simple_regret",
        seeds=(2, 0, 1),
        evaluations=(0, 1),
        values=torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]),
    )
    differences = paired_metric_difference(first, second)
    assert torch.equal(
        differences.values,
        torch.tensor([[2.0, 1.0], [0.0, -1.0], [1.0, 0.0]]),
    )


def test_unpaired_seed_sets_are_rejected() -> None:
    first = _curves()
    second = _curves(seeds=(0, 1, 9))
    with pytest.raises(ValueError, match="identical seed sets"):
        paired_metric_difference(first, second)


def test_nonfinite_curves_are_rejected() -> None:
    curves = _curves()
    curves.values[0, 0] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        summarize_metric_curves(curves)


def test_single_seed_is_rejected() -> None:
    curves = _curves()
    single = MetricCurves(
        metric=curves.metric,
        seeds=(0,),
        evaluations=curves.evaluations,
        values=curves.values[:1],
    )
    with pytest.raises(ValueError, match="At least two"):
        summarize_metric_curves(single)
