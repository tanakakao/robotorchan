"""Paired comparison regression tests."""

from __future__ import annotations

import pytest
import torch

from robotorchan.benchmarks.metric_curves import MetricCurves
from robotorchan.benchmarks.paired_comparison import compare_paired_strategies


def _curve(name: str, values: list[list[float]], seeds=(0, 1, 2)):
    del name
    return MetricCurves(
        metric="simple_regret",
        seeds=seeds,
        evaluations=(0, 1),
        values=torch.tensor(values),
    )


def test_paired_comparison_reports_improvement() -> None:
    baseline = _curve("random", [[5.0, 4.0], [4.0, 3.0], [3.0, 2.0]])
    improved = _curve("qei", [[4.0, 2.0], [3.0, 1.0], [2.0, 0.0]])
    results = compare_paired_strategies(
        {"random": baseline, "qei": improved},
        reference="random",
        bootstrap_samples=100,
    )
    assert len(results) == 1
    result = results[0]
    assert result.first == "qei"
    assert result.second == "random"
    assert result.lower_is_better
    assert torch.equal(result.final_differences, torch.tensor([-2.0, -2.0, -2.0]))
    assert result.final_win_rate.item() == 1
    assert result.final_tie_rate.item() == 0
    assert torch.equal(result.statistics.mean, torch.tensor([-1.0, -2.0]))


def test_comparison_rejects_missing_reference() -> None:
    curve = _curve("random", [[1.0, 0.0], [2.0, 1.0], [3.0, 2.0]])
    with pytest.raises(ValueError, match="Reference strategy"):
        compare_paired_strategies({"random": curve, "qei": curve}, reference="sobol")


def test_comparison_rejects_unpaired_seeds() -> None:
    baseline = _curve("random", [[1.0, 0.0], [2.0, 1.0], [3.0, 2.0]])
    candidate = _curve("qei", [[1.0, 0.0], [2.0, 1.0], [3.0, 2.0]], seeds=(0, 1, 9))
    with pytest.raises(ValueError, match="identical seed sets"):
        compare_paired_strategies({"random": baseline, "qei": candidate}, reference="random")
