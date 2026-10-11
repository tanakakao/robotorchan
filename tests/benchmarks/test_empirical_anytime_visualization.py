"""Tests for anytime curve extraction and optional plotting."""

from __future__ import annotations

from dataclasses import replace

import pytest

from robotorchan.benchmarks.anytime_visualization import (
    plot_anytime_performance,
    prepare_anytime_series,
    save_anytime_plot,
)
from robotorchan.benchmarks.comparison_report import EmpiricalComparisonReport


def _report() -> EmpiricalComparisonReport:
    stats = {
        "evaluations": [0, 1, 2],
        "mean": [4.0, 3.0, 2.0],
        "ci_lower": [3.0, 2.0, 1.0],
        "ci_upper": [5.0, 4.0, 3.0],
    }
    return EmpiricalComparisonReport(
        metric="simple_regret",
        reference="random",
        fairness=None,
        strategies={"sobol": dict(stats), "random": dict(stats)},
        comparisons=(),
        inference=(),
    )


def test_anytime_series_are_sorted_and_aligned() -> None:
    series = prepare_anytime_series(_report())
    assert [item.strategy for item in series] == ["random", "sobol"]
    assert series[0].evaluations == (0, 1, 2)
    assert series[0].mean == (4.0, 3.0, 2.0)


def test_anytime_rejects_misaligned_evaluations() -> None:
    report = _report()
    strategies = {key: dict(value) for key, value in report.strategies.items()}
    strategies["sobol"]["evaluations"] = [0, 1, 3]
    with pytest.raises(ValueError, match="share an evaluation axis"):
        prepare_anytime_series(replace(report, strategies=strategies))


def test_anytime_rejects_invalid_intervals() -> None:
    report = _report()
    strategies = {key: dict(value) for key, value in report.strategies.items()}
    strategies["sobol"]["ci_lower"] = [6.0, 2.0, 1.0]
    with pytest.raises(ValueError, match="bounds"):
        prepare_anytime_series(replace(report, strategies=strategies))


def test_anytime_plot_and_save(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    ax = plot_anytime_performance(_report())
    assert len(ax.lines) == 2
    import matplotlib.pyplot as plt

    plt.close(ax.figure)
    output = tmp_path / "anytime.png"
    save_anytime_plot(_report(), output)
    assert output.is_file()
    assert output.stat().st_size > 0
