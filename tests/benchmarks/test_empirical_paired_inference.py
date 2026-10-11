"""Tests for exact paired inference and multiplicity control."""

from __future__ import annotations

import pytest
import torch

from robotorchan.benchmarks.metric_curves import MetricCurves
from robotorchan.benchmarks.paired_inference import infer_paired_strategies


def _curve(values: list[float], seeds: tuple[int, ...] | None = None) -> MetricCurves:
    if seeds is None:
        seeds = tuple(range(len(values)))
    return MetricCurves(
        metric="simple_regret",
        seeds=seeds,
        evaluations=(0, 1),
        values=torch.tensor([[10.0, value] for value in values]),
    )


def test_exact_sign_test_and_holm_adjustment() -> None:
    baseline = _curve([5.0] * 8)
    better = _curve([4.0] * 8)
    worse = _curve([6.0] * 8)
    results = infer_paired_strategies(
        {"random": baseline, "better": better, "worse": worse},
        reference="random",
    )
    assert len(results) == 2
    assert all(result.p_value == 2 / 256 for result in results)
    assert all(result.adjusted_p_value == 4 / 256 for result in results)
    assert all(result.significant for result in results)
    assert results[0].mean_difference == -1.0
    assert results[0].standardized_mean_difference is None
    assert results[0].n_non_ties == 8


def test_all_ties_are_not_significant() -> None:
    baseline = _curve([5.0] * 8)
    result = infer_paired_strategies(
        {"random": baseline, "same": baseline},
        reference="random",
    )[0]
    assert result.p_value == 1
    assert result.n_non_ties == 0
    assert not result.significant


def test_unpaired_seeds_are_rejected() -> None:
    baseline = _curve([5.0, 5.0], seeds=(0, 1))
    candidate = _curve([4.0, 4.0], seeds=(0, 2))
    with pytest.raises(ValueError, match="identical seed sets"):
        infer_paired_strategies(
            {"random": baseline, "candidate": candidate},
            reference="random",
        )


def test_invalid_alpha_is_rejected() -> None:
    baseline = _curve([5.0, 5.0])
    with pytest.raises(ValueError, match="alpha"):
        infer_paired_strategies({"random": baseline, "candidate": baseline}, reference="random", alpha=1)
