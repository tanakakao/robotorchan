"""Tests for deterministic paired bootstrap comparisons."""

import pytest

from robotorchan.benchmarks.empirical_paired_statistics import paired_bootstrap_comparison


def _records(offset=0.0, failed=()):
    return [
        {
            "seed": seed,
            "status": "failed" if seed in failed else "success",
            "regret": None if seed in failed else float(seed) + offset,
        }
        for seed in range(10)
    ]


def test_paired_bootstrap_improvement_and_reproducibility():
    first = paired_bootstrap_comparison(
        _records(), _records(1.0), metric="regret", direction="minimize"
    )
    second = paired_bootstrap_comparison(
        _records(), _records(1.0), metric="regret", direction="minimize"
    )
    assert first == second
    assert first["paired_success_count"] == 10
    assert first["mean_improvement"] == pytest.approx(-1.0)
    assert first["ci95_lower"] == pytest.approx(-1.0)
    assert first["ci95_upper"] == pytest.approx(-1.0)


def test_paired_bootstrap_keeps_excluded_failure_pairs():
    result = paired_bootstrap_comparison(
        _records(failed=(0,)), _records(failed=(1,)), metric="regret", direction="minimize"
    )
    assert result["paired_success_count"] == 8
    assert [item["seed"] for item in result["excluded_pairs"]] == [0, 1]


def test_paired_bootstrap_rejects_missing_seeds():
    with pytest.raises(ValueError, match="all ten"):
        paired_bootstrap_comparison(
            _records()[:-1], _records(), metric="regret", direction="minimize"
        )
