"""Test status-aware failure rates and conditional stability summaries."""

import pytest

from robotorchan.benchmarks.empirical_stability_evidence import summarize_seed_stability


def test_stability_preserves_failed_and_not_run_seeds():
    records = [
        {
            "seed": seed,
            "status": "success" if seed < 7 else ("failed" if seed < 9 else "not_run"),
            "regret": float(seed) if seed < 7 else None,
        }
        for seed in range(10)
    ]
    result = summarize_seed_stability(records, metric="regret", direction="minimize")
    assert result["success_count"] == 7
    assert result["failed_count"] == 2
    assert result["not_run_count"] == 1
    assert result["observed_failure_rate"] == pytest.approx(2 / 9)
    assert result["completion_rate"] == pytest.approx(0.9)
    assert result["successful_median"] == 3.0


def test_stability_rejects_missing_seed():
    records = [{"seed": seed, "status": "not_run", "regret": None} for seed in range(9)]
    with pytest.raises(ValueError, match="All ten"):
        summarize_seed_stability(records, metric="regret", direction="minimize")


def test_stability_does_not_impute_failed_performance():
    records = [{"seed": seed, "status": "failed", "regret": None} for seed in range(10)]
    result = summarize_seed_stability(records, metric="regret", direction="minimize")
    assert result["successful_mean"] is None
    assert result["observed_failure_rate"] == 1.0
