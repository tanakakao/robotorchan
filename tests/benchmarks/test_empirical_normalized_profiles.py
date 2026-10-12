"""Tests for strict within-stratum performance profile normalization."""

import pytest

from robotorchan.benchmarks.empirical_normalized_profiles import normalized_performance_profiles


def _records():
    return [
        {
            "problem": "branin",
            "strategy": strategy,
            "seed": seed,
            "budget": 50,
            "q": 1,
            "status": "success",
            "regret": float(seed + offset),
        }
        for seed in range(10)
        for strategy, offset in (("random", 2), ("sobol", 1))
    ]


def test_normalized_profile_direction_and_seed_alignment():
    rows = normalized_performance_profiles(_records(), direction="minimize", metric="regret")
    assert len(rows) == 20
    assert all(row["normalized_score"] == 1.0 for row in rows if row["strategy"] == "sobol")
    assert all(row["normalized_score"] == 0.0 for row in rows if row["strategy"] == "random")


def test_normalized_profile_preserves_failure():
    records = _records()
    records[0]["status"] = "failed"
    records[0]["regret"] = None
    rows = normalized_performance_profiles(records, direction="minimize", metric="regret")
    failed = [row for row in rows if row["status"] == "failed"]
    assert len(failed) == 1
    assert failed[0]["normalized_score"] is None


def test_normalized_profile_requires_complete_seed_coverage():
    with pytest.raises(ValueError, match="complete ten-seed"):
        normalized_performance_profiles(_records()[:-1], direction="minimize", metric="regret")
