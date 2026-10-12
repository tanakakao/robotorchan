"""Tests for seed-complete empirical rerun verification."""

import pytest

from robotorchan.benchmarks.empirical_rerun_verification import compare_empirical_reruns


def _records(offset=0.0):
    return [
        {"seed": seed, "status": "success", "regret": seed + 1.0 + offset} for seed in range(10)
    ]


def test_identical_rerun_matches_all_seeds():
    report = compare_empirical_reruns(_records(), _records(), metric="regret")
    assert report["all_successful_seeds_reproduced"]
    assert report["numerical_matches"] == 10


def test_rerun_reports_numerical_mismatch():
    report = compare_empirical_reruns(_records(), _records(0.01), metric="regret")
    assert report["numerical_mismatches"] == 10
    assert not report["all_successful_seeds_reproduced"]


def test_rerun_reports_status_mismatch():
    rerun = _records()
    rerun[0]["status"] = "failed"
    rerun[0]["regret"] = None
    report = compare_empirical_reruns(_records(), rerun, metric="regret")
    assert report["status_mismatches"] == 1
    assert report["successful_pairs"] == 9


def test_rerun_rejects_missing_seed():
    with pytest.raises(ValueError, match="all ten"):
        compare_empirical_reruns(_records()[:-1], _records(), metric="regret")
