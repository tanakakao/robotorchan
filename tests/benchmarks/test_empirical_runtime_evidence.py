"""Validate measured runtime breakdown and reject invalid timing data."""

from types import SimpleNamespace

import pytest

from robotorchan.benchmarks.empirical_runtime_evidence import runtime_breakdown


def test_runtime_breakdown_separates_candidate_and_evaluation():
    run = SimpleNamespace(
        candidate_seconds=(0.2, 0.3),
        evaluation_seconds=(0.1, 0.4),
        initial_evaluation_seconds=0.5,
        completed_batches=(1, 2),
        evaluation_count=2,
    )
    result = runtime_breakdown(run)
    assert result["candidate_seconds"] == pytest.approx(0.5)
    assert result["evaluation_seconds"] == pytest.approx(0.5)
    assert result["initial_evaluation_seconds"] == pytest.approx(0.5)
    assert result["total_measured_seconds"] == pytest.approx(1.5)


def test_runtime_breakdown_rejects_invalid_times():
    run = SimpleNamespace(
        candidate_seconds=(float("nan"),),
        evaluation_seconds=(0.1,),
        initial_evaluation_seconds=0.5,
        completed_batches=(1,),
    )
    with pytest.raises(ValueError, match="Invalid"):
        runtime_breakdown(run)
