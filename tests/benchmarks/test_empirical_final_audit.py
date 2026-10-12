"""Tests for final empirical benchmark completion assessment."""

import pytest

from robotorchan.benchmarks.empirical_final_audit import assess_final_empirical_goals

_GOALS = (
    "optimization_performance",
    "algorithm_comparison",
    "problem_adaptability",
    "efficiency_and_stability",
    "reproducibility",
)


def test_final_audit_fails_closed_without_measured_evidence():
    result = assess_final_empirical_goals({})
    assert not result["complete"]
    assert result["unmet_goals"] == list(_GOALS)


def test_final_audit_requires_all_five_verified_goals():
    evidence = {
        goal: {
            "artifact": f"results/{goal}.json",
            "measured": True,
            "verified": True,
            "reproduced": True,
        }
        for goal in _GOALS
    }
    result = assess_final_empirical_goals(evidence)
    assert result["complete"]
    evidence["reproducibility"]["reproduced"] = False
    result = assess_final_empirical_goals(evidence)
    assert not result["complete"]
    assert result["unmet_goals"] == ["reproducibility"]


def test_final_audit_rejects_unknown_goal():
    with pytest.raises(ValueError, match="Unknown"):
        assess_final_empirical_goals({"unplanned_goal": {}})
