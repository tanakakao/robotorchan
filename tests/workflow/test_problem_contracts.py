"""Tests for purpose-specific ProblemSpec contracts."""

import pytest

from robotorchan.workflow.problem import ProblemPurpose, ProblemSpec
from robotorchan.workflow.recommendation import recommend_compatible_workflows


def test_problem_spec_rejects_non_positive_q() -> None:
    with pytest.raises(ValueError, match="q must be at least 1"):
        ProblemSpec(purpose=ProblemPurpose.BAYESIAN_OPTIMIZATION, q=0)


def test_active_learning_rejects_bo_constraint_semantics() -> None:
    with pytest.raises(
        ValueError,
        match="constrained is only defined for Bayesian optimization",
    ):
        ProblemSpec(purpose=ProblemPurpose.ACTIVE_LEARNING, constrained=True)


def test_q_greater_than_one_includes_only_batch_capable_active_learning() -> None:
    spec = ProblemSpec(
        purpose=ProblemPurpose.ACTIVE_LEARNING,
        q=2,
    )

    recommendations = recommend_compatible_workflows(spec)
    assert recommendations
    assert {item.acquisition_name for item in recommendations} == {
        "qBoundaryVariance",
        "qJointPosteriorEntropy",
        "qPosteriorStd",
        "qPosteriorVariance",
        "qRandomizedStraddle",
        "qStraddle",
    }
