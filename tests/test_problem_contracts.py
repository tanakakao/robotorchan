"""Tests for purpose-specific ProblemSpec contracts."""

import pytest

from robotorchan.problem import ProblemPurpose, ProblemSpec
from robotorchan.recommendation import recommend_compatible_workflows


def test_problem_spec_rejects_non_positive_q() -> None:
    with pytest.raises(ValueError, match="q must be at least 1"):
        ProblemSpec(purpose=ProblemPurpose.BAYESIAN_OPTIMIZATION, q=0)


def test_active_learning_rejects_bo_constraint_semantics() -> None:
    with pytest.raises(
        ValueError,
        match="constrained is only defined for Bayesian optimization",
    ):
        ProblemSpec(purpose=ProblemPurpose.ACTIVE_LEARNING, constrained=True)


def test_q_greater_than_one_excludes_q1_active_learning_acquisitions() -> None:
    spec = ProblemSpec(
        purpose=ProblemPurpose.ACTIVE_LEARNING,
        q=2,
    )

    assert recommend_compatible_workflows(spec) == ()
