"""Semantic contracts for ProblemSpec objective and output arity."""

import pytest

from robotorchan.problem import ObjectiveType, OutputType, ProblemPurpose, ProblemSpec


def test_multi_objective_bo_requires_multi_output_observations() -> None:
    with pytest.raises(
        ValueError,
        match="multi-objective optimization requires multi-output observations",
    ):
        ProblemSpec(
            purpose=ProblemPurpose.BAYESIAN_OPTIMIZATION,
            objective_type=ObjectiveType.MULTI,
        )


def test_multi_objective_multi_output_bo_is_valid() -> None:
    spec = ProblemSpec(
        purpose=ProblemPurpose.BAYESIAN_OPTIMIZATION,
        objective_type=ObjectiveType.MULTI,
        output_type=OutputType.MULTI,
    )

    assert spec.objective_type is ObjectiveType.MULTI
    assert spec.output_type is OutputType.MULTI
