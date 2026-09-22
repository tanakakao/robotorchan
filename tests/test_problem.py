"""Tests for the declarative problem specification."""

import pytest

from robotorchan.models.capabilities import InputType, TaskType
from robotorchan.problem import (
    ObjectiveType,
    OutputType,
    ProblemPurpose,
    ProblemSpec,
)


def test_bo_defaults_to_single_objective_and_single_output() -> None:
    spec = ProblemSpec(purpose=ProblemPurpose.BAYESIAN_OPTIMIZATION)

    assert spec.input_type is InputType.CONTINUOUS
    assert spec.task_type is TaskType.SINGLE
    assert spec.output_type is OutputType.SINGLE
    assert spec.objective_type is ObjectiveType.SINGLE


def test_bo_preserves_multi_objective_and_multi_output_separately() -> None:
    spec = ProblemSpec(
        purpose=ProblemPurpose.BAYESIAN_OPTIMIZATION,
        input_type=InputType.MIXED,
        task_type=TaskType.MULTITASK,
        output_type=OutputType.MULTI,
        objective_type=ObjectiveType.MULTI,
        multi_fidelity=True,
        structured_output=True,
        high_dimensional=True,
        robust=True,
    )

    assert spec.output_type is OutputType.MULTI
    assert spec.objective_type is ObjectiveType.MULTI
    assert spec.multi_fidelity
    assert spec.structured_output
    assert spec.high_dimensional
    assert spec.robust


def test_active_learning_can_be_multi_output() -> None:
    spec = ProblemSpec(
        purpose=ProblemPurpose.ACTIVE_LEARNING,
        output_type=OutputType.MULTI,
    )

    assert spec.output_type is OutputType.MULTI
    assert spec.objective_type is None


def test_active_learning_rejects_optimization_objective_semantics() -> None:
    with pytest.raises(ValueError, match="only defined for Bayesian optimization"):
        ProblemSpec(
            purpose=ProblemPurpose.ACTIVE_LEARNING,
            objective_type=ObjectiveType.MULTI,
        )
