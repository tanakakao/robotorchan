"""Tests for the declarative problem specification."""

from robotorchan.models.capabilities import InputType, TaskType
from robotorchan.problem import ObjectiveType, ProblemPurpose, ProblemSpec


def test_problem_spec_defaults_to_continuous_single_task() -> None:
    spec = ProblemSpec(purpose=ProblemPurpose.BAYESIAN_OPTIMIZATION)

    assert spec.input_type is InputType.CONTINUOUS
    assert spec.task_type is TaskType.SINGLE
    assert spec.objective_type is ObjectiveType.SINGLE
    assert not spec.multi_fidelity
    assert not spec.structured_output
    assert not spec.high_dimensional
    assert not spec.robust


def test_problem_spec_preserves_explicit_requirements() -> None:
    spec = ProblemSpec(
        purpose=ProblemPurpose.BAYESIAN_OPTIMIZATION,
        input_type=InputType.MIXED,
        task_type=TaskType.MULTITASK,
        objective_type=ObjectiveType.MULTI,
        multi_fidelity=True,
        structured_output=True,
        high_dimensional=True,
        robust=True,
    )

    assert spec.input_type is InputType.MIXED
    assert spec.task_type is TaskType.MULTITASK
    assert spec.objective_type is ObjectiveType.MULTI
    assert spec.multi_fidelity
    assert spec.structured_output
    assert spec.high_dimensional
    assert spec.robust



def test_active_learning_can_describe_multiple_outputs() -> None:
    spec = ProblemSpec(
        purpose=ProblemPurpose.ACTIVE_LEARNING,
        objective_type=ObjectiveType.MULTI,
    )

    assert spec.objective_type is ObjectiveType.MULTI
