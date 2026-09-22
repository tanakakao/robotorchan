"""Declarative problem specification for capability-aware workflows."""

from dataclasses import dataclass
from enum import StrEnum

from robotorchan.models.capabilities import InputType, TaskType


class ProblemPurpose(StrEnum):
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    ACTIVE_LEARNING = "active_learning"


class ObjectiveType(StrEnum):
    SINGLE = "single"
    MULTI = "multi"


@dataclass(frozen=True, slots=True)
class ProblemSpec:
    """Describe problem requirements without selecting an implementation."""

    purpose: ProblemPurpose
    input_type: InputType = InputType.CONTINUOUS
    task_type: TaskType = TaskType.SINGLE
    objective_type: ObjectiveType = ObjectiveType.SINGLE
    multi_fidelity: bool = False
    structured_output: bool = False
    high_dimensional: bool = False
    robust: bool = False

    def __post_init__(self) -> None:
        if (
            self.purpose is ProblemPurpose.ACTIVE_LEARNING
            and self.objective_type is ObjectiveType.MULTI
        ):
            raise ValueError(
                "multi-objective optimization is not an active-learning requirement"
            )
