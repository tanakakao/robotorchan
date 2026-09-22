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


class OutputType(StrEnum):
    SINGLE = "single"
    MULTI = "multi"


@dataclass(frozen=True, slots=True)
class ProblemSpec:
    """Describe problem requirements without selecting an implementation."""

    purpose: ProblemPurpose
    input_type: InputType = InputType.CONTINUOUS
    task_type: TaskType = TaskType.SINGLE
    output_type: OutputType = OutputType.SINGLE
    objective_type: ObjectiveType | None = None
    multi_fidelity: bool = False
    structured_output: bool = False
    high_dimensional: bool = False
    robust: bool = False
    preference: bool = False
    q: int = 1
    constrained: bool = False

    def __post_init__(self) -> None:
        """Validate purpose-specific fields."""
        if self.q < 1:
            raise ValueError("q must be at least 1")
        if self.purpose is ProblemPurpose.BAYESIAN_OPTIMIZATION:
            if self.objective_type is None:
                object.__setattr__(self, "objective_type", ObjectiveType.SINGLE)
            if (
                self.objective_type is ObjectiveType.MULTI
                and self.output_type is not OutputType.MULTI
            ):
                raise ValueError(
                    "multi-objective optimization requires multi-output observations"
                )
            return
        if self.objective_type is not None:
            raise ValueError("objective_type is only defined for Bayesian optimization")
        if self.constrained:
            raise ValueError("constrained is only defined for Bayesian optimization")
