"""Runtime representation contract for heterogeneous feasibility semantics."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TypeAlias

from robotorchan.acquisition.classification_constraints import (
    ClassificationProbabilityOfFeasibility,
)


class FeasibilityRepresentationKind(StrEnum):
    """Kinds of runtime feasibility representations."""

    SAMPLE_RESIDUAL = "sample_residual"
    PROBABILITY_OF_FEASIBILITY = "probability_of_feasibility"
    PROBABILITY_RESIDUAL = "probability_residual"


@dataclass(frozen=True, slots=True)
class SampleResidualFeasibility:
    """BoTorch-compatible sample residual where values <= 0 are feasible."""

    constraint: object
    kind: FeasibilityRepresentationKind = field(
        default=FeasibilityRepresentationKind.SAMPLE_RESIDUAL,
        init=False,
    )


@dataclass(frozen=True, slots=True)
class ProbabilityOfFeasibility:
    """Posterior-predictive probability of the feasible class."""

    probability: ClassificationProbabilityOfFeasibility
    kind: FeasibilityRepresentationKind = field(
        default=FeasibilityRepresentationKind.PROBABILITY_OF_FEASIBILITY,
        init=False,
    )


@dataclass(frozen=True, slots=True)
class ProbabilityResidualFeasibility:
    """Posterior-predictive probability residual where values <= 0 are feasible."""

    constraint: object
    kind: FeasibilityRepresentationKind = field(
        default=FeasibilityRepresentationKind.PROBABILITY_RESIDUAL,
        init=False,
    )


FeasibilityRepresentation: TypeAlias = (
    SampleResidualFeasibility | ProbabilityOfFeasibility | ProbabilityResidualFeasibility
)
