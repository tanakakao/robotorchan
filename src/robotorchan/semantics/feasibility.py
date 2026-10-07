"""Runtime representation contract for heterogeneous feasibility semantics."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TypeAlias

from robotorchan.semantics.probability import ClassificationProbabilityOfFeasibility


class FeasibilityRepresentationKind(StrEnum):
    """Kinds of runtime feasibility representations.

    Each kind has distinct evaluation semantics. Callers must dispatch explicitly
    instead of coercing one representation into another.
    """

    SAMPLE_RESIDUAL = "sample_residual"
    PROBABILITY_OF_FEASIBILITY = "probability_of_feasibility"
    PROBABILITY_RESIDUAL = "probability_residual"


@dataclass(frozen=True, slots=True)
class SampleResidualFeasibility:
    """Sample-space constraint residual where values <= 0 are feasible.

    The payload consumes outcome samples. It is neither a Boolean feasibility
    indicator nor a posterior-predictive probability.
    """

    constraint: object
    kind: FeasibilityRepresentationKind = field(
        default=FeasibilityRepresentationKind.SAMPLE_RESIDUAL,
        init=False,
    )


@dataclass(frozen=True, slots=True)
class ProbabilityOfFeasibility:
    """Deterministic posterior-predictive P(feasible | X, D).

    The value is already marginalized over predictive uncertainty and therefore
    must not be treated as a sample-wise Monte Carlo feasibility value.
    """

    probability: ClassificationProbabilityOfFeasibility
    kind: FeasibilityRepresentationKind = field(
        default=FeasibilityRepresentationKind.PROBABILITY_OF_FEASIBILITY,
        init=False,
    )


@dataclass(frozen=True, slots=True)
class ProbabilityResidualFeasibility:
    """Thresholded posterior-predictive probability residual.

    Values <= 0 are feasible. This is a deterministic constraint on predictive
    probability, not the probability of satisfying another outcome constraint.
    """

    constraint: object
    kind: FeasibilityRepresentationKind = field(
        default=FeasibilityRepresentationKind.PROBABILITY_RESIDUAL,
        init=False,
    )


FeasibilityRepresentation: TypeAlias = (
    SampleResidualFeasibility | ProbabilityOfFeasibility | ProbabilityResidualFeasibility
)
