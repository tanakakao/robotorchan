"""Aggregate cross-strategy fairness validation and structured reporting."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from robotorchan.benchmarks.budget_fairness import (
    BudgetFairnessResult,
    validate_budget_fairness,
)
from robotorchan.benchmarks.config import BenchmarkExperimentConfig
from robotorchan.benchmarks.initial_pairing import (
    InitialPairingResult,
    validate_initial_pairing,
)
from robotorchan.benchmarks.runner import BenchmarkTrajectory

BenchmarkRuns = tuple[BenchmarkExperimentConfig, tuple[BenchmarkTrajectory, ...]]


@dataclass(frozen=True)
class FairnessReport:
    """Validated cross-strategy comparison conditions."""

    strategies: tuple[str, ...]
    initial: InitialPairingResult
    budget: BudgetFairnessResult
    observation_pairing_required: bool
    cost_pairing_required: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable fairness report."""
        return asdict(self)


def build_fairness_report(
    experiments: Mapping[str, BenchmarkRuns],
    *,
    require_observation_pairing: bool = True,
    require_cost_pairing: bool = False,
) -> FairnessReport:
    """Validate initial conditions and evaluation schedules before comparison.

    Raises ValueError when any required fairness condition is violated.
    Observation pairing may be relaxed for independently seeded observation
    noise; cost pairing may be relaxed for candidate-dependent costs.
    """
    initial = validate_initial_pairing(
        {name: runs for name, (_, runs) in experiments.items()},
        require_observation_pairing=require_observation_pairing,
    )
    budget = validate_budget_fairness(
        experiments,
        require_cost_pairing=require_cost_pairing,
    )
    return FairnessReport(
        strategies=tuple(experiments),
        initial=initial,
        budget=budget,
        observation_pairing_required=require_observation_pairing,
        cost_pairing_required=require_cost_pairing,
    )
