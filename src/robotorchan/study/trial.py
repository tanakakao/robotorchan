"""Trial records and lifecycle states for experiment tracking."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from math import isfinite
from types import MappingProxyType


class TrialState(StrEnum):
    """Lifecycle states of an experiment trial."""

    WAITING = "WAITING"
    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"
    FAIL = "FAIL"


@dataclass(frozen=True, slots=True)
class Trial:
    """Immutable snapshot of one trial; updates are managed by Study."""

    number: int
    params: Mapping[str, float]
    state: TrialState = TrialState.WAITING
    value: float | None = None
    evaluation_mask: tuple[str, ...] = ()
    observations: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.number < 0:
            raise ValueError("Trial number must be non-negative.")
        params = dict(self.params)
        if any(not isinstance(name, str) or not name for name in params):
            raise ValueError("Parameter names must be nonempty strings.")
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value)
            for value in params.values()
        ):
            raise ValueError("Parameter values must be finite numbers.")
        object.__setattr__(self, "params", MappingProxyType(params))
        observations = dict(self.observations)
        mask = tuple(self.evaluation_mask)
        if len(set(mask)) != len(mask) or any(not isinstance(n, str) or not n for n in mask):
            raise ValueError("Evaluation outputs must be unique nonempty names.")
        if not set(observations).issubset(mask) or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not isfinite(v)
            for v in observations.values()
        ):
            raise ValueError("Observations must be finite numeric values for selected outputs.")
        object.__setattr__(self, "evaluation_mask", mask)
        object.__setattr__(self, "observations", MappingProxyType(observations))
        if mask:
            if self.value is not None:
                raise ValueError("Decoupled trials cannot contain a scalar value.")
            if self.state is TrialState.WAITING and observations:
                raise ValueError("Waiting trials cannot contain observations.")
            if self.state is TrialState.COMPLETE and set(observations) != set(mask):
                raise ValueError("Completed trials require every selected output.")
            if self.state is TrialState.RUNNING and set(observations) == set(mask):
                raise ValueError("Fully observed trials must be completed.")
            return
        if self.state is TrialState.COMPLETE:
            if self.value is None or not isfinite(self.value):
                raise ValueError("Completed trials require a finite observation.")
        elif self.value is not None:
            raise ValueError("Only completed trials may contain an observation.")
