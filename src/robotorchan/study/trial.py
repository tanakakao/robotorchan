"""Trial records and lifecycle states for experiment tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from math import isfinite
from types import MappingProxyType
from typing import Mapping


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

    def __post_init__(self) -> None:
        if self.number < 0:
            raise ValueError("Trial number must be non-negative.")
        params = dict(self.params)
        if any(not isinstance(name, str) or not name for name in params):
            raise ValueError("Parameter names must be nonempty strings.")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(value)
            for value in params.values()
        ):
            raise ValueError("Parameter values must be finite numbers.")
        object.__setattr__(self, "params", MappingProxyType(params))
        if self.state is TrialState.COMPLETE:
            if self.value is None or not isfinite(self.value):
                raise ValueError("Completed trials require a finite observation.")
        elif self.value is not None:
            raise ValueError("Only completed trials may contain an observation.")
