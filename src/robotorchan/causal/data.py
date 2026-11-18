"""Typed records separating observational and interventional data."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType

from robotorchan.causal.intervention import HardIntervention


@dataclass(frozen=True)
class CausalObservation:
    """One observational or hard-interventional record."""

    values: Mapping[str, float]
    intervention: HardIntervention | None = None
    _validated: bool = field(default=False, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.values:
            raise ValueError("Observed values must not be empty.")
        copied = dict(self.values)
        for name, value in copied.items():
            if not isinstance(name, str) or not name:
                raise ValueError("Observed variable names must be nonempty.")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError("Observed values must be real scalars.")
            if not isfinite(float(value)):
                raise ValueError("Observed values must be finite.")
        if self.intervention is not None:
            for name, value in self.intervention.values.items():
                if name not in copied or float(copied[name]) != float(value):
                    raise ValueError("Observed values must match the intervention.")
        object.__setattr__(self, "values", MappingProxyType(copied))

    @property
    def is_interventional(self) -> bool:
        return self.intervention is not None
