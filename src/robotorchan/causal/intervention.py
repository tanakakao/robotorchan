"""Hard interventions for known structural causal models."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType


@dataclass(frozen=True)
class HardIntervention:
    """A do-intervention replacing structural equations for selected nodes."""

    values: Mapping[str, float]

    def __post_init__(self) -> None:
        if not self.values:
            raise ValueError("An intervention must target at least one node.")
        copied: dict[str, float] = {}
        for node, value in self.values.items():
            if not isinstance(node, str) or not node:
                raise ValueError("Intervention node names must be nonempty strings.")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError("Intervention values must be real scalars.")
            number = float(value)
            if not isfinite(number):
                raise ValueError("Intervention values must be finite.")
            copied[node] = number
        object.__setattr__(self, "values", MappingProxyType(copied))
