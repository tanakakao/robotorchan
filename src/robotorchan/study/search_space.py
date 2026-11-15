"""Named continuous search-space definition for Study core."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True, slots=True)
class SearchSpace:
    """Fixed named continuous bounds; mixed semantics arrive in Phase 7."""

    bounds: Mapping[str, tuple[float, float]]

    def __post_init__(self) -> None:
        normalized = {}
        for name, pair in self.bounds.items():
            if not isinstance(name, str) or not name:
                raise ValueError("Parameter names must be nonempty strings.")
            if len(pair) != 2:
                raise ValueError(f"Bounds for {name!r} must contain two values.")
            lower, upper = pair
            if not all(
                isinstance(v, (int, float)) and not isinstance(v, bool) and isfinite(v)
                for v in (lower, upper)
            ):
                raise ValueError(f"Bounds for {name!r} must be finite numbers.")
            if lower >= upper:
                raise ValueError(f"Lower bound must be smaller than upper for {name!r}.")
            normalized[name] = (float(lower), float(upper))
        if not normalized:
            raise ValueError("Search space must contain at least one parameter.")
        object.__setattr__(self, "bounds", MappingProxyType(normalized))

    def validate(self, params: Mapping[str, float]) -> None:
        """Validate a complete candidate against declared bounds."""
        if set(params) != set(self.bounds):
            raise ValueError("Trial parameters must match the search-space names.")
        for name, value in params.items():
            lower, upper = self.bounds[name]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
                or not lower <= value <= upper
            ):
                raise ValueError(f"Parameter {name!r} is outside its finite bounds.")
