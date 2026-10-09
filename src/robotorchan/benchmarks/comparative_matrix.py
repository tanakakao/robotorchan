"""Validated experiment settings for the comparative benchmark matrix."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from robotorchan.benchmarks.config import BenchmarkExperimentConfig

Tier = Literal["smoke", "standard", "full", "full_cuda"]

_PROFILES = {
    "smoke": (1, 4, 4, (1,), "cpu"),
    "standard": (5, 20, 40, (1, 3), "cpu"),
    "full": (20, 30, 100, (1, 3, 5), "cpu"),
    "full_cuda": (20, 30, 100, (1, 3, 5), "cuda"),
}
_METHODS = {
    "branin": ("random", "sobol", "qEI", "qNEI"),
    "hartmann6": ("random", "sobol", "qEI", "qNEI"),
    "branin_currin": ("random", "sobol", "qEHVI", "qNEHVI"),
    "strength_pass": ("random", "sobol", "qEI", "qNEI"),
    "strength_conductivity_pass": ("random", "sobol", "qEHVI", "qNEHVI"),
}
_BLOCKED = frozenset(
    {
        ("strength_pass", "qEI"),
        ("strength_pass", "qNEI"),
        ("strength_conductivity_pass", "qEHVI"),
        ("strength_conductivity_pass", "qNEHVI"),
    }
)


@dataclass(frozen=True)
class ComparativeExperimentCell:
    """One planned experiment, not proof that the strategy is ready."""

    tier: Tier
    config: BenchmarkExperimentConfig

    def __post_init__(self) -> None:
        if self.tier not in _PROFILES:
            raise ValueError(f"Unknown comparison tier: {self.tier}")
        count, initial, budget, batches, device = _PROFILES[self.tier]
        config = self.config
        if config.problem not in _METHODS:
            raise ValueError(f"Problem is not eligible: {config.problem}")
        if config.strategy not in _METHODS[config.problem]:
            raise ValueError("Strategy is not applicable to this problem.")
        if (config.problem, config.strategy) in _BLOCKED:
            raise ValueError("Classification label history is not yet supported.")
        if config.seeds != tuple(range(count)):
            raise ValueError("Comparison seeds must match the tier.")
        if config.initial_points != initial or config.evaluation_budget != budget:
            raise ValueError("Initial design and budget must match the tier.")
        if config.q not in batches:
            raise ValueError("Batch size is not allowed in this tier.")
        if config.dtype != "float64" or config.device != device:
            raise ValueError("Precision and device must match the tier.")

    def to_dict(self) -> dict[str, object]:
        """Serialize a validated comparison cell."""
        return {"tier": self.tier, "config": self.config.to_dict()}

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> ComparativeExperimentCell:
        """Validate a stored cell without accepting extra fields."""
        if not isinstance(data, dict) or set(data) != {"tier", "config"}:
            raise ValueError("Cell requires exactly tier and config.")
        if not isinstance(data["config"], dict):
            raise ValueError("config must be a dictionary.")
        return cls(
            tier=data["tier"],
            config=BenchmarkExperimentConfig.from_dict(data["config"]),
        )
