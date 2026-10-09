"""Named, immutable execution profiles for reproducible benchmark budgets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from robotorchan.benchmarks.config import BenchmarkExperimentConfig

ProfileName = Literal["smoke", "standard", "extended"]


@dataclass(frozen=True)
class BenchmarkExecutionProfile:
    """A named resource budget, independent of the benchmark problem and strategy."""

    name: ProfileName
    seeds: tuple[int, ...]
    initial_points: int
    evaluation_budget: int
    q: int
    dtype: Literal["float32", "float64"] = "float64"
    device: Literal["cpu", "cuda"] = "cpu"

    def __post_init__(self) -> None:
        # Reuse the canonical experiment validation for all profile fields.
        BenchmarkExperimentConfig(
            problem="_profile_validation",
            strategy="_profile_validation",
            seeds=self.seeds,
            initial_points=self.initial_points,
            evaluation_budget=self.evaluation_budget,
            q=self.q,
            dtype=self.dtype,
            device=self.device,
        )

    def make_config(self, problem: str, strategy: str) -> BenchmarkExperimentConfig:
        """Materialize a regular benchmark config without introducing runner coupling."""
        return BenchmarkExperimentConfig(
            problem=problem,
            strategy=strategy,
            seeds=self.seeds,
            initial_points=self.initial_points,
            evaluation_budget=self.evaluation_budget,
            q=self.q,
            dtype=self.dtype,
            device=self.device,
        )


_PROFILES: dict[ProfileName, BenchmarkExecutionProfile] = {
    "smoke": BenchmarkExecutionProfile(
        name="smoke", seeds=(0,), initial_points=4, evaluation_budget=4, q=1
    ),
    "standard": BenchmarkExecutionProfile(
        name="standard", seeds=(0, 1, 2), initial_points=8, evaluation_budget=40, q=1
    ),
    "extended": BenchmarkExecutionProfile(
        name="extended", seeds=(0, 1, 2, 3, 4), initial_points=16, evaluation_budget=100, q=4
    ),
}


def list_execution_profiles() -> tuple[str, ...]:
    """Return stable built-in profile names in alphabetical order."""
    return tuple(sorted(_PROFILES))


def get_execution_profile(
    name: ProfileName,
    *,
    dtype: Literal["float32", "float64"] | None = None,
    device: Literal["cpu", "cuda"] | None = None,
) -> BenchmarkExecutionProfile:
    """Return a fresh profile with optional explicit precision and device overrides."""
    try:
        profile = _PROFILES[name]
    except KeyError as error:
        raise ValueError(f"Unknown benchmark execution profile: {name}") from error
    resolved_dtype = profile.dtype if dtype is None else dtype
    resolved_device = profile.device if device is None else device
    if resolved_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA execution profile requested but CUDA is unavailable.")
    return BenchmarkExecutionProfile(
        name=profile.name,
        seeds=profile.seeds,
        initial_points=profile.initial_points,
        evaluation_budget=profile.evaluation_budget,
        q=profile.q,
        dtype=resolved_dtype,
        device=resolved_device,
    )
