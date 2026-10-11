"""Explicit observation-noise contracts for paired empirical benchmarks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from robotorchan.benchmarks.random_streams import make_stream_generator

NoisePolicy = Literal["independent", "paired"]
ObservationNoise = Callable[[Tensor, torch.Generator], Tensor]


@dataclass(frozen=True)
class SeededObservationNoise:
    """Generate observation noise from a named run-local stream.

    Paired noise shares the same seed across strategies for equal batch schedules.
    Independent noise derives a separate seed for each strategy.
    """

    noise: ObservationNoise
    policy: NoisePolicy = "paired"

    def __post_init__(self) -> None:
        if not callable(self.noise):
            raise TypeError("noise must be callable.")
        if self.policy not in ("independent", "paired"):
            raise ValueError("Unknown observation-noise policy.")

    def generator(self, seed: int, strategy: str, device: torch.device) -> torch.Generator:
        """Create a fresh noise stream independent of candidate generation."""
        stream = "observation" if self.policy == "paired" else f"observation:{strategy}"
        return make_stream_generator(seed, stream, device)

    def observe(self, truth: Tensor, generator: torch.Generator) -> Tensor:
        """Apply and validate a shape-preserving finite observation perturbation."""
        noise = self.noise(truth, generator)
        if not isinstance(noise, Tensor) or noise.shape != truth.shape:
            raise ValueError("Observation noise must return a tensor matching truth shape.")
        if noise.device != truth.device or noise.dtype != truth.dtype:
            raise ValueError("Observation noise must preserve device and dtype.")
        observed = truth + noise
        if not torch.isfinite(observed).all():
            raise ValueError("Noisy observations must be finite.")
        return observed
