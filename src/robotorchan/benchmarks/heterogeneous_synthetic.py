"""Fixed synthetic heterogeneous BO benchmark with a known probabilistic constraint."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class SyntheticObservation:
    """Noisy regression observations and a Bernoulli feasibility label."""

    strength: Tensor
    conductivity: Tensor
    passed: Tensor


def evaluate_truth(X: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Return strength, conductivity, and true Pass probability for X in [0, 1]^3."""
    if X.ndim < 2 or X.shape[-1] != 3:
        raise ValueError("X must have shape (..., n, 3).")
    if not X.is_floating_point() or not torch.isfinite(X).all():
        raise ValueError("X must be finite and floating-point.")
    if ((X < 0) | (X > 1)).any():
        raise ValueError("X must lie inside the unit cube.")

    temperature, pressure, composition = X.unbind(dim=-1)
    strength = (
        1.3
        - 2.0 * (temperature - 0.78).square()
        - 1.5 * (pressure - 0.28).square()
        - 0.7 * (composition - 0.68).square()
    )
    conductivity = (
        1.2
        - 1.8 * (temperature - 0.22).square()
        - 1.6 * (pressure - 0.76).square()
        - 0.9 * (composition - 0.35).square()
    )
    score = (
        9.0
        * (
            0.52
            - (temperature - 0.50).square()
            - 1.2 * (pressure - 0.53).square()
            - 0.7 * (composition - 0.50).square()
        )
        - 2.5
    )
    probability = torch.sigmoid(score)
    return strength, conductivity, probability


def observe(
    X: Tensor,
    *,
    seed: int,
    noise_std: float = 0.02,
) -> SyntheticObservation:
    """Sample independent regression noise and Bernoulli labels reproducibly."""
    if not isinstance(seed, int):
        raise TypeError("seed must be an integer.")
    if not 0.0 <= noise_std < float("inf"):
        raise ValueError("noise_std must be finite and nonnegative.")
    strength, conductivity, probability = evaluate_truth(X)
    generator = torch.Generator(device=X.device).manual_seed(seed)
    noise_1 = torch.randn(strength.shape, dtype=X.dtype, device=X.device, generator=generator)
    noise_2 = torch.randn(
        conductivity.shape, dtype=X.dtype, device=X.device, generator=generator
    )
    uniform = torch.rand(probability.shape, dtype=X.dtype, device=X.device, generator=generator)
    return SyntheticObservation(
        strength=strength + noise_std * noise_1,
        conductivity=conductivity + noise_std * noise_2,
        passed=(uniform < probability).to(dtype=torch.long),
    )


def reference_front(
    *,
    grid_size: int = 25,
    probability_threshold: float = 0.5,
    dtype: torch.dtype = torch.double,
) -> tuple[Tensor, Tensor]:
    """Approximate the feasible Pareto front on a deterministic Cartesian grid.

    Feasibility is defined by true probability >= threshold, not sampled labels.
    Returns feasible input points and nondominated objective pairs.
    """
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2.")
    if not 0.0 <= probability_threshold <= 1.0:
        raise ValueError("probability_threshold must be in [0, 1].")
    axis = torch.linspace(0.0, 1.0, grid_size, dtype=dtype)
    X = torch.cartesian_prod(axis, axis, axis)
    strength, conductivity, probability = evaluate_truth(X)
    feasible = probability >= probability_threshold
    X = X[feasible]
    Y = torch.stack((strength[feasible], conductivity[feasible]), dim=-1)
    if Y.numel() == 0:
        return X, Y
    from botorch.utils.multi_objective.pareto import is_non_dominated

    keep = is_non_dominated(Y)
    return X[keep], Y[keep]
