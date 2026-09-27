"""Runtime validation helpers shared by acquisition optimizer backends."""

from __future__ import annotations

import torch
from torch import Tensor


def validate_bounds(bounds: Tensor) -> None:
    """Validate the common continuous bounds contract."""
    if bounds.ndim != 2 or bounds.shape[0] != 2:
        raise ValueError("bounds must have shape [2, d].")
    if not bounds.is_floating_point():
        raise TypeError("bounds must use a floating-point dtype.")
    if not torch.isfinite(bounds).all():
        raise ValueError("bounds must contain only finite values.")
    if torch.any(bounds[0] >= bounds[1]):
        raise ValueError("Every lower bound must be strictly smaller than its upper bound.")


def make_generator(bounds: Tensor, seed: int | None) -> torch.Generator:
    """Create a backend-local RNG without mutating PyTorch's global RNG state."""
    generator = torch.Generator(device=bounds.device)
    if seed is not None:
        generator.manual_seed(seed)
    return generator
