"""Shared q-batch and fixed-feature operations for optimizer backends."""

from __future__ import annotations

from collections.abc import Callable

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from torch import Tensor

CandidateOptimizer = Callable[..., tuple[Tensor, Tensor]]


def apply_fixed_features(
    candidates: Tensor,
    fixed_features: dict[int, float | Tensor] | None,
) -> Tensor:
    """Return candidates with fixed feature columns overwritten."""
    if not fixed_features:
        return candidates
    result = candidates.clone()
    d = result.shape[-1]
    for index, value in fixed_features.items():
        resolved = index if index >= 0 else d + index
        if resolved < 0 or resolved >= d:
            raise ValueError(f"fixed feature index {index} is outside input dimension {d}.")
        result[..., resolved] = torch.as_tensor(value, dtype=result.dtype, device=result.device)
    return result


def optimize_acqf_sequential(
    optimizer: CandidateOptimizer,
    acq_function: AcquisitionFunction,
    bounds: Tensor,
    q: int,
    *,
    optimizer_kwargs: dict[str, object] | None = None,
) -> tuple[Tensor, Tensor]:
    """Sequentially generate q points for acquisitions supporting X_pending.

    The helper mirrors BoTorch's sequential concept without pretending that
    every backend natively implements it. Existing pending points are restored.
    """
    if q < 1:
        raise ValueError("q must be at least 1.")
    if not hasattr(acq_function, "set_X_pending"):
        raise ValueError("Sequential optimization requires an acquisition with set_X_pending.")

    kwargs = dict(optimizer_kwargs or {})
    original_pending = getattr(acq_function, "X_pending", None)
    selected: list[Tensor] = []
    values: list[Tensor] = []
    try:
        for _ in range(q):
            candidate, value = optimizer(acq_function, bounds, 1, **kwargs)
            selected.append(candidate)
            values.append(value.reshape(()))
            new_pending = torch.cat(selected, dim=0)
            if original_pending is not None:
                new_pending = torch.cat([original_pending, new_pending], dim=-2)
            acq_function.set_X_pending(new_pending)
    finally:
        acq_function.set_X_pending(original_pending)
    return torch.cat(selected, dim=0), torch.stack(values)
