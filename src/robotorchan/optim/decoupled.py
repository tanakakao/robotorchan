"""Enumerate evaluation masks and delegate candidate optimization to BoTorch."""

from typing import Any

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from torch import Tensor

from robotorchan.optim.dispatch import optimize_acqf


def optimize_acqf_decoupled(
    acq_function: AcquisitionFunction,
    bounds: Tensor,
    evaluation_masks: Tensor,
    q: int = 1,
    *,
    num_restarts: int = 10,
    raw_samples: int = 256,
    options: dict[str, Any] | None = None,
    batch_initial_conditions: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return candidate, selected evaluation mask, and acquisition value.

    Phase 4 enumerates allowed masks for q=1 in continuous domains. One-shot
    auxiliary candidates are managed by BoTorch, never included in the mask.
    The original acquisition mask is restored even if optimization fails.
    """
    if q != 1:
        raise NotImplementedError("decoupled mask enumeration currently requires q=1")
    if not hasattr(acq_function, "X_evaluation_mask"):
        raise TypeError("acquisition must support X_evaluation_mask")
    if (
        evaluation_masks.ndim != 2
        or evaluation_masks.shape[0] == 0
        or evaluation_masks.shape[1] != acq_function.model.num_outputs
        or evaluation_masks.dtype != torch.bool
        or not evaluation_masks.any(dim=-1).all()
    ):
        raise ValueError("evaluation_masks must contain nonempty boolean output masks")
    if evaluation_masks.device != bounds.device:
        raise ValueError("evaluation_masks and bounds must share a device")
    if getattr(acq_function, "X_pending", None) is not None:
        raise NotImplementedError("pending evaluations are not supported")
    original_mask = acq_function.X_evaluation_mask
    best = None
    try:
        for mask in evaluation_masks:
            selected = mask.unsqueeze(0).clone()
            acq_function.X_evaluation_mask = selected
            candidate, value = optimize_acqf(
                acq_function=acq_function,
                bounds=bounds,
                q=q,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
                options=options,
                batch_initial_conditions=batch_initial_conditions,
            )
            if candidate.shape != (q, bounds.shape[-1]) or value.numel() != 1:
                raise RuntimeError("optimizer returned unexpected candidate or value shape")
            if not torch.isfinite(candidate).all() or not torch.isfinite(value).all():
                raise RuntimeError("optimizer returned nonfinite candidate or value")
            if best is None or value.item() > best[2].item():
                best = (candidate.detach().clone(), selected, value.detach().clone())
    finally:
        acq_function.X_evaluation_mask = original_mask
    return best
