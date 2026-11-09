"""Conservative safe-set expansion from a finite continuous candidate pool."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from robotorchan.semantics.constraints import ContinuousConstraint
from robotorchan.semantics.safety_bounds import safety_confidence_bounds


def propose_safe_expansion(
    safety_model: object,
    bounds: Tensor,
    constraints: Sequence[ContinuousConstraint],
    *,
    beta: float = 4.0,
    tolerance: float = 0.0,
    num_samples: int = 1024,
    seed: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Choose a conservative safe point with large safety uncertainty.

    Among pessimistically safe points, prioritize those with the greatest
    confidence-width sum. This encourages learning near uncertain safe regions
    but does not establish SafeOpt reachability or a formal safety guarantee.
    """
    if bounds.ndim != 2 or bounds.shape[0] != 2 or bounds.shape[1] < 1:
        raise ValueError("bounds must have shape [2, d].")
    if not bounds.is_floating_point() or not bool(torch.isfinite(bounds).all()):
        raise ValueError("bounds must be finite floating tensors.")
    if not bool((bounds[1] > bounds[0]).all()):
        raise ValueError("bounds must have strictly increasing limits.")
    if isinstance(num_samples, bool) or not isinstance(num_samples, int) or num_samples < 1:
        raise ValueError("num_samples must be a positive integer.")

    engine = torch.quasirandom.SobolEngine(bounds.shape[1], scramble=True, seed=seed)
    unit = engine.draw(num_samples).to(dtype=bounds.dtype, device=bounds.device)
    choices = bounds[0] + (bounds[1] - bounds[0]) * unit
    with torch.no_grad():
        margins, safe = safety_confidence_bounds(
            safety_model, choices.unsqueeze(-2), constraints, beta=beta, tolerance=tolerance
        )
        posterior = safety_model.posterior(choices.unsqueeze(-2))
        widths = posterior.variance.sqrt()
    if safe.shape != (num_samples, 1):
        raise ValueError("safety model must return one decision per candidate.")
    safe_indices = torch.where(safe.squeeze(-1))[0]
    if safe_indices.numel() == 0:
        raise RuntimeError("No posterior-safe candidates were found for expansion.")

    output_indices = []
    for constraint in constraints:
        index = constraint.output
        if isinstance(index, str):
            if not hasattr(safety_model, "resolve_output"):
                raise TypeError("Named safety outputs require model.resolve_output.")
            index = safety_model.resolve_output(index)
        output_indices.append(index)
    score = widths[..., output_indices].sum(dim=-1).squeeze(-1)
    selected = safe_indices[score[safe_indices].argmax()]
    candidate = choices[selected].unsqueeze(0)
    with torch.no_grad():
        _, verified = safety_confidence_bounds(
            safety_model, candidate.unsqueeze(0), constraints, beta=beta, tolerance=tolerance
        )
    if not bool(verified.all()):
        raise RuntimeError("Expansion candidate failed posterior safety recheck.")
    return candidate, score[selected]
