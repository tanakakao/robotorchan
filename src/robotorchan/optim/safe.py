"""Safe q=1 candidate selection over a continuous Sobol candidate pool."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.optim.optimize import optimize_acqf_discrete
from torch import Tensor

from robotorchan.optim.backends.sampling import (
    sample_candidate_batches,
    sample_mixed_candidate_batches,
    select_best_sampled_batch,
)
from robotorchan.optim.domains.variable_space import MixedVariableSpace
from robotorchan.semantics.constraints import ContinuousConstraint
from robotorchan.semantics.safety_bounds import safety_confidence_bounds


def optimize_safe_acqf(
    acq_function: AcquisitionFunction,
    safety_model: object,
    bounds: Tensor,
    constraints: Sequence[ContinuousConstraint],
    *,
    beta: float = 4.0,
    tolerance: float = 0.0,
    num_samples: int = 1024,
    seed: int | None = None,
    variable_space: MixedVariableSpace | None = None,
    q: int = 1,
) -> tuple[Tensor, Tensor]:
    """Select the best posterior-safe candidate from a Sobol pool.

    Continuous q-batch and mixed-variable q=1 optimization are supported. This finite-pool search
    can miss safe regions; it does not certify global safety or optimality.
    An empty predicted-safe pool raises instead of returning an unsafe point.
    """
    if bounds.ndim != 2 or bounds.shape[0] != 2 or bounds.shape[1] == 0:
        raise ValueError("bounds must have shape [2, d].")
    if not bounds.is_floating_point() or not bool(torch.isfinite(bounds).all()):
        raise ValueError("bounds must contain finite floating values.")
    if not bool((bounds[1] > bounds[0]).all()):
        raise ValueError("bounds upper limits must exceed lower limits.")
    if isinstance(num_samples, bool) or not isinstance(num_samples, int) or num_samples < 1:
        raise ValueError("num_samples must be a positive integer.")

    if isinstance(q, bool) or not isinstance(q, int) or q < 1:
        raise ValueError("q must be a positive integer.")
    if q > 1 and variable_space is not None:
        raise ValueError("Mixed q-batch Safe BO is reserved for Phase 4.")
    if variable_space is not None and not torch.equal(variable_space.bounds, bounds):
        raise ValueError("variable_space bounds must match bounds.")
    if q > 1:
        batches = sample_candidate_batches(bounds, num_samples, q, seed=seed)
        with torch.no_grad():
            _, safe_mask = safety_confidence_bounds(
                safety_model, batches, constraints, beta=beta, tolerance=tolerance
            )
        if safe_mask.shape != (num_samples, q):
            raise ValueError("safety model must return one safety decision per batch point.")
        safe_batches = batches[safe_mask.all(dim=-1)]
        if safe_batches.shape[0] == 0:
            raise RuntimeError("No posterior-safe candidate batches were found in the sampled pool.")
        candidate, value = select_best_sampled_batch(acq_function, safe_batches)
        with torch.no_grad():
            _, verified = safety_confidence_bounds(
                safety_model, candidate, constraints, beta=beta, tolerance=tolerance
            )
        if not bool(verified.all()):
            raise RuntimeError("Selected batch failed the posterior safety recheck.")
        return candidate, value
    if variable_space is None:
        engine = torch.quasirandom.SobolEngine(bounds.shape[1], scramble=True, seed=seed)
        unit = engine.draw(num_samples).to(dtype=bounds.dtype, device=bounds.device)
        choices = bounds[0] + (bounds[1] - bounds[0]) * unit
    else:
        choices = sample_mixed_candidate_batches(variable_space, num_samples, seed=seed).squeeze(1)
    with torch.no_grad():
        _, safe_mask = safety_confidence_bounds(
            safety_model, choices.unsqueeze(-2), constraints, beta=beta, tolerance=tolerance
        )
    if safe_mask.shape != (num_samples, 1):
        raise ValueError("safety model must return one safety decision per candidate.")
    safe_choices = choices[safe_mask.squeeze(-1)]
    if safe_choices.shape[0] == 0:
        raise RuntimeError("No posterior-safe candidates were found in the sampled pool.")

    candidate, value = optimize_acqf_discrete(acq_function=acq_function, q=1, choices=safe_choices)
    with torch.no_grad():
        _, verified = safety_confidence_bounds(
            safety_model, candidate.unsqueeze(0), constraints, beta=beta, tolerance=tolerance
        )
    if not bool(verified.all()):
        raise RuntimeError("Selected candidate failed the posterior safety recheck.")
    return candidate, value
