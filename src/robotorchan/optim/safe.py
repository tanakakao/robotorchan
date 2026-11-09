"""Safe q=1 candidate selection over a continuous Sobol candidate pool."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.optim.optimize import optimize_acqf_discrete
from torch import Tensor

from robotorchan.optim.backends.sampling import sample_mixed_candidate_batches
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
) -> tuple[Tensor, Tensor]:
    """Select the best posterior-safe candidate from a Sobol pool.

    Continuous and mixed-variable q=1 optimization are supported. This finite-pool search
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

    if variable_space is not None and not torch.equal(variable_space.bounds, bounds):
        raise ValueError("variable_space bounds must match bounds.")
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
