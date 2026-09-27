"""Differential-evolution acquisition optimizer backend."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from scipy.optimize import differential_evolution
from torch import Tensor

from robotorchan.optim.constraint_evaluation import candidate_constraint_violation
from robotorchan.optim.constraints import CandidateConstraints
from robotorchan.optim.runtime import validate_bounds
from robotorchan.optim.cross_cutting import apply_fixed_features


def optimize_acqf_de(
    acq_function: AcquisitionFunction,
    bounds: Tensor,
    q: int,
    *,
    options: dict[str, Any] | None = None,
    constraints: CandidateConstraints | None = None,
    fixed_features: dict[int, float | Tensor] | None = None,
    seed: int | None = None,
    constraint_penalty: float = 1e6,
    equality_tolerance: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    """Optimize an acquisition function with SciPy Differential Evolution.

    The joint ``q x d`` candidate batch is flattened into one DE decision
    vector. Acquisition evaluation is performed on the original tensor
    device and dtype. Candidate constraints are deliberately rejected until
    the dedicated cross-optimizer constraint phase.
    """
    if q < 1:
        raise ValueError("q must be at least 1.")
    if bounds.ndim != 2 or bounds.shape[0] != 2:
        raise ValueError("bounds must have shape [2, d].")
    candidate_constraints = constraints or CandidateConstraints()
    if constraint_penalty <= 0:
        raise ValueError("constraint_penalty must be positive.")

    scipy_bounds = list(
        zip(
            bounds[0].detach().cpu().tolist() * q,
            bounds[1].detach().cpu().tolist() * q,
            strict=True,
        )
    )
    resolved_options = {} if options is None else dict(options)
    resolved_options.setdefault("polish", False)
    resolved_options.setdefault("updating", "immediate")

    def objective(flat_candidate: np.ndarray) -> float:
        candidate = torch.as_tensor(
            flat_candidate, dtype=bounds.dtype, device=bounds.device
        ).reshape(q, bounds.shape[-1])
        candidate = apply_fixed_features(candidate, fixed_features)
        with torch.no_grad():
            value = acq_function(candidate.unsqueeze(0))
        if value.numel() != 1:
            raise ValueError(
                "Differential Evolution requires a scalar acquisition value per q-batch."
            )
        violation = candidate_constraint_violation(
            candidate.unsqueeze(0),
            candidate_constraints,
            equality_tolerance=equality_tolerance,
        ).reshape(())
        return -float(value.reshape(()).detach().cpu()) + constraint_penalty * float(
            violation.detach().cpu()
        )

    result = differential_evolution(
        objective,
        scipy_bounds,
        seed=seed,
        **resolved_options,
    )
    candidates = torch.as_tensor(
        result.x,
        dtype=bounds.dtype,
        device=bounds.device,
    ).reshape(q, bounds.shape[-1])
    candidates = apply_fixed_features(candidates, fixed_features)
    with torch.no_grad():
        acquisition_value = acq_function(candidates.unsqueeze(0)).reshape(())
    return candidates, acquisition_value
