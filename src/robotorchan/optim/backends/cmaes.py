"""CMA-ES acquisition optimizer backend."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from torch import Tensor

from robotorchan.optim.constraint_evaluation import candidate_constraint_violation
from robotorchan.optim.constraints import CandidateConstraints
from robotorchan.optim.runtime import validate_bounds


def optimize_acqf_cmaes(
    acq_function: AcquisitionFunction,
    bounds: Tensor,
    q: int,
    *,
    sigma: float = 0.25,
    population_size: int | None = None,
    max_generations: int = 200,
    seed: int | None = None,
    options: dict[str, Any] | None = None,
    constraints: CandidateConstraints | None = None,
    constraint_penalty: float = 1e6,
    equality_tolerance: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    """Optimize a BoTorch acquisition function with CMA-ES."""
    validate_bounds(bounds)
    if q < 1:
        raise ValueError("q must be at least 1.")
    if sigma <= 0:
        raise ValueError("sigma must be positive.")
    if max_generations < 1:
        raise ValueError("max_generations must be at least 1.")
    candidate_constraints = constraints or CandidateConstraints()
    if constraint_penalty <= 0:
        raise ValueError("constraint_penalty must be positive.")

    try:
        from cmaes import CMA
    except ImportError as error:
        raise ImportError(
            "CMA-ES optimization requires the optional 'cmaes' dependency. "
            "Install robotorchan[cmaes]."
        ) from error

    lower = bounds[0].detach().cpu().numpy()
    upper = bounds[1].detach().cpu().numpy()
    lower = np.tile(lower, q)
    upper = np.tile(upper, q)
    mean = (lower + upper) / 2.0
    scale = float(np.max(upper - lower))
    resolved_sigma = sigma * scale

    optimizer_options = {} if options is None else dict(options)
    optimizer_options.setdefault("bounds", np.stack([lower, upper], axis=1))
    optimizer_options.setdefault("seed", seed)
    if population_size is not None:
        optimizer_options.setdefault("population_size", population_size)

    optimizer = CMA(mean=mean, sigma=resolved_sigma, **optimizer_options)
    best_candidate: np.ndarray | None = None
    best_value = float("-inf")

    for _ in range(max_generations):
        solutions: list[tuple[np.ndarray, float]] = []
        for _ in range(optimizer.population_size):
            flat_candidate = optimizer.ask()
            value = _evaluate(acq_function, flat_candidate, bounds, q)
            candidate = torch.as_tensor(
                flat_candidate, dtype=bounds.dtype, device=bounds.device
            ).reshape(1, q, bounds.shape[-1])
            violation = candidate_constraint_violation(
                candidate,
                candidate_constraints,
                equality_tolerance=equality_tolerance,
            ).reshape(())
            penalized_objective = -value + constraint_penalty * float(violation.detach().cpu())
            solutions.append((flat_candidate, penalized_objective))
            if value > best_value:
                best_value = value
                best_candidate = flat_candidate.copy()
        optimizer.tell(solutions)
        if optimizer.should_stop():
            break

    if best_candidate is None:
        raise RuntimeError("CMA-ES did not generate any candidate.")
    candidates = torch.as_tensor(
        best_candidate,
        dtype=bounds.dtype,
        device=bounds.device,
    ).reshape(q, bounds.shape[-1])
    with torch.no_grad():
        acquisition_value = acq_function(candidates.unsqueeze(0)).reshape(())
    return candidates, acquisition_value


def _evaluate(
    acq_function: AcquisitionFunction,
    flat_candidate: np.ndarray,
    bounds: Tensor,
    q: int,
) -> float:
    candidate = torch.as_tensor(
        flat_candidate,
        dtype=bounds.dtype,
        device=bounds.device,
    ).reshape(q, bounds.shape[-1])
    with torch.no_grad():
        value = acq_function(candidate.unsqueeze(0))
    if value.numel() != 1:
        raise ValueError("CMA-ES requires a scalar acquisition value per q-batch.")
    return float(value.reshape(()).detach().cpu())
