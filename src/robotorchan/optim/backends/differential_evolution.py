"""Differential-evolution acquisition optimizer backend."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from scipy.optimize import NonlinearConstraint, differential_evolution
from torch import Tensor

from robotorchan.optim.constraints.contracts import CandidateConstraints
from robotorchan.optim.constraints.evaluation import candidate_constraint_violation
from robotorchan.optim.cross_cutting import apply_fixed_features
from robotorchan.optim.runtime import validate_bounds
from robotorchan.optim.variable_space import MixedVariableSpace


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
    integer_dims: Sequence[int] = (),
    variable_space: MixedVariableSpace | None = None,
    categorical_values: Mapping[int, Sequence[float]] | None = None,
) -> tuple[Tensor, Tensor]:
    """Optimize an acquisition function with SciPy Differential Evolution.

    The joint ``q x d`` candidate batch is flattened into one DE decision
    vector. Acquisition evaluation is performed on the original tensor
    device and dtype. Integer coordinates are repaired before acquisition
    and constraint evaluation. Unordered categorical coordinates are rejected
    rather than assigned an artificial numeric geometry.
    """
    validate_bounds(bounds)
    if variable_space is not None:
        if integer_dims or categorical_values:
            raise ValueError("Use variable_space or integer_dims/categorical_values, not both.")
        if not torch.equal(variable_space.bounds, bounds):
            raise ValueError("variable_space bounds must match bounds.")
        integer_dims = variable_space.integer_dims
        categorical_values = dict(variable_space.categorical_values)
        variable_space.validate_fixed_features(fixed_features)
    else:
        categorical_values = dict(categorical_values or {})
    integer_dims = tuple(integer_dims)
    _validate_structured_dims(integer_dims, categorical_values, bounds)
    if categorical_values:
        raise NotImplementedError(
            "Differential Evolution does not support unordered categorical variables. "
            "Use Mixed GA or a sampling backend instead."
        )
    if q < 1:
        raise ValueError("q must be at least 1.")
    candidate_constraints = constraints or CandidateConstraints()
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
        candidate = _repair_structured_dims(candidate, integer_dims, categorical_values, bounds)
        candidate = apply_fixed_features(candidate, fixed_features)
        with torch.no_grad():
            value = acq_function(candidate.unsqueeze(0))
        if value.numel() != 1:
            raise ValueError(
                "Differential Evolution requires a scalar acquisition value per q-batch."
            )
        return -float(value.reshape(()).detach().cpu())

    scipy_constraints = ()
    if candidate_constraints != CandidateConstraints():

        def feasibility(flat_candidate: np.ndarray) -> float:
            candidate = torch.as_tensor(
                flat_candidate, dtype=bounds.dtype, device=bounds.device
            ).reshape(q, bounds.shape[-1])
            candidate = _repair_structured_dims(candidate, integer_dims, categorical_values, bounds)
            candidate = apply_fixed_features(candidate, fixed_features)
            violation = candidate_constraint_violation(
                candidate.unsqueeze(0),
                candidate_constraints,
                equality_tolerance=equality_tolerance,
            ).reshape(())
            return -float(violation.detach().cpu())

        scipy_constraints = (NonlinearConstraint(feasibility, 0.0, np.inf),)

    result = differential_evolution(
        objective,
        scipy_bounds,
        seed=seed,
        constraints=scipy_constraints,
        **resolved_options,
    )
    candidates = torch.as_tensor(
        result.x,
        dtype=bounds.dtype,
        device=bounds.device,
    ).reshape(q, bounds.shape[-1])
    candidates = _repair_structured_dims(candidates, integer_dims, categorical_values, bounds)
    candidates = apply_fixed_features(candidates, fixed_features)
    with torch.no_grad():
        acquisition_value = acq_function(candidates.unsqueeze(0)).reshape(())
    return candidates, acquisition_value


def _validate_structured_dims(
    integer_dims: tuple[int, ...],
    categorical_values: dict[int, Sequence[float]],
    bounds: Tensor,
) -> None:
    if len(integer_dims) != len(set(integer_dims)):
        raise ValueError("integer_dims must not contain duplicates.")
    d = bounds.shape[-1]
    structured_dims = set(integer_dims) | set(categorical_values)
    if any(dim < 0 or dim >= d for dim in structured_dims):
        raise ValueError("Structured dimensions must lie within the input dimension.")
    if set(integer_dims) & set(categorical_values):
        raise ValueError("A dimension cannot be both integer and categorical.")
    for dim in integer_dims:
        lower = int(torch.ceil(bounds[0, dim]).item())
        upper = int(torch.floor(bounds[1, dim]).item())
        if lower > upper:
            raise ValueError(f"integer dimension {dim} has no legal integer value.")
    for dim, values in categorical_values.items():
        if not values:
            raise ValueError(f"categorical_values[{dim}] must not be empty.")
        tensor_values = torch.as_tensor(values, device=bounds.device, dtype=bounds.dtype)
        if not torch.isfinite(tensor_values).all():
            raise ValueError(f"categorical_values[{dim}] must be finite.")
        if tensor_values.unique().numel() != tensor_values.numel():
            raise ValueError(f"categorical_values[{dim}] must not contain duplicates.")
        if torch.any(tensor_values < bounds[0, dim]) or torch.any(tensor_values > bounds[1, dim]):
            raise ValueError(f"categorical_values[{dim}] must lie within bounds.")


def _repair_structured_dims(
    candidates: Tensor,
    integer_dims: tuple[int, ...],
    categorical_values: dict[int, Sequence[float]],
    bounds: Tensor,
) -> Tensor:
    result = candidates.clone()
    if integer_dims:
        index = torch.tensor(integer_dims, device=candidates.device)
        lower = torch.ceil(bounds[0, index])
        upper = torch.floor(bounds[1, index])
        result[..., index] = result[..., index].round()
        result[..., index] = torch.maximum(torch.minimum(result[..., index], upper), lower)
    for dim, values in categorical_values.items():
        legal_values = torch.as_tensor(values, device=candidates.device, dtype=candidates.dtype)
        distances = (result[..., dim, None] - legal_values).abs()
        result[..., dim] = legal_values[distances.argmin(dim=-1)]
    return result
