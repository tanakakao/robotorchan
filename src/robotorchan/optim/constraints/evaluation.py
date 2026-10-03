"""Constraint evaluation helpers for derivative-free optimizer backends."""

from __future__ import annotations

import torch
from torch import Tensor

from robotorchan.optim.constraints.contracts import CandidateConstraints, LinearConstraint


def candidate_constraint_violation(
    candidates: Tensor,
    constraints: CandidateConstraints | None,
    *,
    equality_tolerance: float = 1e-6,
) -> Tensor:
    """Return non-negative total violation for each candidate q-batch."""
    if candidates.ndim < 2:
        raise ValueError("candidates must end in [q, d].")
    batch_shape = candidates.shape[:-2]
    violation = torch.zeros(batch_shape, dtype=candidates.dtype, device=candidates.device)
    if constraints is None:
        return violation
    for constraint in constraints.inequality_constraints:
        residual = _linear_residual(candidates, constraint)
        violation = violation + torch.relu(-residual)
    for constraint in constraints.equality_constraints:
        residual = _linear_residual(candidates, constraint)
        violation = violation + torch.relu(residual.abs() - equality_tolerance)
    for callable_, is_intrapoint in constraints.nonlinear_inequality_constraints:
        if is_intrapoint:
            values = _evaluate_intrapoint_callable(callable_, candidates)
            violation = violation + torch.relu(-values).sum(dim=-1)
        else:
            values = _evaluate_interpoint_callable(callable_, candidates)
            violation = violation + torch.relu(-values)
    return violation


def feasibility_first_ranks(values: Tensor, violation: Tensor) -> Tensor:
    """Return ordinal scores implementing Deb-style feasibility-first ordering."""
    if values.shape != violation.shape:
        raise ValueError("values and violation must have the same shape.")
    if values.ndim != 1:
        raise ValueError("values and violation must be one-dimensional.")

    feasible = violation <= 0
    feasible_indices = torch.nonzero(feasible, as_tuple=False).flatten()
    infeasible_indices = torch.nonzero(~feasible, as_tuple=False).flatten()

    if feasible_indices.numel():
        feasible_order = feasible_indices[
            torch.argsort(values[feasible_indices], descending=True, stable=True)
        ]
    else:
        feasible_order = feasible_indices
    if infeasible_indices.numel():
        infeasible_order = infeasible_indices[
            torch.argsort(violation[infeasible_indices], stable=True)
        ]
    else:
        infeasible_order = infeasible_indices

    order = torch.cat([feasible_order, infeasible_order])
    ranks = torch.empty_like(values)
    ranks[order] = torch.arange(order.numel(), 0, -1, dtype=values.dtype, device=values.device)
    return ranks


def candidate_is_feasible(
    candidates: Tensor,
    constraints: CandidateConstraints | None,
    *,
    equality_tolerance: float = 1e-6,
) -> Tensor:
    """Return feasibility for each candidate q-batch."""
    return (
        candidate_constraint_violation(
            candidates, constraints, equality_tolerance=equality_tolerance
        )
        <= 0
    )


def _linear_residual(candidates: Tensor, constraint: LinearConstraint) -> Tensor:
    indices, coefficients, rhs = constraint
    indices = indices.to(device=candidates.device)
    coefficients = coefficients.to(device=candidates.device, dtype=candidates.dtype)
    if indices.ndim == 1:
        selected = candidates[..., indices]
        return (selected * coefficients).sum(dim=-1).amin(dim=-1) - rhs
    if indices.ndim == 2 and indices.shape[-1] == 2:
        values = torch.stack(
            [candidates[..., int(q_idx), int(d_idx)] for q_idx, d_idx in indices.tolist()],
            dim=-1,
        )
        return (values * coefficients).sum(dim=-1) - rhs
    raise ValueError("Linear constraint indices must be one- or two-dimensional.")


def _evaluate_intrapoint_callable(callable_, candidates: Tensor) -> Tensor:
    flat = candidates.reshape(-1, candidates.shape[-1])
    values = [_validate_nonlinear_value(callable_(candidate), candidate) for candidate in flat]
    return torch.stack(values).reshape(candidates.shape[:-1])


def _evaluate_interpoint_callable(callable_, candidates: Tensor) -> Tensor:
    if candidates.ndim == 2:
        return _validate_nonlinear_value(callable_(candidates), candidates)
    flat = candidates.reshape(-1, candidates.shape[-2], candidates.shape[-1])
    values = [_validate_nonlinear_value(callable_(candidate), candidate) for candidate in flat]
    return torch.stack(values).reshape(candidates.shape[:-2])


def _validate_nonlinear_value(value: object, candidate: Tensor) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError("Nonlinear candidate constraint callable must return a Tensor.")
    if value.numel() != 1:
        raise ValueError("Nonlinear candidate constraint callable must return a scalar Tensor.")
    if value.device != candidate.device:
        raise ValueError("Nonlinear candidate constraint callable must preserve candidate device.")
    if value.dtype != candidate.dtype:
        raise ValueError("Nonlinear candidate constraint callable must preserve candidate dtype.")
    if not torch.isfinite(value).all():
        raise ValueError("Nonlinear candidate constraint callable must return a finite value.")
    return value.reshape(())
