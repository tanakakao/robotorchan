"""Runtime validation for candidate-space constraints."""

from __future__ import annotations

import torch
from botorch.exceptions.errors import UnsupportedError
from torch import Tensor

from robotorchan.optim.constraints.contracts import CandidateConstraints


def validate_candidate_constraints(
    constraints: CandidateConstraints | None,
    *,
    bounds: Tensor,
    q: int,
    sequential: bool = False,
) -> None:
    """Validate candidate constraints once public input dimension and q are known."""
    if constraints is None:
        return
    d = bounds.shape[-1]
    for name, linear_constraints in (
        ("inequality_constraints", constraints.inequality_constraints),
        ("equality_constraints", constraints.equality_constraints),
    ):
        for indices, _, _ in linear_constraints:
            if indices.numel() == 0:
                raise ValueError(f"{name} indices must not be empty.")
            if indices.ndim == 1:
                if (indices < 0).any() or (indices >= d).any():
                    raise ValueError(f"{name} contains an input-dimension index outside [0, d).")
            else:
                q_indices = indices[:, 0]
                d_indices = indices[:, 1]
                if (q_indices < 0).any() or (q_indices >= q).any():
                    raise ValueError(f"{name} contains a q index outside [0, q).")
                if (d_indices < 0).any() or (d_indices >= d).any():
                    raise ValueError(f"{name} contains an input-dimension index outside [0, d).")
                if sequential and q > 1:
                    raise UnsupportedError(
                        "inter-point linear constraints require joint q-batch optimization; "
                        "sequential=True is unsupported."
                    )

    if (
        sequential
        and q > 1
        and any(
            not is_intrapoint for _, is_intrapoint in constraints.nonlinear_inequality_constraints
        )
    ):
        raise UnsupportedError(
            "inter-point nonlinear constraints require joint q-batch optimization; "
            "sequential=True is unsupported."
        )


def validate_nonlinear_initial_conditions(
    batch_initial_conditions: Tensor | None,
    constraints: CandidateConstraints,
    *,
    bounds: Tensor,
    q: int,
    tolerance: float = 1e-8,
) -> None:
    """Validate explicit restart points before nonlinear BoTorch optimization."""
    if batch_initial_conditions is None or not constraints.has_nonlinear_constraints:
        return
    if batch_initial_conditions.ndim != 3:
        raise ValueError(
            "batch_initial_conditions for nonlinear constraints must have shape "
            "[num_restarts, q, d]."
        )
    if batch_initial_conditions.shape[1:] != (q, bounds.shape[-1]):
        raise ValueError(
            "batch_initial_conditions for nonlinear constraints must have shape "
            f"[num_restarts, {q}, {bounds.shape[-1]}]."
        )
    if batch_initial_conditions.shape[0] == 0:
        raise ValueError("batch_initial_conditions must contain at least one restart.")
    if batch_initial_conditions.device != bounds.device:
        raise ValueError("batch_initial_conditions must use the same device as bounds.")
    if batch_initial_conditions.dtype != bounds.dtype:
        raise ValueError("batch_initial_conditions must use the same dtype as bounds.")
    if not torch.isfinite(batch_initial_conditions).all():
        raise ValueError("batch_initial_conditions must contain only finite values.")
    lower, upper = bounds
    if (batch_initial_conditions < lower - tolerance).any() or (
        batch_initial_conditions > upper + tolerance
    ).any():
        raise ValueError("batch_initial_conditions must lie within bounds.")

    for restart_index, restart in enumerate(batch_initial_conditions):
        for constraint, is_intrapoint in constraints.nonlinear_inequality_constraints:
            if is_intrapoint:
                values = [constraint(candidate) for candidate in restart]
            else:
                values = [constraint(restart)]
            for value in values:
                if not isinstance(value, Tensor) or value.numel() != 1:
                    raise ValueError(
                        "Nonlinear constraint callables must return one scalar Tensor."
                    )
                if not torch.isfinite(value).all():
                    raise ValueError("Nonlinear constraint values must be finite.")
                if value.detach().item() < -tolerance:
                    raise ValueError(
                        "batch_initial_conditions must be feasible for all nonlinear "
                        f"constraints; restart {restart_index} is infeasible."
                    )
