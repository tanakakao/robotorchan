"""Feasibility checks for Study candidates in public search coordinates."""

from __future__ import annotations

import torch
from torch import Tensor

from robotorchan.optim.constraints.contracts import CandidateConstraints
from robotorchan.optim.constraints.evaluation import validate_nonlinear_value
from robotorchan.optim.constraints.validation import (
    validate_candidate_constraints,
    validate_mixed_constraint_dimensions,
)
from robotorchan.optim.domains.variable_space import MixedVariableSpace


def is_feasible(
    candidates: Tensor,
    constraints: CandidateConstraints | None,
    *,
    variable_space: MixedVariableSpace | None = None,
    tolerance: float = 1e-7,
) -> bool:
    """Check a [q, d] batch against the existing candidate constraint contract."""
    if constraints is None or not constraints.has_constraints:
        return True
    if candidates.ndim != 2:
        raise ValueError("Candidates must have shape [q, d].")
    q, _ = candidates.shape
    validate_candidate_constraints(
        constraints,
        bounds=torch.stack((candidates.min(dim=0).values - 1, candidates.max(dim=0).values + 1)),
        q=q,
    )
    validate_mixed_constraint_dimensions(constraints, variable_space)
    for linear, equality in (
        (constraints.inequality_constraints, False),
        (constraints.equality_constraints, True),
    ):
        for indices, coefficients, rhs in linear:
            if indices.ndim == 1:
                values = candidates[:, indices.long()] @ coefficients.to(candidates)
            else:
                values = (
                    candidates[indices[:, 0].long(), indices[:, 1].long()]
                    * coefficients.to(candidates)
                ).sum().reshape(1)
            if equality:
                if torch.any(torch.abs(values - rhs) > tolerance):
                    return False
            elif torch.any(values < rhs - tolerance):
                return False
    for function, is_intrapoint in constraints.nonlinear_inequality_constraints:
        inputs = candidates if is_intrapoint else candidates.unsqueeze(0)
        for x in inputs:
            value = validate_nonlinear_value(function(x), x)
            if value.detach().item() < -tolerance:
                return False
    return True
