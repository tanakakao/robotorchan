"""Hamming geometry for finite combinatorial candidate spaces."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor

from robotorchan.optim.constraints.contracts import CandidateConstraints
from robotorchan.optim.constraints.evaluation import candidate_is_feasible
from robotorchan.optim.domains.variable_space import MixedVariableSpace


def _levels(space: MixedVariableSpace) -> tuple[tuple[float, ...], ...]:
    """Return legal coordinate values; continuous dimensions are unsupported."""
    if space.continuous_dims:
        raise ValueError("Hamming search requires a fully discrete MixedVariableSpace.")
    result = []
    for dim in range(space.input_dim):
        if dim in space.categorical_dims:
            result.append(tuple(space.categorical_values[dim]))
        else:
            lower = int(torch.ceil(space.bounds[0, dim]).item())
            upper = int(torch.floor(space.bounds[1, dim]).item())
            result.append(range(lower, upper + 1))
    return tuple(result)


def hamming_distance(x: Tensor, y: Tensor, space: MixedVariableSpace) -> Tensor:
    """Count differing discrete coordinates, broadcasting leading dimensions."""
    if space.continuous_dims:
        raise ValueError("Hamming search requires a fully discrete MixedVariableSpace.")
    if x.shape[-1] != space.input_dim or y.shape[-1] != space.input_dim:
        raise ValueError("Candidate feature dimension does not match the space.")
    _validate_points(x, space)
    _validate_points(y, space)
    return (x != y).sum(dim=-1)


def _validate_points(
    points: Tensor,
    space: MixedVariableSpace,
) -> None:
    if points.ndim < 1 or points.shape[-1] != space.input_dim:
        raise ValueError("Candidates must end with the space's feature dimension.")
    if not torch.isfinite(points).all():
        raise ValueError("Candidates must be finite.")
    for dim in range(space.input_dim):
        values = points[..., dim]
        if dim in space.integer_dims:
            valid = (
                (values == values.round())
                & (values >= space.bounds[0, dim])
                & (values <= space.bounds[1, dim])
            )
        else:
            allowed = torch.as_tensor(
                space.categorical_values[dim], dtype=points.dtype, device=points.device
            )
            valid = torch.isin(values, allowed)
        if not valid.all():
            raise ValueError(f"Candidate contains an illegal value at dimension {dim}.")


def hamming_neighbors(
    center: Tensor,
    space: MixedVariableSpace,
    *,
    constraints: CandidateConstraints | None = None,
    excluded: Iterable[tuple[float, ...]] = (),
) -> Tensor:
    """Enumerate unique feasible radius-one neighbors without full-space enumeration.

    The center is one valid [d] candidate. Constraints use the existing q=1
    CandidateConstraints contract. Empty neighborhoods return shape [0, d].
    """
    levels = _levels(space)
    _validate_points(center, space)
    if center.ndim != 1:
        raise ValueError("center must have shape [d].")
    excluded_keys = set(excluded)
    candidates = []
    for dim, values in enumerate(levels):
        for value in values:
            neighbor = center.clone()
            neighbor[dim] = value
            if neighbor[dim] == center[dim]:
                continue
            key = tuple(neighbor.tolist())
            if key not in excluded_keys:
                excluded_keys.add(key)
                candidates.append(neighbor)
    if not candidates:
        return center.new_empty((0, space.input_dim))
    result = torch.stack(candidates)
    if constraints is not None and constraints.has_constraints:
        mask = candidate_is_feasible(result.unsqueeze(-2), constraints)
        result = result[mask]
    return result
