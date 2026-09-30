"""Particle-swarm acquisition optimizer backend."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from torch import Tensor

from robotorchan.optim.constraints.evaluation import candidate_constraint_violation
from robotorchan.optim.constraints.contracts import CandidateConstraints
from robotorchan.optim.cross_cutting import apply_fixed_features
from robotorchan.optim.runtime import make_generator, validate_bounds
from robotorchan.optim.variable_space import MixedVariableSpace


def optimize_acqf_pso(
    acq_function: AcquisitionFunction,
    bounds: Tensor,
    q: int,
    *,
    swarm_size: int = 128,
    iterations: int = 100,
    inertia: float = 0.7298,
    cognitive: float = 1.49618,
    social: float = 1.49618,
    seed: int | None = None,
    constraints: CandidateConstraints | None = None,
    fixed_features: dict[int, float | Tensor] | None = None,
    constraint_penalty: float = 1e6,
    equality_tolerance: float = 1e-6,
    integer_dims: Sequence[int] = (),
    variable_space: MixedVariableSpace | None = None,
) -> tuple[Tensor, Tensor]:
    """Optimize a scalar acquisition function with particle swarm optimization."""
    validate_bounds(bounds)
    if variable_space is not None:
        if integer_dims:
            raise ValueError("Use variable_space or integer_dims, not both.")
        if variable_space.categorical_dims:
            raise ValueError("PSO does not support categorical variables.")
        if not torch.equal(variable_space.bounds, bounds):
            raise ValueError("variable_space bounds must match bounds.")
        integer_dims = variable_space.integer_dims
        variable_space.validate_fixed_features(fixed_features)
    integer_dims = tuple(integer_dims)
    _validate_integer_dims(integer_dims, bounds)
    _validate(bounds, q, swarm_size, iterations, inertia, cognitive, social)
    if constraint_penalty <= 0:
        raise ValueError("constraint_penalty must be positive.")
    candidate_constraints = constraints or CandidateConstraints()
    generator = make_generator(bounds, seed)

    d = bounds.shape[-1]
    lower = bounds[0].repeat(q)
    upper = bounds[1].repeat(q)
    span = upper - lower
    positions = lower + span * torch.rand(
        swarm_size, q * d, dtype=bounds.dtype, device=bounds.device, generator=generator
    )
    velocities = (
        2.0
        * torch.rand(
            swarm_size, q * d, dtype=bounds.dtype, device=bounds.device, generator=generator
        )
        - 1.0
    ) * span

    scores, _ = _evaluate(
        acq_function,
        _repair_integer_positions(positions, integer_dims, bounds, q),
        q,
        d,
        candidate_constraints,
        constraint_penalty,
        equality_tolerance,
        fixed_features,
    )
    personal_positions = positions.clone()
    personal_scores = scores.clone()
    best_index = personal_scores.argmax()
    global_position = personal_positions[best_index].clone()
    global_score = personal_scores[best_index].clone()

    for _ in range(iterations):
        r1 = torch.rand(
            positions.shape, dtype=bounds.dtype, device=bounds.device, generator=generator
        )
        r2 = torch.rand(
            positions.shape, dtype=bounds.dtype, device=bounds.device, generator=generator
        )
        velocities = (
            inertia * velocities
            + cognitive * r1 * (personal_positions - positions)
            + social * r2 * (global_position - positions)
        )
        positions = torch.maximum(torch.minimum(positions + velocities, upper), lower)
        repaired_positions = _repair_integer_positions(positions, integer_dims, bounds, q)
        scores, _ = _evaluate(
            acq_function,
            repaired_positions,
            q,
            d,
            candidate_constraints,
            constraint_penalty,
            equality_tolerance,
            fixed_features,
        )
        improved = scores > personal_scores
        personal_positions[improved] = positions[improved]
        personal_scores[improved] = scores[improved]
        best_index = personal_scores.argmax()
        if personal_scores[best_index] > global_score:
            global_score = personal_scores[best_index].clone()
            global_position = personal_positions[best_index].clone()

    candidate = _repair_integer_positions(
        global_position.unsqueeze(0), integer_dims, bounds, q
    ).reshape(q, d)
    candidate = apply_fixed_features(candidate, fixed_features)
    with torch.no_grad():
        value = acq_function(candidate.unsqueeze(0)).reshape(())
    return candidate, value


def _evaluate(
    acq_function: AcquisitionFunction,
    positions: Tensor,
    q: int,
    d: int,
    constraints: CandidateConstraints,
    constraint_penalty: float,
    equality_tolerance: float,
    fixed_features: dict[int, float | Tensor] | None,
) -> tuple[Tensor, Tensor]:
    candidates = positions.reshape(positions.shape[0], q, d)
    candidates = apply_fixed_features(candidates, fixed_features)
    with torch.no_grad():
        values = acq_function(candidates)
    if values.numel() != positions.shape[0]:
        raise ValueError("PSO requires one scalar acquisition value per particle.")
    raw_values = values.reshape(positions.shape[0])
    violation = candidate_constraint_violation(
        candidates, constraints, equality_tolerance=equality_tolerance
    )
    return raw_values - constraint_penalty * violation, raw_values


def _validate(
    bounds: Tensor,
    q: int,
    swarm_size: int,
    iterations: int,
    inertia: float,
    cognitive: float,
    social: float,
) -> None:
    if q < 1:
        raise ValueError("q must be at least 1.")
    if bounds.ndim != 2 or bounds.shape[0] != 2:
        raise ValueError("bounds must have shape [2, d].")
    if torch.any(bounds[0] >= bounds[1]):
        raise ValueError("Every lower bound must be strictly smaller than its upper bound.")
    if swarm_size < 2:
        raise ValueError("swarm_size must be at least 2.")
    if iterations < 0:
        raise ValueError("iterations must be non-negative.")
    if inertia < 0 or cognitive < 0 or social < 0:
        raise ValueError("PSO coefficients must be non-negative.")


def _validate_integer_dims(integer_dims: tuple[int, ...], bounds: Tensor) -> None:
    if len(integer_dims) != len(set(integer_dims)):
        raise ValueError("integer_dims must not contain duplicates.")
    d = bounds.shape[-1]
    for dim in integer_dims:
        if dim < 0 or dim >= d:
            raise ValueError(f"integer dimension {dim} is out of range.")
        if torch.ceil(bounds[0, dim]) > torch.floor(bounds[1, dim]):
            raise ValueError(f"integer dimension {dim} has no legal integer value.")


def _repair_integer_positions(
    positions: Tensor,
    integer_dims: tuple[int, ...],
    bounds: Tensor,
    q: int,
) -> Tensor:
    if not integer_dims:
        return positions
    d = bounds.shape[-1]
    candidates = positions.reshape(positions.shape[0], q, d).clone()
    index = torch.tensor(integer_dims, device=positions.device)
    lower = torch.ceil(bounds[0, index])
    upper = torch.floor(bounds[1, index])
    candidates[..., index] = candidates[..., index].round()
    candidates[..., index] = torch.maximum(torch.minimum(candidates[..., index], upper), lower)
    return candidates.reshape(positions.shape[0], q * d)
