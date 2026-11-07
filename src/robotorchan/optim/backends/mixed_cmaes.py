"""Mixed-variable CMA-ES using the upstream categorical-with-margin algorithm."""

from __future__ import annotations

import numpy as np
import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from torch import Tensor

from robotorchan.optim.backend_support.runtime import validate_bounds
from robotorchan.optim.constraints.contracts import (
    CandidateConstraints,
    LinearConstraint,
    NonlinearConstraint,
    resolve_candidate_constraints,
)
from robotorchan.optim.constraints.evaluation import candidate_constraint_violation
from robotorchan.optim.constraints.validation import validate_candidate_constraints
from robotorchan.optim.domains.variable_space import MixedVariableSpace


def optimize_acqf_mixed_cmaes(
    acq_function: AcquisitionFunction,
    bounds: Tensor,
    q: int,
    *,
    variable_space: MixedVariableSpace,
    population_size: int | None = None,
    max_generations: int = 200,
    seed: int | None = None,
    constraints: CandidateConstraints | None = None,
    inequality_constraints: tuple[LinearConstraint, ...] | None = None,
    equality_constraints: tuple[LinearConstraint, ...] | None = None,
    nonlinear_inequality_constraints: tuple[NonlinearConstraint, ...] | None = None,
    fixed_features: dict[int, float | Tensor] | None = None,
    constraint_penalty: float = 1e6,
    equality_tolerance: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    """Optimize mixed domains with upstream CatCMAwM, without ordinal category encoding."""
    validate_bounds(bounds)
    candidate_constraints = resolve_candidate_constraints(
        constraints,
        inequality_constraints=inequality_constraints,
        equality_constraints=equality_constraints,
        nonlinear_inequality_constraints=nonlinear_inequality_constraints,
    )
    validate_candidate_constraints(candidate_constraints, bounds=bounds, q=q)
    if not torch.equal(variable_space.bounds, bounds):
        raise ValueError("variable_space bounds must match bounds.")
    if q < 1 or max_generations < 1:
        raise ValueError("q and max_generations must be positive.")
    if constraint_penalty <= 0:
        raise ValueError("constraint_penalty must be positive.")
    variable_space.validate_fixed_features(fixed_features)
    if not (variable_space.integer_dims or variable_space.categorical_dims):
        raise ValueError("Mixed CMA-ES requires an integer or categorical dimension.")

    try:
        from cmaes import CatCMAwM
    except ImportError as error:
        raise ImportError(
            "Mixed CMA-ES requires the optional 'cmaes' dependency. Install robotorchan[cmaes]."
        ) from error

    d = bounds.shape[-1]
    fixed = {
        dim: float(torch.as_tensor(value).reshape(()))
        for dim, value in (fixed_features or {}).items()
    }
    continuous: list[int] = []
    integer: list[int] = []
    categorical: list[int] = []
    x_space: list[list[float]] = []
    z_space: list[list[int]] = []
    c_space: list[int] = []
    for index in range(q * d):
        dim = index % d
        if dim in fixed:
            continue
        if dim in variable_space.categorical_dims:
            values = variable_space.categorical_values[dim]
            if len(values) < 2:
                raise ValueError(
                    "Categorical CMA-ES requires at least two values per free category."
                )
            categorical.append(index)
            c_space.append(len(values))
        elif dim in variable_space.integer_dims:
            lower = int(torch.ceil(bounds[0, dim]).item())
            upper = int(torch.floor(bounds[1, dim]).item())
            if lower == upper:
                fixed[dim] = float(lower)
                continue
            integer.append(index)
            z_space.append(list(range(lower, upper + 1)))
        else:
            continuous.append(index)
            x_space.append([float(bounds[0, dim]), float(bounds[1, dim])])
    if not (continuous or integer or categorical):
        candidate = torch.stack(
            [
                torch.tensor(
                    [fixed[dim] for dim in range(d)], dtype=bounds.dtype, device=bounds.device
                )
                for _ in range(q)
            ]
        )
        with torch.no_grad():
            violation = candidate_constraint_violation(
                candidate.unsqueeze(0),
                candidate_constraints,
                equality_tolerance=equality_tolerance,
            )
            if bool((violation > 0).any()):
                raise ValueError("The only available mixed candidate violates constraints.")
            return candidate, acq_function(candidate.unsqueeze(0)).reshape(())

    kwargs: dict = {"seed": seed}
    if population_size is not None:
        kwargs["population_size"] = population_size
    optimizer = CatCMAwM(
        x_space=x_space or None,
        z_space=z_space or None,
        c_space=c_space or None,
        **kwargs,
    )
    best_candidate: Tensor | None = None
    best_score = float("-inf")
    for _ in range(max_generations):
        solutions = []
        for _ in range(optimizer.population_size):
            solution = optimizer.ask()
            flat = torch.empty(q * d, dtype=bounds.dtype, device=bounds.device)
            for index in range(q * d):
                if index % d in fixed:
                    flat[index] = fixed[index % d]
            if continuous:
                flat[continuous] = torch.as_tensor(
                    solution.x, dtype=bounds.dtype, device=bounds.device
                )
            if integer:
                flat[integer] = torch.as_tensor(
                    solution.z, dtype=bounds.dtype, device=bounds.device
                )
            for i, index in enumerate(categorical):
                dim = index % d
                category = int(np.argmax(solution.c[i]))
                flat[index] = variable_space.categorical_values[dim][category]
            candidate = flat.reshape(q, d)
            with torch.no_grad():
                value = acq_function(candidate.unsqueeze(0))
            if value.numel() != 1:
                raise ValueError("Mixed CMA-ES requires a scalar acquisition value per q-batch.")
            violation = candidate_constraint_violation(
                candidate.unsqueeze(0),
                candidate_constraints,
                equality_tolerance=equality_tolerance,
            ).reshape(())
            score = float(value.reshape(())) - constraint_penalty * float(violation)
            solutions.append((solution, -score))
            if score > best_score:
                best_score = score
                best_candidate = candidate.clone()
        optimizer.tell(solutions)
        if optimizer.should_stop():
            break
    if best_candidate is None:
        raise RuntimeError("Mixed CMA-ES did not generate any candidate.")
    with torch.no_grad():
        best_value = acq_function(best_candidate.unsqueeze(0)).reshape(())
    return best_candidate, best_value
