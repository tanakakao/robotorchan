"""Correctness-first optimization for mixed one-shot acquisitions."""

from collections.abc import Sequence
from itertools import product

from botorch.acquisition.acquisition import AcquisitionFunction, OneShotAcquisitionFunction
from botorch.optim import optimize_acqf
from torch import Tensor

from robotorchan.optim.constraints.contracts import CandidateConstraints, LinearConstraint


def optimize_mixed_one_shot_acqf(
    acq_function: AcquisitionFunction,
    bounds: Tensor,
    *,
    categorical_features: dict[int, Sequence[float]],
    q: int = 1,
    num_restarts: int = 10,
    raw_samples: int = 256,
    max_assignments: int = 256,
    constraints: CandidateConstraints | None = None,
) -> tuple[Tensor, Tensor]:
    """Optimize row-specific discrete assignments, including one-shot auxiliary rows."""
    if q < 1:
        raise ValueError("q must be positive")
    one_shot = isinstance(acq_function, OneShotAcquisitionFunction)
    augmented_q = acq_function.get_augmented_q_batch_size(q) if one_shot else q
    dimensions = sorted(categorical_features)
    feature_count = bounds.shape[-1]
    for dimension in dimensions:
        if dimension < 0 or dimension >= feature_count:
            raise ValueError(
                f"categorical feature index {dimension} is outside [0, {feature_count})"
            )
    choices = [tuple(categorical_features[index]) for index in dimensions]
    for dimension, values in zip(dimensions, choices, strict=True):
        if not values:
            raise ValueError(f"categorical feature {dimension} has no candidate values")
        lower = bounds[0, dimension].item()
        upper = bounds[1, dimension].item()
        if any(value < lower or value > upper for value in values):
            raise ValueError(
                f"categorical feature {dimension} contains a value outside bounds "
                f"[{lower}, {upper}]"
            )
    row_assignment_count = 1
    for values in choices:
        row_assignment_count *= len(values)
    assignment_count = row_assignment_count**augmented_q
    if assignment_count > max_assignments:
        raise ValueError(
            f"mixed one-shot assignment space has {assignment_count} combinations; "
            f"limit is {max_assignments}"
        )
    row_assignments = list(product(*choices))
    if constraints is not None and constraints.has_nonlinear_constraints:
        raise NotImplementedError("flattened optimization supports linear constraints only")

    def flatten(entries: tuple[LinearConstraint, ...]) -> list[LinearConstraint]:
        result = []
        for indices, coefficients, rhs in entries:
            if indices.ndim == 1:
                for row in range(q):
                    result.append((indices + row * feature_count, coefficients, rhs))
            else:
                if (indices[:, 0] < 0).any() or (indices[:, 0] >= q).any():
                    raise ValueError("constraint row must refer to an actual candidate")
                result.append((indices[:, 0] * feature_count + indices[:, 1], coefficients, rhs))
        return result

    inequalities = flatten(constraints.inequality_constraints) if constraints else []
    equalities = flatten(constraints.equality_constraints) if constraints else []

    best_augmented: Tensor | None = None
    best_value: Tensor | None = None
    for assignment in product(row_assignments, repeat=augmented_q):
        fixed_features = {
            row * bounds.shape[-1] + dimension: value
            for row, values in enumerate(assignment)
            for dimension, value in zip(dimensions, values, strict=True)
        }

        def flattened_acquisition(x: Tensor) -> Tensor:
            batch_shape = x.shape[:-2]
            augmented = x.reshape(*batch_shape, augmented_q, bounds.shape[-1])
            return acq_function(augmented)

        flat_bounds = bounds.repeat(1, augmented_q)
        flat_candidate, value = optimize_acqf(
            acq_function=flattened_acquisition,
            bounds=flat_bounds,
            q=1,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            fixed_features=fixed_features,
            inequality_constraints=inequalities or None,
            equality_constraints=equalities or None,
        )
        augmented = flat_candidate.reshape(augmented_q, bounds.shape[-1])
        if best_value is None or value.max() > best_value.max():
            best_augmented = augmented
            best_value = value

    if best_augmented is None or best_value is None:
        raise RuntimeError("mixed one-shot optimization produced no candidate")
    candidate = (
        acq_function.extract_candidates(best_augmented.unsqueeze(0)).squeeze(0)
        if one_shot
        else best_augmented
    )
    return candidate, best_value
