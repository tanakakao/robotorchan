"""Correctness-first optimization for mixed one-shot acquisitions."""

from collections.abc import Sequence
from itertools import product

from botorch.acquisition.acquisition import OneShotAcquisitionFunction
from botorch.optim import optimize_acqf
from torch import Tensor


def optimize_mixed_one_shot_acqf(
    acq_function: OneShotAcquisitionFunction,
    bounds: Tensor,
    *,
    categorical_features: dict[int, Sequence[float]],
    q: int = 1,
    num_restarts: int = 10,
    raw_samples: int = 256,
    max_assignments: int = 256,
) -> tuple[Tensor, Tensor]:
    """Optimize a q=1 one-shot acquisition with row-specific categorical assignments."""
    if q != 1:
        raise NotImplementedError("mixed one-shot optimization currently supports q=1 only")
    augmented_q = acq_function.get_augmented_q_batch_size(q)
    dimensions = sorted(categorical_features)
    choices = [tuple(categorical_features[index]) for index in dimensions]
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
        )
        augmented = flat_candidate.reshape(augmented_q, bounds.shape[-1])
        if best_value is None or value.max() > best_value.max():
            best_augmented = augmented
            best_value = value

    if best_augmented is None or best_value is None:
        raise RuntimeError("mixed one-shot optimization produced no candidate")
    candidate = acq_function.extract_candidates(best_augmented.unsqueeze(0)).squeeze(0)
    return candidate, best_value
