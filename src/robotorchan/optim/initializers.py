"""Initial-condition helpers for acquisition optimization gaps."""

from __future__ import annotations

from typing import Any

from botorch.acquisition.acquisition import OneShotAcquisitionFunction
from botorch.optim.initializers import gen_batch_initial_conditions
from torch import Tensor


def gen_augmented_one_shot_initial_conditions(
    acq_function: OneShotAcquisitionFunction,
    bounds: Tensor,
    q: int,
    num_restarts: int,
    raw_samples: int,
    fixed_features: dict[int, float] | None = None,
    options: dict[str, Any] | None = None,
    inequality_constraints: list[tuple[Tensor, Tensor, float]] | None = None,
    equality_constraints: list[tuple[Tensor, Tensor, float]] | None = None,
) -> Tensor:
    """Generate standard BoTorch initial conditions over a full one-shot batch.

    This helper is for one-shot acquisitions without a dedicated BoTorch
    initializer, such as ``qMultiStepLookahead``. Knowledge Gradient should
    continue to use BoTorch's specialized KG initializer.

    Args:
        acq_function: One-shot acquisition being optimized.
        bounds: Lower and upper bounds with shape ``2 x d``.
        q: Number of actual candidates requested by the caller.
        num_restarts: Number of multistart optimization restarts.
        raw_samples: Number of raw samples used by BoTorch initialization.
        fixed_features: Optional fixed feature values.
        options: Options forwarded to ``gen_batch_initial_conditions``.
        inequality_constraints: Optional linear inequality constraints.
        equality_constraints: Optional linear equality constraints.

    Returns:
        Initial conditions with shape
        ``num_restarts x acq_function.get_augmented_q_batch_size(q) x d``.
    """
    augmented_q = acq_function.get_augmented_q_batch_size(q=q)
    return gen_batch_initial_conditions(
        acq_function=acq_function,
        bounds=bounds,
        q=augmented_q,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        fixed_features=fixed_features,
        options=options,
        inequality_constraints=inequality_constraints,
        equality_constraints=equality_constraints,
    )
