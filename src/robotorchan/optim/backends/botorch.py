"""BoTorch-native acquisition optimizer backends.

These functions deliberately preserve BoTorch's acquisition-optimization
vocabulary. They are thin delegation points used by robotorchan search
strategies and, later, by the convenience optimizer dispatcher.
"""

from __future__ import annotations

from typing import Any

from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.optim import optimize_acqf as botorch_optimize_acqf
from botorch.optim import optimize_acqf_mixed as botorch_optimize_acqf_mixed
from botorch.optim.initializers import TGenInitialConditions
from torch import Tensor

from robotorchan.optim.constraints import CandidateConstraints


def optimize_acqf_botorch(
    acq_function: AcquisitionFunction,
    bounds: Tensor,
    q: int,
    num_restarts: int,
    raw_samples: int | None,
    *,
    options: dict[str, Any] | None = None,
    constraints: CandidateConstraints | None = None,
    fixed_features: dict[int, float] | None = None,
    batch_initial_conditions: Tensor | None = None,
    sequential: bool = False,
    ic_generator: TGenInitialConditions | None = None,
    ic_gen_kwargs: dict[str, Any] | None = None,
) -> tuple[Tensor, Tensor]:
    """Delegate original-space acquisition optimization to BoTorch."""
    candidate_constraints = constraints or CandidateConstraints()
    resolved_options = _options_for_nonlinear_constraints(
        options,
        candidate_constraints,
        batch_initial_conditions,
        ic_generator,
    )
    return botorch_optimize_acqf(
        acq_function=acq_function,
        bounds=bounds,
        q=q,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        options=resolved_options,
        inequality_constraints=_or_none(candidate_constraints.inequality_constraints),
        equality_constraints=_or_none(candidate_constraints.equality_constraints),
        nonlinear_inequality_constraints=_or_none(
            candidate_constraints.nonlinear_inequality_constraints
        ),
        fixed_features=fixed_features,
        batch_initial_conditions=batch_initial_conditions,
        sequential=sequential,
        ic_generator=ic_generator,
        ic_gen_kwargs=ic_gen_kwargs,
    )


def optimize_acqf_mixed_botorch(
    acq_function: AcquisitionFunction,
    bounds: Tensor,
    q: int,
    num_restarts: int,
    fixed_features_list: list[dict[int, float]],
    raw_samples: int | None,
    *,
    options: dict[str, Any] | None = None,
    constraints: CandidateConstraints | None = None,
    batch_initial_conditions: Tensor | None = None,
    ic_generator: TGenInitialConditions | None = None,
    ic_gen_kwargs: dict[str, Any] | None = None,
) -> tuple[Tensor, Tensor]:
    """Delegate enumerated mixed-space acquisition optimization to BoTorch."""
    candidate_constraints = constraints or CandidateConstraints()
    if any(
        not is_intrapoint
        for _, is_intrapoint in candidate_constraints.nonlinear_inequality_constraints
    ):
        raise ValueError(
            "BoTorch mixed acquisition optimization does not support "
            "inter-point nonlinear constraints."
        )
    resolved_options = _options_for_nonlinear_constraints(
        options,
        candidate_constraints,
        batch_initial_conditions,
        ic_generator,
    )
    return botorch_optimize_acqf_mixed(
        acq_function=acq_function,
        bounds=bounds,
        q=q,
        num_restarts=num_restarts,
        fixed_features_list=fixed_features_list,
        raw_samples=raw_samples,
        options=resolved_options,
        inequality_constraints=_or_none(candidate_constraints.inequality_constraints),
        equality_constraints=_or_none(candidate_constraints.equality_constraints),
        nonlinear_inequality_constraints=_or_none(
            candidate_constraints.nonlinear_inequality_constraints
        ),
        batch_initial_conditions=batch_initial_conditions,
        ic_generator=ic_generator,
        ic_gen_kwargs=ic_gen_kwargs,
    )


def _options_for_nonlinear_constraints(
    options: dict[str, Any] | None,
    constraints: CandidateConstraints,
    batch_initial_conditions: Tensor | None,
    ic_generator: TGenInitialConditions | None,
) -> dict[str, Any] | None:
    if not constraints.has_nonlinear_constraints:
        return None if options is None else dict(options)
    if batch_initial_conditions is None and ic_generator is None:
        raise ValueError(
            "Nonlinear candidate constraints require feasible batch_initial_conditions "
            "or an ic_generator."
        )
    resolved = {} if options is None else dict(options)
    resolved.setdefault("batch_limit", 1)
    return resolved


def _or_none(values: tuple[Any, ...]) -> list[Any] | None:
    return list(values) if values else None
