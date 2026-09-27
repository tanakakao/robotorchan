"""Hybrid global-to-local acquisition optimizer backend."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.optim import optimize_acqf as botorch_optimize_acqf
from torch import Tensor

from robotorchan.optim.backends.cmaes import optimize_acqf_cmaes
from robotorchan.optim.backends.differential_evolution import optimize_acqf_de
from robotorchan.optim.constraints import CandidateConstraints

GlobalOptimizerName = Literal["de", "cmaes"]
GlobalOptimizer = Callable[..., tuple[Tensor, Tensor]]


def optimize_acqf_hybrid(
    acq_function: AcquisitionFunction,
    bounds: Tensor,
    q: int,
    *,
    global_optimizer: GlobalOptimizerName | GlobalOptimizer = "de",
    global_options: dict[str, Any] | None = None,
    num_restarts: int = 1,
    local_options: dict[str, Any] | None = None,
    constraints: CandidateConstraints | None = None,
    fixed_features: dict[int, float | Tensor] | None = None,
    seed: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Run a derivative-free global search followed by BoTorch local refinement.

    The global stage discovers a promising joint q-batch. The resulting candidate
    is then supplied to BoTorch as an explicit batch initial condition, preserving
    BoTorch's constrained local-optimization semantics.
    """
    if q < 1:
        raise ValueError("q must be at least 1.")
    if num_restarts < 1:
        raise ValueError("num_restarts must be at least 1.")
    if fixed_features:
        raise NotImplementedError(
            "Hybrid optimization does not support fixed_features in the global stage yet."
        )

    optimizer = _resolve_global_optimizer(global_optimizer)
    resolved_global_options = dict(global_options or {})
    resolved_global_options.setdefault("constraints", constraints)
    resolved_global_options.setdefault("seed", seed)
    global_candidate, _ = optimizer(
        acq_function,
        bounds,
        q,
        **resolved_global_options,
    )

    initial_conditions = global_candidate.unsqueeze(0)
    if num_restarts > 1:
        raise ValueError(
            "num_restarts > 1 requires distinct global seeds and is not supported yet."
        )
    options = dict(local_options or {})
    candidate_constraints = constraints or CandidateConstraints()
    if candidate_constraints.has_nonlinear_constraints:
        options.setdefault("batch_limit", 1)

    return botorch_optimize_acqf(
        acq_function=acq_function,
        bounds=bounds,
        q=q,
        num_restarts=num_restarts,
        raw_samples=None,
        options=options,
        inequality_constraints=list(candidate_constraints.inequality_constraints) or None,
        equality_constraints=list(candidate_constraints.equality_constraints) or None,
        nonlinear_inequality_constraints=(
            list(candidate_constraints.nonlinear_inequality_constraints) or None
        ),
        fixed_features=fixed_features,
        batch_initial_conditions=initial_conditions,
    )


def _resolve_global_optimizer(
    optimizer: GlobalOptimizerName | GlobalOptimizer,
) -> GlobalOptimizer:
    if callable(optimizer) and not isinstance(optimizer, str):
        return optimizer
    if optimizer == "de":
        return optimize_acqf_de
    if optimizer == "cmaes":
        return optimize_acqf_cmaes
    raise ValueError("global_optimizer must be 'de', 'cmaes', or a compatible callable.")
