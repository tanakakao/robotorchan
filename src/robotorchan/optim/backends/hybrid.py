"""Hybrid global-to-local acquisition optimizer backend."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.optim import optimize_acqf as botorch_optimize_acqf
from torch import Tensor

from robotorchan.optim.backends.botorch import optimize_acqf_mixed_botorch
from robotorchan.optim.backends.cmaes import optimize_acqf_cmaes
from robotorchan.optim.backends.differential_evolution import optimize_acqf_de
from robotorchan.optim.backends.mixed_genetic_algorithm import optimize_acqf_mixed_ga
from robotorchan.optim.constraints import CandidateConstraints
from robotorchan.optim.variable_space import MixedVariableSpace

GlobalOptimizerName = Literal["de", "cmaes", "mixed_ga"]
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
    variable_space: MixedVariableSpace | None = None,
    mixed_fixed_features_list: list[dict[int, float]] | None = None,
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
    if variable_space is not None:
        variable_space.validate_fixed_features(fixed_features)
        if mixed_fixed_features_list is not None:
            for config in mixed_fixed_features_list:
                variable_space.validate_fixed_features(config)
        if variable_space.categorical_dims and mixed_fixed_features_list is None:
            raise ValueError(
                "Categorical hybrid optimization requires mixed_fixed_features_list "
                "for the BoTorch mixed local stage."
            )

    optimizer = _resolve_global_optimizer(global_optimizer)
    resolved_global_options = dict(global_options or {})
    resolved_global_options.setdefault("constraints", constraints)
    resolved_global_options.setdefault("seed", seed)
    if variable_space is not None and global_optimizer != "cmaes":
        resolved_global_options.setdefault("variable_space", variable_space)
    if fixed_features is not None:
        resolved_global_options.setdefault("fixed_features", fixed_features)
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

    if variable_space is not None and variable_space.categorical_dims:
        resolved_fixed_features_list = _merge_fixed_features_list(
            mixed_fixed_features_list or [], fixed_features
        )
        return optimize_acqf_mixed_botorch(
            acq_function=acq_function,
            bounds=bounds,
            q=q,
            num_restarts=num_restarts,
            fixed_features_list=resolved_fixed_features_list,
            raw_samples=None,
            options=options,
            constraints=candidate_constraints,
            batch_initial_conditions=initial_conditions,
        )

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
    if optimizer == "mixed_ga":
        return optimize_acqf_mixed_ga
    raise ValueError(
        "global_optimizer must be 'de', 'cmaes', 'mixed_ga', or a compatible callable."
    )


def _merge_fixed_features_list(
    fixed_features_list: list[dict[int, float]],
    fixed_features: dict[int, float | Tensor] | None,
) -> list[dict[int, float | Tensor]]:
    """Merge common fixed features into every mixed discrete configuration."""
    if not fixed_features:
        return [dict(config) for config in fixed_features_list]
    merged: list[dict[int, float | Tensor]] = []
    for config in fixed_features_list:
        resolved: dict[int, float | Tensor] = dict(config)
        for dim, value in fixed_features.items():
            if dim in resolved:
                scalar = (
                    float(value.reshape(()).item()) if isinstance(value, Tensor) else float(value)
                )
                if float(resolved[dim]) != scalar:
                    raise ValueError(
                        f"fixed_features[{dim}] conflicts with mixed_fixed_features_list."
                    )
            resolved[dim] = value
        merged.append(resolved)
    return merged
