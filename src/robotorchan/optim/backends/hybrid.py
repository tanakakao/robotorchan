"""Hybrid global-to-local acquisition optimizer backend."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from botorch.acquisition.acquisition import AcquisitionFunction
from torch import Tensor

from robotorchan.optim.backends.botorch import (
    optimize_acqf_botorch,
    optimize_acqf_mixed_botorch,
)
from robotorchan.optim.backends.cmaes import optimize_acqf_cmaes
from robotorchan.optim.backends.differential_evolution import optimize_acqf_de
from robotorchan.optim.backends.mixed_cmaes import optimize_acqf_mixed_cmaes
from robotorchan.optim.backends.mixed_genetic_algorithm import optimize_acqf_mixed_ga
from robotorchan.optim.backends.pso import optimize_acqf_pso
from robotorchan.optim.constraints.contracts import (
    CandidateConstraints,
    LinearConstraint,
    NonlinearConstraint,
    resolve_candidate_constraints,
)
from robotorchan.optim.constraints.evaluation import candidate_is_feasible
from robotorchan.optim.domains.variable_space import MixedVariableSpace

GlobalOptimizerName = Literal["de", "cmaes", "mixed_ga", "pso"]
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
    inequality_constraints: tuple[LinearConstraint, ...] | None = None,
    equality_constraints: tuple[LinearConstraint, ...] | None = None,
    nonlinear_inequality_constraints: tuple[NonlinearConstraint, ...] | None = None,
    nonlinear_equality_constraints: tuple[NonlinearConstraint, ...] | None = None,
    nonlinear_equality_tolerance: float = 1e-6,
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
    candidate_constraints = resolve_candidate_constraints(
        constraints,
        inequality_constraints=inequality_constraints,
        equality_constraints=equality_constraints,
        nonlinear_inequality_constraints=nonlinear_inequality_constraints,
        nonlinear_equality_constraints=nonlinear_equality_constraints,
        nonlinear_equality_tolerance=nonlinear_equality_tolerance,
    )
    constraints = candidate_constraints
    if q < 1:
        raise ValueError("q must be at least 1.")
    if num_restarts < 1:
        raise ValueError("num_restarts must be at least 1.")
    if num_restarts > 1:
        raise ValueError(
            "num_restarts > 1 requires distinct global seeds and is not supported yet."
        )
    if variable_space is not None:
        variable_space.validate_fixed_features(fixed_features)
        if mixed_fixed_features_list is not None:
            for config in mixed_fixed_features_list:
                variable_space.validate_fixed_features(config)
        structured_dims = set(variable_space.integer_dims) | set(variable_space.categorical_dims)
        if structured_dims:
            if not mixed_fixed_features_list:
                raise ValueError(
                    "Structured hybrid optimization requires mixed_fixed_features_list "
                    "for the BoTorch mixed local stage."
                )
            for config in mixed_fixed_features_list:
                missing = structured_dims - set(config) - set(fixed_features or {})
                if missing:
                    raise ValueError(
                        "Each mixed_fixed_features_list entry must fix every structured "
                        f"dimension; missing {sorted(missing)}."
                    )

    has_discrete = variable_space is not None and bool(
        variable_space.integer_dims or variable_space.categorical_dims
    )
    if (
        has_discrete
        and constraints is not None
        and any(
            not is_intrapoint for _, is_intrapoint in constraints.nonlinear_inequality_constraints
        )
    ):
        raise ValueError(
            "BoTorch mixed acquisition optimization does not support "
            "inter-point nonlinear constraints."
        )
    if (
        has_discrete
        and q > 1
        and any(
            indices.ndim == 2
            for indices, _, _ in (
                *candidate_constraints.inequality_constraints,
                *candidate_constraints.equality_constraints,
            )
        )
    ):
        raise ValueError(
            "Mixed Hybrid q-batch optimization cannot preserve inter-point linear "
            "constraints during sequential BoTorch mixed local refinement."
        )
    optimizer = (
        optimize_acqf_mixed_cmaes
        if global_optimizer == "cmaes" and has_discrete
        else _resolve_global_optimizer(global_optimizer)
    )
    resolved_global_options = dict(global_options or {})
    if any(
        name in resolved_global_options
        for name in (
            "constraints",
            "equality_constraints",
            "inequality_constraints",
            "nonlinear_inequality_constraints",
            "nonlinear_equality_constraints",
            "nonlinear_equality_tolerance",
        )
    ):
        raise ValueError(
            "Specify candidate constraints on optimize_acqf_hybrid, not in global_options."
        )
    resolved_global_options["constraints"] = candidate_constraints
    resolved_global_options.setdefault("seed", seed)
    if variable_space is not None and (global_optimizer != "cmaes" or has_discrete):
        resolved_global_options.setdefault("variable_space", variable_space)
    if fixed_features is not None:
        if global_optimizer == "cmaes" and not has_discrete:
            raise NotImplementedError(
                "Continuous CMA-ES global search does not support fixed_features."
            )
        resolved_global_options.setdefault("fixed_features", fixed_features)
    global_candidate, global_value = optimizer(
        acq_function,
        bounds,
        q,
        **resolved_global_options,
    )

    if candidate_constraints.has_constraints and not bool(
        candidate_is_feasible(
            global_candidate.unsqueeze(0),
            candidate_constraints,
            equality_tolerance=resolved_global_options.get("equality_tolerance", 1e-6),
        ).all()
    ):
        raise ValueError(
            "Hybrid global optimizer returned a candidate that violates candidate constraints."
        )

    initial_conditions = global_candidate.unsqueeze(0)
    options = dict(local_options or {})

    if candidate_constraints.has_nonlinear_constraints:
        options.setdefault("batch_limit", 1)

    if variable_space is not None and (
        variable_space.integer_dims or variable_space.categorical_dims
    ):
        resolved_fixed_features_list = _merge_fixed_features_list(
            mixed_fixed_features_list or [], fixed_features
        )
        local_candidate, local_value = optimize_acqf_mixed_botorch(
            acq_function=acq_function,
            bounds=bounds,
            q=q,
            num_restarts=num_restarts,
            fixed_features_list=resolved_fixed_features_list,
            raw_samples=None,
            options=options,
            constraints=candidate_constraints,
            batch_initial_conditions=(
                initial_conditions if q == 1 else initial_conditions[:, :1, :]
            ),
        )
    else:
        local_candidate, local_value = optimize_acqf_botorch(
            acq_function=acq_function,
            bounds=bounds,
            q=q,
            num_restarts=num_restarts,
            raw_samples=None,
            options=options,
            constraints=candidate_constraints,
            fixed_features=fixed_features,
            batch_initial_conditions=initial_conditions,
        )

    if candidate_constraints.has_constraints and not bool(
        candidate_is_feasible(
            local_candidate.unsqueeze(0),
            candidate_constraints,
            equality_tolerance=resolved_global_options.get("equality_tolerance", 1e-6),
        ).all()
    ):
        return global_candidate, global_value
    return local_candidate, local_value


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
    if optimizer == "pso":
        return optimize_acqf_pso
    raise ValueError(
        "global_optimizer must be 'de', 'cmaes', 'mixed_ga', 'pso', or a compatible callable."
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
