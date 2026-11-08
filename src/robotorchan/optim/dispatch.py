"""BoTorch-shaped convenience dispatch for acquisition optimization."""

from __future__ import annotations

from typing import Any, Literal

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.optim.initializers import TGenInitialConditions
from torch import Tensor

from robotorchan.optim.backend_support.runtime import validate_bounds
from robotorchan.optim.backends.botorch import (
    optimize_acqf_botorch,
    optimize_acqf_mixed_botorch,
)
from robotorchan.optim.backends.cmaes import optimize_acqf_cmaes
from robotorchan.optim.backends.differential_evolution import optimize_acqf_de
from robotorchan.optim.backends.genetic_algorithm import optimize_acqf_ga
from robotorchan.optim.backends.hybrid import optimize_acqf_hybrid
from robotorchan.optim.backends.mixed_cmaes import optimize_acqf_mixed_cmaes
from robotorchan.optim.backends.mixed_genetic_algorithm import optimize_acqf_mixed_ga
from robotorchan.optim.backends.pso import optimize_acqf_pso
from robotorchan.optim.backends.sampling import optimize_acqf_sampling
from robotorchan.optim.backends.torch import optimize_acqf_torch
from robotorchan.optim.capabilities import (
    BOTORCH_MIXED_OPTIMIZER_CAPABILITIES,
    MIXED_CMAES_OPTIMIZER_CAPABILITIES,
    MIXED_GENETIC_ALGORITHM_OPTIMIZER_CAPABILITIES,
    OPTIMIZER_CAPABILITIES,
    get_optimizer_capabilities,
)
from robotorchan.optim.constraints.contracts import (
    CandidateConstraints,
    LinearConstraint,
    NonlinearConstraint,
    resolve_candidate_constraints,
)
from robotorchan.optim.constraints.validation import validate_candidate_constraints
from robotorchan.optim.domains.variable_space import MixedVariableSpace

OptimizerName = Literal[
    "botorch",
    "torch_adam",
    "torch_adamw",
    "torch_sgd",
    "random",
    "sobol",
    "de",
    "cmaes",
    "ga",
    "pso",
    "hybrid",
]


def optimize_acqf(
    acq_function: AcquisitionFunction,
    bounds: Tensor,
    q: int,
    *,
    optimizer: OptimizerName = "botorch",
    num_restarts: int = 10,
    raw_samples: int | None = 512,
    options: dict[str, Any] | None = None,
    constraints: CandidateConstraints | None = None,
    inequality_constraints: tuple[LinearConstraint, ...] | None = None,
    equality_constraints: tuple[LinearConstraint, ...] | None = None,
    nonlinear_inequality_constraints: tuple[NonlinearConstraint, ...] | None = None,
    nonlinear_equality_constraints: tuple[NonlinearConstraint, ...] | None = None,
    nonlinear_equality_tolerance: float = 1e-6,
    fixed_features: dict[int, float | Tensor] | None = None,
    batch_initial_conditions: Tensor | None = None,
    sequential: bool = False,
    ic_generator: TGenInitialConditions | None = None,
    ic_gen_kwargs: dict[str, Any] | None = None,
    seed: int | None = None,
    optimizer_options: dict[str, Any] | None = None,
    variable_space: MixedVariableSpace | None = None,
) -> tuple[Tensor, Tensor]:
    """Optimize an acquisition function using a named robotorchan backend."""
    constraints = resolve_candidate_constraints(
        constraints,
        inequality_constraints=inequality_constraints,
        equality_constraints=equality_constraints,
        nonlinear_inequality_constraints=nonlinear_inequality_constraints,
        nonlinear_equality_constraints=nonlinear_equality_constraints,
        nonlinear_equality_tolerance=nonlinear_equality_tolerance,
    )
    validate_bounds(bounds)
    if q < 1:
        raise ValueError("q must be at least 1.")
    if variable_space is not None and not torch.equal(variable_space.bounds, bounds):
        raise ValueError("variable_space bounds must match bounds.")
    backend_options = dict(optimizer_options or {})
    validate_candidate_constraints(constraints, bounds=bounds, q=q, sequential=sequential)
    name = optimizer.lower()
    capabilities = get_optimizer_capabilities(name)
    has_discrete = variable_space is not None and bool(
        variable_space.integer_dims or variable_space.categorical_dims
    )
    if name == "cmaes" and has_discrete:
        capabilities = MIXED_CMAES_OPTIMIZER_CAPABILITIES
    if name == "ga" and has_discrete:
        capabilities = MIXED_GENETIC_ALGORITHM_OPTIMIZER_CAPABILITIES
    if name == "botorch" and has_discrete:
        capabilities = BOTORCH_MIXED_OPTIMIZER_CAPABILITIES
    _validate_requested_capabilities(
        name,
        capabilities,
        constraints=constraints,
        fixed_features=fixed_features,
        sequential=sequential,
        variable_space=variable_space,
    )

    if name == "botorch":
        if has_discrete:
            fixed_features_list = backend_options.pop("fixed_features_list", None)
            _reject_backend_options(name, backend_options)
            if fixed_features_list is None:
                raise ValueError(
                    "Mixed BoTorch optimization requires optimizer_options['fixed_features_list']."
                )
            if not isinstance(fixed_features_list, list) or not fixed_features_list:
                raise ValueError("fixed_features_list must be a nonempty list of dictionaries.")
            structured_dims = set(variable_space.integer_dims) | set(
                variable_space.categorical_dims
            )
            resolved_configs = []
            for config in fixed_features_list:
                if not isinstance(config, dict):
                    raise TypeError("Each fixed_features_list entry must be a dictionary.")
                merged = dict(config)
                for dim, value in (fixed_features or {}).items():
                    if dim in merged and float(merged[dim]) != float(value):
                        raise ValueError(f"Conflicting fixed feature at dimension {dim}.")
                    merged[dim] = value
                variable_space.validate_fixed_features(merged)
                if not structured_dims.issubset(merged):
                    raise ValueError(
                        "Each fixed_features_list entry must fix every discrete dimension."
                    )
                resolved_configs.append(merged)
            if sequential:
                raise NotImplementedError(
                    "Mixed BoTorch dispatch does not support sequential optimization."
                )
            _reject_seed(name, seed)
            return optimize_acqf_mixed_botorch(
                acq_function,
                bounds,
                q,
                num_restarts,
                resolved_configs,
                raw_samples,
                options=options,
                constraints=constraints,
                batch_initial_conditions=batch_initial_conditions,
                ic_generator=ic_generator,
                ic_gen_kwargs=ic_gen_kwargs,
            )
        _reject_backend_options(name, backend_options)
        return optimize_acqf_botorch(
            acq_function,
            bounds,
            q,
            num_restarts,
            raw_samples,
            options=options,
            constraints=constraints,
            fixed_features=fixed_features,
            batch_initial_conditions=batch_initial_conditions,
            sequential=sequential,
            ic_generator=ic_generator,
            ic_gen_kwargs=ic_gen_kwargs,
        )
    if name.startswith("torch_"):
        _reject_initial_condition_generator(name, ic_generator, ic_gen_kwargs)
        _reject_seed(name, seed)
        _reject_sequential(name, sequential)
        constraint_mode = backend_options.pop("constraint_mode", "strict")
        _reject_backend_options(name, backend_options)
        return optimize_acqf_torch(
            acq_function,
            bounds,
            q,
            num_restarts,
            raw_samples,
            optimizer=name.removeprefix("torch_"),
            options=options,
            constraints=constraints,
            constraint_mode=constraint_mode,
            fixed_features=fixed_features,
            batch_initial_conditions=batch_initial_conditions,
        )

    _reject_botorch_local_arguments(
        name,
        options=options,
        batch_initial_conditions=batch_initial_conditions,
        ic_generator=ic_generator,
        ic_gen_kwargs=ic_gen_kwargs,
    )
    if sequential:
        raise NotImplementedError(
            "Named derivative-free backends use joint q-batch optimization. "
            "Use optimize_acqf_sequential explicitly when sequential selection is required."
        )

    if name in {"random", "sobol"}:
        num_samples = backend_options.pop("num_samples", raw_samples or 512)
        return optimize_acqf_sampling(
            acq_function,
            bounds,
            q,
            num_samples=num_samples,
            method=name,
            seed=seed,
            constraints=constraints,
            fixed_features=fixed_features,
            variable_space=variable_space,
            **backend_options,
        )
    if name == "de":
        return optimize_acqf_de(
            acq_function,
            bounds,
            q,
            seed=seed,
            constraints=constraints,
            fixed_features=fixed_features,
            variable_space=variable_space,
            **backend_options,
        )
    if name == "ga":
        if has_discrete:
            return optimize_acqf_mixed_ga(
                acq_function,
                bounds,
                q,
                seed=seed,
                constraints=constraints,
                fixed_features=fixed_features,
                variable_space=variable_space,
                **backend_options,
            )
        return optimize_acqf_ga(
            acq_function,
            bounds,
            q,
            seed=seed,
            constraints=constraints,
            fixed_features=fixed_features,
            **backend_options,
        )
    if name == "pso":
        return optimize_acqf_pso(
            acq_function,
            bounds,
            q,
            seed=seed,
            constraints=constraints,
            fixed_features=fixed_features,
            variable_space=variable_space,
            **backend_options,
        )
    if name == "cmaes":
        if has_discrete:
            return optimize_acqf_mixed_cmaes(
                acq_function,
                bounds,
                q,
                seed=seed,
                constraints=constraints,
                fixed_features=fixed_features,
                variable_space=variable_space,
                **backend_options,
            )
        return optimize_acqf_cmaes(
            acq_function,
            bounds,
            q,
            seed=seed,
            constraints=constraints,
            **backend_options,
        )
    if name == "hybrid":
        return optimize_acqf_hybrid(
            acq_function,
            bounds,
            q,
            seed=seed,
            constraints=constraints,
            fixed_features=fixed_features,
            variable_space=variable_space,
            **backend_options,
        )
    supported = ", ".join(sorted(OPTIMIZER_CAPABILITIES))
    raise ValueError(f"Unknown optimizer {optimizer!r}. Supported optimizers: {supported}.")


def _reject_backend_options(name: str, options: dict[str, Any]) -> None:
    if options:
        raise ValueError(f"optimizer_options are not used by optimizer={name!r}.")


def _reject_seed(name: str, seed: int | None) -> None:
    if seed is not None:
        raise ValueError(f"seed is not a direct argument for optimizer={name!r}.")


def _reject_sequential(name: str, sequential: bool) -> None:
    if sequential:
        raise NotImplementedError(f"optimizer={name!r} does not expose sequential dispatch yet.")


def _reject_botorch_local_arguments(
    name: str,
    *,
    options: dict[str, Any] | None,
    batch_initial_conditions: Tensor | None,
    ic_generator: TGenInitialConditions | None,
    ic_gen_kwargs: dict[str, Any] | None,
) -> None:
    if options is not None:
        raise ValueError(
            f"BoTorch local options are not used by optimizer={name!r}; "
            "use optimizer_options for backend-specific settings."
        )
    if batch_initial_conditions is not None:
        raise ValueError(f"batch_initial_conditions are not used by optimizer={name!r}.")
    _reject_initial_condition_generator(name, ic_generator, ic_gen_kwargs)


def _reject_initial_condition_generator(
    name: str,
    ic_generator: TGenInitialConditions | None,
    ic_gen_kwargs: dict[str, Any] | None,
) -> None:
    if ic_generator is not None:
        raise ValueError(f"ic_generator is only used by optimizer='botorch', not {name!r}.")
    if ic_gen_kwargs is not None:
        raise ValueError(f"ic_gen_kwargs are only used by optimizer='botorch', not {name!r}.")


def _validate_requested_capabilities(
    name: str,
    capabilities: Any,
    *,
    constraints: CandidateConstraints | None,
    fixed_features: dict[int, float | Tensor] | None,
    sequential: bool,
    variable_space: MixedVariableSpace | None,
) -> None:
    if variable_space is not None:
        if variable_space.integer_dims and not capabilities.integer:
            raise NotImplementedError(f"optimizer={name!r} does not support integer variables.")
        if variable_space.categorical_dims and not capabilities.categorical:
            raise NotImplementedError(f"optimizer={name!r} does not support categorical variables.")
        if variable_space.is_mixed and not capabilities.mixed:
            raise NotImplementedError(f"optimizer={name!r} does not support mixed variables.")
        variable_space.validate_fixed_features(fixed_features)

    if fixed_features and not capabilities.fixed_features:
        raise NotImplementedError(f"optimizer={name!r} does not support fixed_features.")
    if sequential and not capabilities.sequential:
        raise NotImplementedError(f"optimizer={name!r} does not support sequential optimization.")
    if constraints is None:
        return
    if constraints.inequality_constraints and not capabilities.linear_inequality_constraints:
        raise NotImplementedError(f"optimizer={name!r} does not support linear inequalities.")
    if constraints.equality_constraints and not capabilities.linear_equality_constraints:
        raise NotImplementedError(f"optimizer={name!r} does not support linear equalities.")
    if constraints.nonlinear_inequality_constraints:
        if not capabilities.nonlinear_inequality_constraints:
            raise NotImplementedError(f"optimizer={name!r} does not support nonlinear constraints.")
        if (
            any(
                not is_intrapoint
                for _, is_intrapoint in constraints.nonlinear_inequality_constraints
            )
            and not capabilities.interpoint_nonlinear_constraints
        ):
            raise NotImplementedError(
                f"optimizer={name!r} does not support inter-point nonlinear constraints."
            )
