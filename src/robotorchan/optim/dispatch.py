"""BoTorch-shaped convenience dispatch for acquisition optimization."""

from __future__ import annotations

from typing import Any, Literal

from botorch.acquisition.acquisition import AcquisitionFunction
from torch import Tensor

from robotorchan.optim.backends import (
    optimize_acqf_botorch,
    optimize_acqf_cmaes,
    optimize_acqf_de,
    optimize_acqf_ga,
    optimize_acqf_hybrid,
    optimize_acqf_pso,
    optimize_acqf_sampling,
    optimize_acqf_torch,
)
from robotorchan.optim.capabilities import get_optimizer_capabilities
from robotorchan.optim.constraints import CandidateConstraints
from robotorchan.optim.variable_space import MixedVariableSpace

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
    fixed_features: dict[int, float | Tensor] | None = None,
    batch_initial_conditions: Tensor | None = None,
    sequential: bool = False,
    seed: int | None = None,
    optimizer_options: dict[str, Any] | None = None,
    variable_space: MixedVariableSpace | None = None,
) -> tuple[Tensor, Tensor]:
    """Optimize an acquisition function using a named robotorchan backend."""
    backend_options = dict(optimizer_options or {})
    name = optimizer.lower()
    capabilities = get_optimizer_capabilities(name)
    _validate_requested_capabilities(
        name,
        capabilities,
        constraints=constraints,
        fixed_features=fixed_features,
        sequential=sequential,
        variable_space=variable_space,
    )

    if name == "botorch":
        if variable_space is not None and variable_space.is_mixed:
            raise NotImplementedError(
                "optimizer='botorch' does not consume MixedVariableSpace directly; "
                "use the explicit BoTorch mixed backend or a mixed-capable named optimizer."
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
        )
    if name.startswith("torch_"):
        _reject_seed(name, seed)
        _reject_sequential(name, sequential)
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
            fixed_features=fixed_features,
            batch_initial_conditions=batch_initial_conditions,
        )

    _reject_botorch_local_arguments(
        name,
        options=options,
        batch_initial_conditions=batch_initial_conditions,
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
        return optimize_acqf_ga(
            acq_function,
            bounds,
            q,
            seed=seed,
            constraints=constraints,
            fixed_features=fixed_features,
            variable_space=variable_space,
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
    supported = ", ".join(
        (
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
        )
    )
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
) -> None:
    if options is not None:
        raise ValueError(
            f"BoTorch local options are not used by optimizer={name!r}; "
            "use optimizer_options for backend-specific settings."
        )
    if batch_initial_conditions is not None:
        raise ValueError(f"batch_initial_conditions are not used by optimizer={name!r}.")


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
