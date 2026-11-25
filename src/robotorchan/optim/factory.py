"""Configure existing optimizer entry points without changing their contracts."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from inspect import signature
from typing import Any

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
from robotorchan.optim.backends.nsga2 import optimize_vector_nsga2, optimize_vector_nsga2_mixed
from robotorchan.optim.backends.pso import optimize_acqf_pso
from robotorchan.optim.backends.sampling import optimize_acqf_sampling
from robotorchan.optim.backends.torch import optimize_acqf_torch
from robotorchan.optim.capabilities import OPTIMIZER_CAPABILITIES
from robotorchan.optim.contextual import ContextualOptimizer
from robotorchan.optim.decoupled import optimize_acqf_decoupled
from robotorchan.optim.dispatch import optimize_acqf
from robotorchan.optim.one_shot.mixed import optimize_mixed_one_shot_acqf
from robotorchan.optim.safe import optimize_safe_acqf

# Dedicated APIs retain their actual names and signatures. They are not
# additional scalar dispatcher names or duplicate capability descriptors.
_SPECIALISTS = {
    function.__name__: function
    for function in (
        optimize_acqf_botorch,
        optimize_acqf_decoupled,
        optimize_acqf_mixed_botorch,
        optimize_acqf_cmaes,
        optimize_acqf_de,
        optimize_acqf_ga,
        optimize_acqf_hybrid,
        optimize_acqf_mixed_cmaes,
        optimize_acqf_mixed_ga,
        optimize_acqf_pso,
        optimize_acqf_sampling,
        optimize_acqf_torch,
        optimize_vector_nsga2,
        optimize_vector_nsga2_mixed,
        optimize_mixed_one_shot_acqf,
        optimize_safe_acqf,
        ContextualOptimizer,
    )
}


def list_optimizers() -> tuple[str, ...]:
    """List scalar dispatcher names and dedicated optimizer entry points."""
    return tuple(sorted(OPTIMIZER_CAPABILITIES.keys() | _SPECIALISTS.keys()))


def create_optimizer(optimizer: str | Callable[..., Any], /, **kwargs: Any) -> Callable[..., Any]:
    """Bind defaults to an existing optimizer without running optimization.

    Scalar names use the existing dispatcher, including its mixed-domain
    selection and validation. Dedicated names and custom callables retain
    their own signatures and return types. Call-time keywords override defaults.
    ContextualOptimizer returns its instance when the configured callable is called.
    """
    if isinstance(optimizer, str):
        if optimizer in _SPECIALISTS:
            function = _SPECIALISTS[optimizer]
        elif optimizer.lower() in OPTIMIZER_CAPABILITIES:
            function = optimize_acqf
            if "optimizer" in kwargs:
                raise TypeError("Pass the dispatcher optimizer name only once.")
            kwargs = {"optimizer": optimizer.lower(), **kwargs}
        else:
            raise ValueError(f"Unknown optimizer {optimizer!r}. Use list_optimizers().")
    elif callable(optimizer):
        function = optimizer
    else:
        raise TypeError("optimizer must be a name or callable")
    # Validate known keyword names now, leaving required runtime inputs unbound.
    signature(function).bind_partial(**kwargs)
    return partial(function, **kwargs)
