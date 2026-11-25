"""Resolve acquisition constructors without wrapping their implementations."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from importlib import import_module
from typing import Any, TypeVar, overload

from botorch.acquisition.acquisition import AcquisitionFunction

from robotorchan.acquisition.registry import ACQUISITION_REGISTRY

# Public BoTorch modules used by the library. Discover concrete classes defined
# in these modules, not imported base classes or private implementation details.
_NATIVE_MODULES = (
    "analytic",
    "monte_carlo",
    "logei",
    "knowledge_gradient",
    "multi_step_lookahead",
    "max_value_entropy_search",
    "joint_entropy_search",
    "predictive_entropy_search",
    "preference",
    "active_learning",
    "multi_objective.analytic",
    "multi_objective.monte_carlo",
    "multi_objective.logei",
    "multi_objective.hypervolume_knowledge_gradient",
    "multi_objective.parego",
    "multi_objective.joint_entropy_search",
    "multi_objective.predictive_entropy_search",
    "multi_objective.max_value_entropy_search",
)
_LOCAL_MODULES = (
    "active_learning.classification",
    "active_learning.epig",
    "active_learning.joint_entropy",
    "active_learning.randomized_straddle",
    "active_learning.straddle",
    "active_learning.variance",
    "classification_constraints",
)
_BASE_NAMES = frozenset(
    {
        "AnalyticAcquisitionFunction",
        "MultiObjectiveAnalyticAcquisitionFunction",
        "MultiObjectiveMCAcquisitionFunction",
        "MCAcquisitionFunction",
        "SampleReducingMCAcquisitionFunction",
        "LogImprovementMCAcquisitionFunction",
        "MultiObjectiveLogImprovementMCAcquisitionFunction",
        "MaxValueBase",
    }
)

T = TypeVar("T")


def _constructors() -> dict[str, type[AcquisitionFunction]]:
    constructors: dict[str, type[AcquisitionFunction]] = {}
    for prefix, modules in (
        ("botorch.acquisition", _NATIVE_MODULES),
        ("robotorchan.acquisition", _LOCAL_MODULES),
    ):
        for suffix in modules:
            module = import_module(f"{prefix}.{suffix}")
            for name, value in vars(module).items():
                if (
                    name.startswith("_")
                    or name in _BASE_NAMES
                    or not inspect.isclass(value)
                    or not issubclass(value, AcquisitionFunction)
                    or inspect.isabstract(value)
                    or value.__module__ != module.__name__
                    or value.__name__ != name
                ):
                    continue
                if name in constructors and constructors[name] is not value:
                    raise RuntimeError(f"Conflicting acquisition name: {name!r}")
                constructors[name] = value
    missing = ACQUISITION_REGISTRY.keys() - constructors.keys()
    if missing:
        raise RuntimeError(f"Missing registered acquisition constructors: {sorted(missing)}")
    return constructors


def list_acquisitions() -> tuple[str, ...]:
    """List constructor names, independently of verified workflow compatibility."""
    return tuple(sorted(_constructors()))


def get_acquisition_class(name: str, /) -> type[AcquisitionFunction]:
    """Return the original acquisition class by its exact name."""
    try:
        return _constructors()[name]
    except KeyError as error:
        raise KeyError(
            f"Unknown acquisition {name!r}. Use list_acquisitions() for available names."
        ) from error


@overload
def create_acquisition(acquisition: Callable[..., T], /, *args: Any, **kwargs: Any) -> T: ...


@overload
def create_acquisition(acquisition: str, /, *args: Any, **kwargs: Any) -> AcquisitionFunction: ...


def create_acquisition(
    acquisition: str | Callable[..., T], /, *args: Any, **kwargs: Any
) -> AcquisitionFunction | T:
    """Forward constructor arguments unchanged; do not fit or select a sampler."""
    constructor = (
        get_acquisition_class(acquisition) if isinstance(acquisition, str) else acquisition
    )
    if not callable(constructor):
        raise TypeError("acquisition must be a registered name, class or callable")
    return constructor(*args, **kwargs)
