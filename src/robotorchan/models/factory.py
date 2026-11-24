"""Resolve public model constructors without changing their training contracts."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from importlib import import_module
from typing import Any, TypeVar, overload

from robotorchan.models.registry import MODEL_REGISTRY

T = TypeVar("T")

# Public exports that are metadata, likelihoods, posteriors or utilities rather
# than model constructors. Do not infer capabilities from class names.
_NON_MODELS = frozenset({
    "DocumentationLinks",
    "HighDimensionalStrategy",
    "InferenceType",
    "InputType",
    "ModelCapabilities",
    "ModelRegistryEntry",
    "PosteriorSamplingType",
    "RobustnessType",
    "TaskType",
    "UnsupportedModelOperationError",
    "ClassificationEnsemblePosterior",
    "ClassificationInputUncertaintyType",
    "ClassificationLikelihoodFamily",
    "ClassificationMetadata",
    "ClassificationModelRegistryEntry",
    "ClassificationProbabilityPosterior",
    "ClassificationProbabilityRiskType",
    "ClassificationReliabilityEvaluator",
    "ClassificationRobustnessType",
    "ClassificationUncertaintyIntegration",
    "ClassificationUncertaintyTarget",
    "ContaminatedBernoulliLikelihood",
    "InputDependentLabelNoiseLikelihood",
    "LabelNoiseBernoulliLikelihood",
    "LabelNoiseOrderedProbitLikelihood",
    "LatentOutputStructure",
    "OrderedProbitLikelihood",
    "ProbabilityCalibrator",
    "RobustClassificationMetadata",
    "RobustProbabilityTransform",
    "SplitConformalClassifier",
    "TemperatureScalingCalibrator",
    "UncertainClassificationMetadata",
})

# Public constructors outside the two aggregate namespaces. Import only on
# resolution so optional causal dependencies are not required by list_models.
_SPECIAL_MODELS = {
    "BOCSSurrogate": "robotorchan.models.standard",
    "COMBOSurrogate": "robotorchan.models.standard",
    "RGPEModel": "robotorchan.optim.transfer.rgpe_model",
    "InterventionGPSurrogate": "robotorchan.causal",
    "GPStructuralCausalModel": "robotorchan.causal",
    "StructuralCausalModel": "robotorchan.causal",
}


def _public_constructors() -> dict[str, type]:
    from robotorchan import models
    from robotorchan.models import classification

    constructors: dict[str, type] = {}
    for namespace in (models, classification):
        for name in namespace.__all__:
            if name in _NON_MODELS or name.endswith("Mixin"):
                continue
            candidate = getattr(namespace, name)
            if not isinstance(candidate, type) or inspect.isabstract(candidate):
                continue
            if name in constructors and constructors[name] is not candidate:
                raise RuntimeError(f"Conflicting public model name: {name!r}")
            constructors[name] = candidate

    # Preserve existing classification IDs; class names are public names,
    # not independently maintained aliases or copied capability metadata.
    for name, entry in classification.CLASSIFICATION_MODEL_REGISTRY.items():
        if name in constructors and constructors[name] is not entry.model_class:
            raise RuntimeError(f"Conflicting registered model name: {name!r}")
        constructors[name] = entry.model_class

    # A registry entry without a constructor is a programming error, not a
    # reason to silently omit a supported model from the factory.
    missing = MODEL_REGISTRY.keys() - constructors.keys() - _SPECIAL_MODELS.keys()
    if missing:
        raise RuntimeError(f"Missing registered model constructors: {sorted(missing)}")
    return constructors


def list_models() -> tuple[str, ...]:
    """List accepted public model names and existing classification IDs.

    Listing indicates constructor availability, not verified workflow
    compatibility or the installation of optional dependencies.
    """
    return tuple(sorted(_public_constructors().keys() | _SPECIAL_MODELS.keys()))


def get_model_class(name: str, /) -> type:
    """Resolve an existing public model class by its exact name or registry ID."""
    if name in _SPECIAL_MODELS:
        return getattr(import_module(_SPECIAL_MODELS[name]), name)
    try:
        return _public_constructors()[name]
    except KeyError as error:
        raise KeyError(f"Unknown model {name!r}. Use list_models() for available names.") from error


@overload
def create_model(model: Callable[..., T], /, *args: Any, **kwargs: Any) -> T: ...


@overload
def create_model(model: str, /, *args: Any, **kwargs: Any) -> Any: ...


def create_model(
    model: str | Callable[..., T], /, *args: Any, **kwargs: Any
) -> T | Any:
    """Call the original constructor with unchanged arguments.

    No fit, fallback, data conversion or capability inference is performed.
    User-supplied callables retain their own construction semantics.
    """
    constructor = get_model_class(model) if isinstance(model, str) else model
    if not callable(constructor):
        raise TypeError("model must be a registered name, class or callable")
    return constructor(*args, **kwargs)
