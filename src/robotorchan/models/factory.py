"""Resolve public model constructors without changing their training contracts."""

from __future__ import annotations

import inspect
from importlib import import_module
from types import ModuleType

from robotorchan.models.registry import MODEL_REGISTRY

# Public exports that are metadata, likelihoods, posteriors or utilities rather
# than model constructors. Do not infer capabilities from class names.
_NON_MODELS = frozenset(
    {
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
    }
)

# Public constructors outside the two aggregate namespaces. Import only on
# resolution so optional causal dependencies are not required by list_models.
_SPECIAL_MODELS = {
    "BOCSSurrogate": "robotorchan.models.standard.bocs",
    "COMBOSurrogate": "robotorchan.models.standard.combo",
    "RGPEModel": "robotorchan.optim.transfer.rgpe_model",
    "InterventionGPSurrogate": "robotorchan.causal.surrogate",
    "GPStructuralCausalModel": "robotorchan.causal.gp_scm",
    "StructuralCausalModel": "robotorchan.causal.scm",
}


def public_constructors(models: ModuleType, classification: ModuleType) -> dict[str, type]:
    """Collect concrete constructors from the public facade namespaces."""
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


def list_model_names(constructors: dict[str, type]) -> tuple[str, ...]:
    """List accepted public model names and existing classification IDs.

    Listing indicates constructor availability, not verified workflow
    compatibility or the installation of optional dependencies.
    """
    return tuple(sorted(constructors.keys() | _SPECIAL_MODELS.keys()))


def resolve_model_class(name: str, constructors: dict[str, type]) -> type:
    """Resolve an existing public model class by its exact name or registry ID."""
    if name in _SPECIAL_MODELS:
        return getattr(import_module(_SPECIAL_MODELS[name]), name)
    try:
        return constructors[name]
    except KeyError as error:
        raise KeyError(f"Unknown model {name!r}. Use list_models() for available names.") from error
