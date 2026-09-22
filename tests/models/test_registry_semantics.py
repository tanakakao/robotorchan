"""Semantic contracts for explicit model capability metadata."""

from robotorchan.acquisition.compatibility import (
    check_model_acquisition_compatibility,
)
from robotorchan.models.capabilities import InferenceType, TaskType
from robotorchan.models.registry import MODEL_REGISTRY


def test_deep_gp_inference_is_variational() -> None:
    names = (
        "SingleTaskDeepGP",
        "MultiTaskDeepGP",
        "MixedSingleTaskDeepGP",
        "MixedMultiTaskDeepGP",
    )

    assert all(
        MODEL_REGISTRY[name].capabilities.inference is InferenceType.VARIATIONAL for name in names
    )


def test_lcem_models_are_multitask() -> None:
    for name in ("LCEMGP", "MixedLCEMGP"):
        assert MODEL_REGISTRY[name].capabilities.task_type is TaskType.MULTITASK


def test_non_gp_inference_is_not_applicable() -> None:
    names = (
        "RandomForestSurrogate",
        "ExtraTreesSurrogate",
        "GradientBoostingSurrogate",
        "HistGradientBoostingSurrogate",
    )

    assert all(
        MODEL_REGISTRY[name].capabilities.inference is InferenceType.NOT_APPLICABLE
        for name in names
    )
    assert all(MODEL_REGISTRY[name].capabilities.ensemble_posterior for name in names)


def test_map_saas_ensemble_is_rejected_by_non_ensemble_acquisition() -> None:
    result = check_model_acquisition_compatibility(
        "EnsembleMapSaasSingleTaskGP",
        "PosteriorVariance",
    )

    assert not result.compatible
    assert "acquisition does not support ensemble posteriors" in result.reasons
