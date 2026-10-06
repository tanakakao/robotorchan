"""Regression guards for classification-specific capability extensions."""

from robotorchan.acquisition.compatibility import (
    CompatibilityStatus,
    check_model_acquisition_compatibility,
)
from robotorchan.models.capabilities import ObservationType, PosteriorSamplingType
from robotorchan.models.registry import MODEL_REGISTRY


def test_regression_registry_does_not_advertise_probability_samples() -> None:
    for entry in MODEL_REGISTRY.values():
        capabilities = entry.capabilities

        assert capabilities.observation_type is ObservationType.REGRESSION
        assert not capabilities.supports_probability_samples


def test_single_task_gp_keeps_gaussian_posterior_contract() -> None:
    capabilities = MODEL_REGISTRY["SingleTaskGP"].capabilities

    assert capabilities.supports_posterior_samples
    assert capabilities.posterior_sampling_type is PosteriorSamplingType.GAUSSIAN
    assert capabilities.supports_fantasize


def test_regression_non_gp_does_not_use_classification_probability_exception() -> None:
    result = check_model_acquisition_compatibility(
        "RandomForestSurrogate",
        "ExpectedPredictiveInformationGain",
    )

    assert result.status is CompatibilityStatus.INCOMPATIBLE
    assert "non-GP models require BoTorch Monte Carlo acquisitions" in result.reasons
    assert "acquisition requires a joint Gaussian posterior" in result.reasons


def test_regression_non_gp_monte_carlo_bo_remains_compatible() -> None:
    result = check_model_acquisition_compatibility(
        "RandomForestSurrogate",
        "qLogExpectedImprovement",
    )

    assert result.status is CompatibilityStatus.COMPATIBLE
    assert result.reasons == ()


def test_multitask_regression_output_contract_is_unchanged() -> None:
    capabilities = MODEL_REGISTRY["KroneckerMultiTaskGP"].capabilities

    assert capabilities.observation_type is ObservationType.REGRESSION
    assert capabilities.supports_multi_output
    assert capabilities.supports_posterior_samples
    assert not capabilities.supports_probability_samples
