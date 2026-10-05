"""Cross-registry capability-matrix tests for classification models."""

from robotorchan.acquisition.capabilities import (
    AcquisitionPurpose,
    AcquisitionTarget,
    PosteriorRequirement,
)
from robotorchan.acquisition.compatibility import (
    CompatibilityStatus,
    check_capabilities_acquisition_compatibility,
)
from robotorchan.acquisition.registry import ACQUISITION_REGISTRY
from robotorchan.models.capabilities import (
    HighDimensionalStrategy,
    InputType,
    ObservationType,
    PosteriorSamplingType,
    TaskType,
)
from robotorchan.models.classification import CLASSIFICATION_MODEL_REGISTRY

_CLASSIFICATION_ACQUISITIONS = {
    "BALD",
    "LatentStraddle",
    "MarginUncertainty",
    "PredictiveEntropy",
    "ProbabilityVariance",
}


def test_all_classification_models_have_complete_binary_capabilities() -> None:
    for model_id, entry in CLASSIFICATION_MODEL_REGISTRY.items():
        capabilities = entry.capabilities

        assert model_id.startswith("binary.")
        assert entry.num_classes == 2
        assert capabilities.observation_type is ObservationType.CLASSIFICATION
        if capabilities.non_gp and not capabilities.ensemble_posterior:
            assert not capabilities.supports_posterior_samples
            assert capabilities.posterior_sampling_type is PosteriorSamplingType.NONE
        else:
            assert capabilities.supports_posterior_samples
            assert capabilities.posterior_sampling_type is not PosteriorSamplingType.NONE
        if capabilities.ensemble_posterior:
            assert capabilities.non_gp
            assert capabilities.posterior_sampling_type is PosteriorSamplingType.ENSEMBLE
        assert not capabilities.supports_fantasize


def test_structural_model_ids_match_capability_axes() -> None:
    registry = CLASSIFICATION_MODEL_REGISTRY

    assert registry["binary.mixed"].family == "standard"
    assert registry["binary.mixed"].capabilities.input_type is InputType.MIXED
    for model_id in ("binary.multitask", "binary.kronecker_multitask"):
        assert registry[model_id].family == "standard"
        capabilities = registry[model_id].capabilities
        assert capabilities.task_type is TaskType.MULTITASK
        assert capabilities.supports_multi_output

    expected_high_dimensional = {
        "binary.map_saas": HighDimensionalStrategy.MAP_SAAS,
        "binary.saas": HighDimensionalStrategy.SAAS,
        "binary.reduced": HighDimensionalStrategy.REDUCTION,
        "binary.pca": HighDimensionalStrategy.REDUCTION,
        "binary.pls": HighDimensionalStrategy.REDUCTION,
        "binary.random_projection": HighDimensionalStrategy.REDUCTION,
        "binary.alebo": HighDimensionalStrategy.RANDOM_EMBEDDING,
        "binary.joint_encoder": HighDimensionalStrategy.NEURAL_REDUCTION,
        "binary.deep_gp": HighDimensionalStrategy.DEEP,
    }
    for model_id, strategy in expected_high_dimensional.items():
        assert registry[model_id].capabilities.high_dimensional is strategy


def test_classification_acquisition_matrix_is_explicit() -> None:
    entries = {name: ACQUISITION_REGISTRY[name] for name in _CLASSIFICATION_ACQUISITIONS}
    assert set(entries) == _CLASSIFICATION_ACQUISITIONS

    for entry in entries.values():
        capabilities = entry.capabilities
        assert capabilities.purpose is AcquisitionPurpose.ACTIVE_LEARNING
        assert capabilities.observation_types == frozenset({ObservationType.CLASSIFICATION})
        assert capabilities.max_q == 1
        assert not capabilities.supports_multi_output


def test_classification_acquisition_targets_and_posterior_requirements_match() -> None:
    expected = {
        "PredictiveEntropy": (
            AcquisitionTarget.LABEL_UNCERTAINTY,
            PosteriorRequirement.MARGINAL_MOMENTS,
        ),
        "MarginUncertainty": (
            AcquisitionTarget.CLASS_PROBABILITY,
            PosteriorRequirement.MARGINAL_MOMENTS,
        ),
        "ProbabilityVariance": (
            AcquisitionTarget.CLASS_PROBABILITY,
            PosteriorRequirement.POSTERIOR_SAMPLES,
        ),
        "BALD": (
            AcquisitionTarget.LABEL_UNCERTAINTY,
            PosteriorRequirement.POSTERIOR_SAMPLES,
        ),
        "LatentStraddle": (
            AcquisitionTarget.LATENT,
            PosteriorRequirement.MARGINAL_MOMENTS,
        ),
    }

    for name, (target, posterior_requirement) in expected.items():
        capabilities = ACQUISITION_REGISTRY[name].capabilities
        assert capabilities.target is target
        assert capabilities.posterior_requirement is posterior_requirement


def test_single_task_classifiers_accept_all_classification_acquisitions() -> None:
    for model_id, entry in CLASSIFICATION_MODEL_REGISTRY.items():
        capabilities = entry.capabilities
        if capabilities.supports_multi_output or capabilities.non_gp:
            continue
        for acquisition_name in _CLASSIFICATION_ACQUISITIONS:
            result = check_capabilities_acquisition_compatibility(
                capabilities,
                acquisition_name,
            )
            assert result.status is CompatibilityStatus.COMPATIBLE, (
                model_id,
                acquisition_name,
                result.reasons,
            )


def test_multitask_classifiers_are_conservatively_rejected_by_current_al() -> None:
    for model_id in ("binary.multitask", "binary.kronecker_multitask"):
        capabilities = CLASSIFICATION_MODEL_REGISTRY[model_id].capabilities
        for acquisition_name in _CLASSIFICATION_ACQUISITIONS:
            result = check_capabilities_acquisition_compatibility(
                capabilities,
                acquisition_name,
            )
            assert result.status is CompatibilityStatus.INCOMPATIBLE
            assert "acquisition does not support multi-output posteriors" in result.reasons


def test_single_non_gp_classifiers_reject_posterior_dependent_active_learning() -> None:
    posterior_dependent = {"BALD", "ProbabilityVariance", "LatentStraddle"}
    for model_id, entry in CLASSIFICATION_MODEL_REGISTRY.items():
        if not entry.capabilities.non_gp:
            continue
        for acquisition_name in posterior_dependent:
            result = check_capabilities_acquisition_compatibility(
                entry.capabilities,
                acquisition_name,
            )
            assert result.status is CompatibilityStatus.INCOMPATIBLE, (
                model_id,
                acquisition_name,
                result.reasons,
            )
