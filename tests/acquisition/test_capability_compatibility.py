"""Tests for acquisition capability metadata and model compatibility."""

import robotorchan.acquisition as acquisition
from robotorchan.acquisition.capabilities import (
    AcquisitionPurpose,
    PosteriorRequirement,
)
from robotorchan.acquisition.compatibility import (
    CompatibilityStatus,
    check_capabilities_acquisition_compatibility,
    check_model_acquisition_compatibility,
)
from robotorchan.acquisition.registry import ACQUISITION_REGISTRY
from robotorchan.models.classification import get_classification_model_entry


def test_registry_covers_public_acquisition_classes() -> None:
    helpers = {
        "AcquisitionCapabilities",
        "AcquisitionCompositionPlan",
        "AcquisitionPurpose",
        "AcquisitionRegistryEntry",
        "BoTorchMultiObjectiveBridge",
        "BoTorchObjectiveBridge",
        "FeasibilityBinding",
        "ObjectiveBinding",
        "PosteriorRequirement",
        "ProbabilityObjectiveBridge",
        "SampleShapeContract",
        "create_acquisition",
        "get_acquisition_class",
        "get_acquisition_registry_entry",
        "list_acquisitions",
        "make_botorch_multiobjective_bridge",
        "make_botorch_objective_bridge",
        "make_classification_constrained_qei_acquisition",
        "make_classification_feasibility_bridge",
        "make_constrained_qehvi_acquisition",
        "make_continuous_constrained_qei_acquisition",
        "make_continuous_constrained_qnei_acquisition",
        "make_continuous_constraint_bridge",
        "make_deterministic_pof_acquisition",
        "make_mixed_constrained_qei_acquisition",
        "make_multiple_learned_constrained_qnei_acquisition",
        "make_non_gp_acquisition",
        "make_probability_objective_bridge",
        "make_qehvi_acquisition",
        "make_qei_acquisition",
        "make_qlognparego_acquisition",
        "make_qnehvi_acquisition",
        "make_qnei_acquisition",
        "make_sample_classification_feasibility_bridge",
        "resolve_acquisition_composition",
        "select_thompson_candidates",
        "validate_non_gp_acquisition",
    }
    expected = set(acquisition.__all__) - helpers
    public_extensions = {
        name
        for name, entry in ACQUISITION_REGISTRY.items()
        if entry.capabilities.purpose is AcquisitionPurpose.ACTIVE_LEARNING
    }
    bo_extensions = {"qDecoupledKnowledgeGradient", "qDecoupledParticleKnowledgeGradient"}
    assert public_extensions == (expected - bo_extensions) | {
        "PairwiseBayesianActiveLearningByDisagreement"
    }
    for name in bo_extensions:
        assert name in expected
        assert ACQUISITION_REGISTRY[name].capabilities.purpose is (
            AcquisitionPurpose.BAYESIAN_OPTIMIZATION
        )
        assert acquisition.get_acquisition_class(name) is getattr(acquisition, name)
    assert "PairwiseBayesianActiveLearningByDisagreement" not in acquisition.__all__
    assert acquisition.get_acquisition_class("PairwiseBayesianActiveLearningByDisagreement")


def test_single_task_gp_supports_scalar_active_learning() -> None:
    result = check_model_acquisition_compatibility("SingleTaskGP", "qStraddle")
    assert result.status is CompatibilityStatus.COMPATIBLE
    assert result.compatible


def test_structured_output_requires_scalarization() -> None:
    result = check_model_acquisition_compatibility("HigherOrderGP", "qPosteriorVariance")
    assert result.status is CompatibilityStatus.INCOMPATIBLE
    assert "structured-output posteriors require scalarization" in result.reasons


def test_epig_rejects_multitask_model() -> None:
    result = check_model_acquisition_compatibility(
        "KroneckerMultiTaskGP",
        "ExpectedPredictiveInformationGain",
    )
    assert result.status is CompatibilityStatus.INCOMPATIBLE
    assert "acquisition requires a single-output posterior" in result.reasons


def test_random_forest_rejects_variance_active_learning() -> None:
    result = check_model_acquisition_compatibility(
        "RandomForestSurrogate",
        "qPosteriorVariance",
    )
    assert result.status is CompatibilityStatus.INCOMPATIBLE
    assert "acquisition does not support ensemble posteriors" in result.reasons


def test_registry_keys_match_acquisition_names() -> None:
    entries = ACQUISITION_REGISTRY.values()
    registered_names = {entry.acquisition_name for entry in entries}
    assert registered_names == set(ACQUISITION_REGISTRY)


def test_non_gp_model_accepts_registered_monte_carlo_bo_acquisition() -> None:
    result = check_model_acquisition_compatibility(
        "RandomForestSurrogate",
        "qLogExpectedImprovement",
    )
    assert result.status is CompatibilityStatus.COMPATIBLE
    assert result.compatible


def test_qkg_rejects_multi_output_model() -> None:
    result = check_model_acquisition_compatibility(
        "KroneckerMultiTaskGP",
        "qKnowledgeGradient",
    )
    assert result.status is CompatibilityStatus.INCOMPATIBLE
    assert "acquisition does not support multi-output posteriors" in result.reasons


def test_epig_rejects_model_list_multi_output_model() -> None:
    result = check_model_acquisition_compatibility(
        "ModelListGP",
        "ExpectedPredictiveInformationGain",
    )
    assert result.status is CompatibilityStatus.INCOMPATIBLE
    assert "acquisition requires a single-output posterior" in result.reasons


def test_ngboost_supports_marginal_variance_active_learning() -> None:
    result = check_model_acquisition_compatibility("NGBoostSurrogate", "qPosteriorVariance")
    assert result.status is CompatibilityStatus.COMPATIBLE


def test_ngboost_rejects_joint_gaussian_information_gain() -> None:
    result = check_model_acquisition_compatibility(
        "NGBoostSurrogate",
        "ExpectedPredictiveInformationGain",
    )
    assert result.status is CompatibilityStatus.INCOMPATIBLE
    assert "acquisition requires a joint Gaussian posterior" in result.reasons


def test_registry_capabilities_are_internally_consistent() -> None:
    for entry in ACQUISITION_REGISTRY.values():
        capabilities = entry.capabilities

        assert capabilities.max_q is None or capabilities.max_q >= 1
        if capabilities.requires_single_output:
            assert not capabilities.supports_multi_output
        if capabilities.supports_multi_objective:
            assert capabilities.supports_multi_output
        if capabilities.one_shot:
            assert capabilities.requires_fantasize
        if capabilities.monte_carlo:
            assert capabilities.posterior_requirement is PosteriorRequirement.POSTERIOR_SAMPLES


def test_static_compatibility_does_not_imply_runtime_validation() -> None:
    result = check_model_acquisition_compatibility(
        "RandomForestSurrogate",
        "qLogExpectedImprovement",
    )

    assert result.status is CompatibilityStatus.COMPATIBLE
    assert result.reasons == ()


def test_binary_classifier_rejects_regression_active_learning_acquisition() -> None:
    model = get_classification_model_entry("binary.standard").capabilities
    result = check_capabilities_acquisition_compatibility(model, "qStraddle")
    assert result.status is CompatibilityStatus.INCOMPATIBLE
    assert "acquisition does not support classification observations" in result.reasons


def test_binary_classifier_rejects_botorch_regression_bo_acquisition() -> None:
    model = get_classification_model_entry("binary.standard").capabilities
    result = check_capabilities_acquisition_compatibility(model, "qLogExpectedImprovement")
    assert result.status is CompatibilityStatus.INCOMPATIBLE
    assert "acquisition does not support classification observations" in result.reasons


def test_decoupled_al_requires_explicit_heterogeneous_composition() -> None:
    entry = ACQUISITION_REGISTRY["qDecoupledActiveLearning"]
    assert entry.capabilities.requires_heterogeneous_model
    result = check_model_acquisition_compatibility("SingleTaskGP", "qDecoupledActiveLearning")
    assert not result.compatible
    assert "requires explicit HeterogeneousModel and child acquisition validation" in result.reasons


def test_particle_kg_optimizer_capability() -> None:
    from robotorchan.acquisition.optimizer_compatibility import (
        CandidateOptimizer,
        check_acquisition_optimizer_compatibility,
    )

    name = "qDecoupledParticleKnowledgeGradient"
    assert not check_acquisition_optimizer_compatibility(
        name, CandidateOptimizer.CONTINUOUS
    ).compatible
    assert check_acquisition_optimizer_compatibility(
        name, CandidateOptimizer.DERIVATIVE_FREE, q=2
    ).compatible
