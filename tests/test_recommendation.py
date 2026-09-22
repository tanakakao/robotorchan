"""Tests for explainable capability-based recommendations."""

from robotorchan.acquisition.registry import ACQUISITION_REGISTRY
from robotorchan.problem import OutputType, ProblemPurpose, ProblemSpec
from robotorchan.recommendation import recommend_compatible_workflows


def test_bo_recommendations_use_registered_botorch_metadata() -> None:
    spec = ProblemSpec(purpose=ProblemPurpose.BAYESIAN_OPTIMIZATION)

    recommendations = recommend_compatible_workflows(spec)

    assert recommendations
    assert all(item.acquisition_name in ACQUISITION_REGISTRY for item in recommendations)


def test_active_learning_recommendations_respect_model_acquisition_compatibility() -> None:
    spec = ProblemSpec(purpose=ProblemPurpose.ACTIVE_LEARNING)

    recommendations = recommend_compatible_workflows(spec)

    assert recommendations
    assert all(item.acquisition_name is not None for item in recommendations)
    assert all(item.rationale for item in recommendations)


def test_default_problem_excludes_special_structural_models() -> None:
    spec = ProblemSpec(purpose=ProblemPurpose.BAYESIAN_OPTIMIZATION)

    names = {item.model_name for item in recommend_compatible_workflows(spec)}

    assert "SingleTaskMultiFidelityGP" not in names
    assert "PairwiseGP" not in names


def test_multi_output_active_learning_excludes_single_output_acquisitions() -> None:
    spec = ProblemSpec(
        purpose=ProblemPurpose.ACTIVE_LEARNING,
        output_type=OutputType.MULTI,
    )

    names = {item.acquisition_name for item in recommend_compatible_workflows(spec)}

    assert "ExpectedPredictiveInformationGain" not in names


def test_multifidelity_kg_is_only_recommended_for_multifidelity_problems() -> None:
    standard = ProblemSpec(purpose=ProblemPurpose.BAYESIAN_OPTIMIZATION)
    multifidelity = ProblemSpec(
        purpose=ProblemPurpose.BAYESIAN_OPTIMIZATION,
        multi_fidelity=True,
    )

    standard_acquisitions = {
        item.acquisition_name for item in recommend_compatible_workflows(standard)
    }
    multifidelity_acquisitions = {
        item.acquisition_name for item in recommend_compatible_workflows(multifidelity)
    }

    assert "qMultiFidelityKnowledgeGradient" not in standard_acquisitions
    assert "qMultiFidelityKnowledgeGradient" in multifidelity_acquisitions


def test_multifidelity_kg_is_not_paired_with_standard_models() -> None:
    spec = ProblemSpec(
        purpose=ProblemPurpose.BAYESIAN_OPTIMIZATION,
        multi_fidelity=True,
    )

    workflows = recommend_compatible_workflows(spec)
    mfkg_models = {
        item.model_name
        for item in workflows
        if item.acquisition_name == "qMultiFidelityKnowledgeGradient"
    }

    assert mfkg_models
    assert "SingleTaskGP" not in mfkg_models
    assert all(
        ACQUISITION_REGISTRY["qMultiFidelityKnowledgeGradient"]
        .capabilities.requires_multi_fidelity
        for _ in mfkg_models
    )
