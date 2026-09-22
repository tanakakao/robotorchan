"""Tests for model output arity and fantasy-model capability semantics."""

from robotorchan.acquisition.compatibility import (
    CompatibilityStatus,
    check_model_acquisition_compatibility,
)
from robotorchan.models.capabilities import ModelCapabilities, TaskType
from robotorchan.problem import OutputType, ProblemPurpose, ProblemSpec
from robotorchan.selector import select_compatible_models


def test_multi_output_problem_excludes_single_output_non_gp_models() -> None:
    spec = ProblemSpec(
        purpose=ProblemPurpose.BAYESIAN_OPTIMIZATION,
        output_type=OutputType.MULTI,
    )

    names = set(select_compatible_models(spec))

    assert "RandomForestSurrogate" not in names
    assert "ExtraTreesSurrogate" not in names


def test_multitask_multi_output_model_remains_selectable() -> None:
    spec = ProblemSpec(
        purpose=ProblemPurpose.BAYESIAN_OPTIMIZATION,
        task_type=TaskType.MULTITASK,
        output_type=OutputType.MULTI,
    )

    assert "KroneckerMultiTaskGP" in select_compatible_models(spec)


def test_qkg_rejects_non_fantasizing_non_gp_model() -> None:
    result = check_model_acquisition_compatibility(
        "RandomForestSurrogate",
        "qKnowledgeGradient",
    )

    assert result.status is CompatibilityStatus.INCOMPATIBLE
    assert "acquisition requires fantasy-model support" in result.reasons


def test_posterior_sampling_is_opt_in() -> None:
    assert not ModelCapabilities().supports_posterior_samples


def test_fantasy_support_is_opt_in() -> None:
    assert not ModelCapabilities().supports_fantasize


def test_registered_gp_declares_sampling_and_fantasy_support() -> None:
    from robotorchan.models.registry import MODEL_REGISTRY

    entry = MODEL_REGISTRY["SingleTaskGP"]
    original = entry.capabilities
    assert original.supports_posterior_samples
    assert original.supports_fantasize


def test_single_task_gp_supports_qkg_fantasy_requirement() -> None:
    result = check_model_acquisition_compatibility("SingleTaskGP", "qKnowledgeGradient")

    assert result.status is CompatibilityStatus.COMPATIBLE
