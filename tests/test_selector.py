"""Tests for capability-based model selection."""

from robotorchan.models.capabilities import (
    DocumentationLinks,
    HighDimensionalStrategy,
    InputType,
    ModelCapabilities,
    ModelRegistryEntry,
    RobustnessType,
    TaskType,
)
from robotorchan.problem import ProblemPurpose, ProblemSpec
from robotorchan.selector import evaluate_models, select_compatible_models


def _entry(name: str, capabilities: ModelCapabilities) -> ModelRegistryEntry:
    docs = DocumentationLinks("guide", "theory", "notebook")
    return ModelRegistryEntry(name, capabilities, docs, "test model")


def test_selector_filters_by_declared_capabilities() -> None:
    registry = {
        "plain": _entry("plain", ModelCapabilities()),
        "mixed": _entry(
            "mixed",
            ModelCapabilities(input_type=InputType.MIXED),
        ),
    }
    spec = ProblemSpec(
        purpose=ProblemPurpose.BAYESIAN_OPTIMIZATION,
        input_type=InputType.MIXED,
    )

    assert select_compatible_models(spec, registry) == ("mixed",)


def test_selector_combines_multitask_high_dimensional_and_robust_requirements() -> None:
    capable = ModelCapabilities(
        task_type=TaskType.MULTITASK,
        high_dimensional=HighDimensionalStrategy.SAAS,
        robustness=frozenset({RobustnessType.RELEVANCE_PURSUIT}),
    )
    registry = {
        "plain": _entry("plain", ModelCapabilities()),
        "capable": _entry("capable", capable),
    }
    spec = ProblemSpec(
        purpose=ProblemPurpose.ACTIVE_LEARNING,
        task_type=TaskType.MULTITASK,
        high_dimensional=True,
        robust=True,
    )

    assert select_compatible_models(spec, registry) == ("capable",)


def test_evaluation_explains_unmet_requirements() -> None:
    registry = {"plain": _entry("plain", ModelCapabilities())}
    spec = ProblemSpec(
        purpose=ProblemPurpose.BAYESIAN_OPTIMIZATION,
        multi_fidelity=True,
        structured_output=True,
        high_dimensional=True,
        robust=True,
    )

    result = evaluate_models(spec, registry)[0]

    assert not result.compatible
    assert result.reasons == (
        "requires multi-fidelity support",
        "requires structured-output support",
        "requires a high-dimensional strategy",
        "requires an explicit robustness strategy",
    )


def test_selector_does_not_rank_compatible_models() -> None:
    registry = {
        "first": _entry("first", ModelCapabilities()),
        "second": _entry("second", ModelCapabilities()),
    }
    spec = ProblemSpec(purpose=ProblemPurpose.BAYESIAN_OPTIMIZATION)

    assert select_compatible_models(spec, registry) == ("first", "second")
