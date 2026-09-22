"""Tests for the initial capability registry schema."""

from robotorchan.models.capabilities import (
    HighDimensionalStrategy,
    InferenceType,
    InputType,
    RobustnessType,
    TaskType,
)
from robotorchan.models.registry import MODEL_REGISTRY, get_model_registry_entry


def test_representative_models_are_registered() -> None:
    for model_name in MODEL_REGISTRY:
        assert get_model_registry_entry(model_name).model_name == model_name


def test_registry_covers_cross_cutting_capabilities() -> None:
    model = MODEL_REGISTRY["MixedSaasFullyBayesianMultiTaskGP"].capabilities
    assert model.input_type is InputType.MIXED
    assert model.task_type is TaskType.MULTITASK
    assert model.inference is InferenceType.FULLY_BAYESIAN
    assert model.high_dimensional is HighDimensionalStrategy.SAAS

    robust = MODEL_REGISTRY["MixedRobustRelevancePursuitMultiTaskGP"].capabilities
    assert robust.robustness == frozenset({RobustnessType.RELEVANCE_PURSUIT})
    assert MODEL_REGISTRY["SingleTaskMultiFidelityGP"].capabilities.multi_fidelity
    variational = MODEL_REGISTRY["SingleTaskVariationalGP"].capabilities
    assert variational.inference is InferenceType.VARIATIONAL


def test_registry_documentation_paths_are_repository_relative() -> None:
    for entry in MODEL_REGISTRY.values():
        assert entry.documentation.guide.startswith("docs/")
        assert entry.documentation.theory.startswith("docs/theory/")
        assert entry.documentation.notebook.startswith("examples/notebooks/")


def test_registry_covers_every_public_model() -> None:
    import robotorchan.models as models

    expected = set(models.__all__) - {"UnsupportedModelOperationError"}
    assert set(MODEL_REGISTRY) == expected


def test_structured_output_is_not_ordinary_multitask() -> None:
    assert not MODEL_REGISTRY["KroneckerMultiTaskGP"].capabilities.structured_output
    assert MODEL_REGISTRY["HigherOrderGP"].capabilities.structured_output
    assert MODEL_REGISTRY["LatentKroneckerGP"].capabilities.structured_output


def test_registry_model_names_match_keys() -> None:
    assert all(name == entry.model_name for name, entry in MODEL_REGISTRY.items())
