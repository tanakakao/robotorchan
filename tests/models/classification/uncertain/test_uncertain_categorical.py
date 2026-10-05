"""Tests for uncertain categorical binary classification."""

import torch

from robotorchan.models.capabilities import InputType, RobustnessType
from robotorchan.models.classification.binary.registry import BINARY_CLASSIFICATION_MODEL_SPECS
from robotorchan.models.classification.binary.uncertain import (
    ClassificationInputUncertaintyType,
    ClassificationUncertaintyIntegration,
    ClassificationUncertaintyTarget,
    UncertainCategoricalBinarySingleTaskGPClassifier,
)


def _model() -> UncertainCategoricalBinarySingleTaskGPClassifier:
    train_x = torch.tensor(
        [[0.0, 0.0], [0.2, 1.0], [0.8, 0.0], [1.0, 1.0]],
        dtype=torch.double,
    )
    train_y = torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.double)
    return UncertainCategoricalBinarySingleTaskGPClassifier(
        train_x,
        train_y,
        uncertain_cat_dim=1,
        cat_dims=[1],
    )


def test_uncertain_categorical_metadata() -> None:
    model = _model()

    assert model.classification_input_uncertainty == frozenset(
        {ClassificationInputUncertaintyType.CATEGORICAL}
    )
    assert (
        model.classification_uncertainty_target is ClassificationUncertaintyTarget.CANDIDATE_INPUTS
    )
    assert (
        model.classification_uncertainty_integration
        is ClassificationUncertaintyIntegration.ANALYTIC
    )


def test_one_hot_category_probability_matches_mixed_prediction() -> None:
    model = _model()
    x = torch.tensor([[0.4, 1.0]], dtype=torch.double)
    standard = model.predict_proba(x)

    uncertain = model.predict_proba_with_category_uncertainty(
        x,
        category_values=torch.tensor([0.0, 1.0], dtype=torch.double),
        category_probabilities=torch.tensor([[0.0, 1.0]], dtype=torch.double),
    )

    torch.testing.assert_close(uncertain, standard)


def test_soft_category_probability_matches_explicit_weighted_prediction() -> None:
    model = _model()
    x = torch.tensor([[0.4, 0.0]], dtype=torch.double)
    values = torch.tensor([0.0, 1.0], dtype=torch.double)
    weights = torch.tensor([[0.25, 0.75]], dtype=torch.double)
    alternatives = torch.tensor([[0.4, 0.0], [0.4, 1.0]], dtype=torch.double)
    expected = (model.predict_proba(alternatives) * weights.squeeze(0).unsqueeze(-1)).sum(dim=0)

    actual = model.predict_proba_with_category_uncertainty(
        x,
        category_values=values,
        category_probabilities=weights,
    )

    torch.testing.assert_close(actual.squeeze(0), expected)


def test_category_marginalization_keeps_continuous_candidate_autograd() -> None:
    model = _model()
    x = torch.tensor([[0.4, 0.0]], dtype=torch.double, requires_grad=True)

    probabilities = model.predict_proba_with_category_uncertainty(
        x,
        category_values=torch.tensor([0.0, 1.0], dtype=torch.double),
        category_probabilities=torch.tensor([[0.5, 0.5]], dtype=torch.double),
    )
    probabilities[..., 1].sum().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad[..., 0]).all()


def test_uncertain_category_dimension_must_be_categorical() -> None:
    train_x = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
    train_y = torch.tensor([0.0, 1.0], dtype=torch.double)

    try:
        UncertainCategoricalBinarySingleTaskGPClassifier(
            train_x,
            train_y,
            uncertain_cat_dim=0,
            cat_dims=[1],
        )
    except ValueError as error:
        assert "categorical dimension" in str(error)
    else:
        raise AssertionError("Expected ValueError for a continuous uncertain_cat_dim.")


def test_registry_advertises_mixed_uncertain_input_capability() -> None:
    specs = {
        model_id: (family, capabilities)
        for model_id, _, family, capabilities in BINARY_CLASSIFICATION_MODEL_SPECS
    }
    family, capabilities = specs["binary.uncertain.categorical_input"]

    assert family == "uncertain"
    assert capabilities.input_type is InputType.MIXED
    assert capabilities.robustness == frozenset({RobustnessType.UNCERTAIN_INPUT})
