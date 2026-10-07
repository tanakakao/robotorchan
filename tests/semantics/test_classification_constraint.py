"""Tests for classification constraint semantics."""

import pytest
import torch

from robotorchan.acquisition.classification_constraints import (
    ClassificationProbabilityOfFeasibility,
)
from robotorchan.models.classification.binary.standard.single_task import (
    BinarySingleTaskGPClassifier,
)
from robotorchan.models.heterogeneous import HeterogeneousModel
from robotorchan.models.standard.single_task import SingleTaskGP
from robotorchan.semantics import ClassificationConstraint


def _model() -> HeterogeneousModel:
    train_X = torch.rand(6, 2, dtype=torch.double)
    return HeterogeneousModel(
        SingleTaskGP(train_X, torch.rand(6, 1, dtype=torch.double)),
        BinarySingleTaskGPClassifier(
            train_X,
            torch.tensor([0, 1, 0, 1, 0, 1]),
        ),
        output_names=["strength", "pass"],
    )


def test_classification_constraint_resolves_classification_output() -> None:
    model = _model()
    constraint = ClassificationConstraint(output="pass", feasible_class=1)

    assert constraint.resolve_output(model) == 1
    assert constraint.resolve_owner(model) == (1, 0)


def test_classification_constraint_rejects_regression_output() -> None:
    model = _model()

    with pytest.raises(TypeError, match="requires a classification output"):
        ClassificationConstraint(output="strength").resolve_output(model)


@pytest.mark.parametrize("feasible_class", [True, 1.0, "1"])
def test_classification_constraint_requires_integer_class(
    feasible_class: object,
) -> None:
    model = _model()

    with pytest.raises(TypeError, match="must be an integer"):
        ClassificationConstraint(
            output="pass",
            feasible_class=feasible_class,  # type: ignore[arg-type]
        ).resolve_output(model)


@pytest.mark.parametrize("feasible_class", [-1, 2])
def test_classification_constraint_validates_class_range(feasible_class: int) -> None:
    model = _model()

    with pytest.raises(ValueError, match="outside the classifier class range"):
        ClassificationConstraint(
            output="pass",
            feasible_class=feasible_class,
        ).resolve_output(model)


def test_classification_constraint_reuses_existing_probability_adapter() -> None:
    model = _model()
    constraint = ClassificationConstraint(output="pass", feasible_class=0)

    probability = constraint.to_probability_of_feasibility(model)

    assert isinstance(probability, ClassificationProbabilityOfFeasibility)
    assert probability.model is model[1]
    assert probability.feasible_class == 0


def test_classification_constraint_probability_matches_classifier() -> None:
    torch.manual_seed(11)
    model = _model()
    X = torch.rand(3, 2, dtype=torch.double)
    constraint = ClassificationConstraint(output="pass", feasible_class=1)
    probability = constraint.to_probability_of_feasibility(model)

    expected = model.entry_predict_proba(1, X)[..., 1]

    torch.testing.assert_close(probability(X), expected)
