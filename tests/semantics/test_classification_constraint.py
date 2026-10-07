"""Tests for classification constraint semantics."""

import pytest
import torch

from robotorchan.acquisition.classification_constraints import (
    ClassificationProbabilityOfFeasibility,
)
from robotorchan.models.classification.binary.non_gp.sklearn import (
    RandomForestBinaryClassifier,
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


def test_classification_constraint_supports_classifier_without_num_outputs() -> None:
    train_X = torch.rand(8, 2, dtype=torch.double)
    classifier = RandomForestBinaryClassifier(
        train_X,
        torch.tensor([0, 1, 0, 1, 0, 1, 0, 1]),
    )
    model = HeterogeneousModel(
        classifier,
        output_names=["pass"],
    )
    constraint = ClassificationConstraint(output="pass", feasible_class=1)

    probability = constraint.to_probability_of_feasibility(model)

    assert constraint.resolve_owner(model) == (0, 0)
    assert probability.model is classifier



@pytest.mark.parametrize("threshold", [True, "0.8", torch.tensor(0.8)])
def test_classification_constraint_rejects_non_scalar_probability_threshold(
    threshold: object,
) -> None:
    with pytest.raises(TypeError, match="real scalar"):
        ClassificationConstraint(
            output="pass",
            probability_threshold=threshold,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("threshold", [float("nan"), float("inf"), float("-inf")])
def test_classification_constraint_requires_finite_probability_threshold(
    threshold: float,
) -> None:
    with pytest.raises(ValueError, match="finite"):
        ClassificationConstraint(output="pass", probability_threshold=threshold)


@pytest.mark.parametrize("threshold", [-0.1, 1.1])
def test_classification_constraint_bounds_probability_threshold(threshold: float) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        ClassificationConstraint(output="pass", probability_threshold=threshold)


@pytest.mark.parametrize("threshold", [0.0, 1.0])
def test_classification_constraint_accepts_probability_boundary(
    threshold: float,
) -> None:
    constraint = ClassificationConstraint(
        output="pass",
        probability_threshold=threshold,
    )

    assert constraint.probability_threshold == threshold


def test_classification_constraint_requires_threshold_for_probability_constraint() -> None:
    model = _model()

    with pytest.raises(ValueError, match="probability_threshold is required"):
        ClassificationConstraint(output="pass").to_probability_constraint(model)


def test_classification_probability_constraint_uses_nonpositive_feasibility() -> None:
    model = _model()
    constraint = ClassificationConstraint(
        output="pass",
        feasible_class=1,
        probability_threshold=0.7,
    )
    probability_constraint = constraint.to_probability_constraint(model)
    probability_constraint.probability_of_feasibility.forward = lambda X: torch.tensor(
        [0.6, 0.7, 0.8],
        dtype=X.dtype,
        device=X.device,
    )
    X = torch.rand(3, 2, dtype=torch.double)

    residual = probability_constraint(X)

    torch.testing.assert_close(
        residual,
        torch.tensor([0.1, 0.0, -0.1], dtype=torch.double),
    )
