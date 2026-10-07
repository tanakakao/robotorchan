"""Tests for heterogeneous feasibility representation contracts."""

import pytest
import torch

from robotorchan.models.classification.binary.standard.single_task import (
    BinarySingleTaskGPClassifier,
)
from robotorchan.models.heterogeneous import HeterogeneousModel
from robotorchan.models.standard.single_task import SingleTaskGP
from robotorchan.semantics import (
    ClassificationConstraint,
    ContinuousConstraint,
    FeasibilityRepresentationKind,
    ProbabilityOfFeasibility,
    ProbabilityResidualFeasibility,
    SampleProbabilityOfFeasibility,
    SampleResidualFeasibility,
)


def _model() -> HeterogeneousModel:
    train_X = torch.rand(6, 2, dtype=torch.double)
    return HeterogeneousModel(
        SingleTaskGP(train_X, torch.rand(6, 1, dtype=torch.double)),
        BinarySingleTaskGPClassifier(
            train_X,
            torch.tensor([0, 1, 0, 1, 0, 1]),
        ),
        output_names=["cost", "pass"],
    )


def test_continuous_constraint_exposes_sample_residual_representation() -> None:
    model = _model()
    representation = ContinuousConstraint(
        output="cost",
        threshold=0.5,
    ).to_feasibility_representation(model)

    assert isinstance(representation, SampleResidualFeasibility)
    assert representation.kind is FeasibilityRepresentationKind.SAMPLE_RESIDUAL
    samples = torch.tensor([[[0.4], [0.5], [0.6]]], dtype=torch.double)
    torch.testing.assert_close(
        representation.constraint(samples),
        torch.tensor([[-0.1, 0.0, 0.1]], dtype=torch.double),
    )


def test_classification_constraint_exposes_probability_of_feasibility() -> None:
    model = _model()
    representation = ClassificationConstraint(
        output="pass",
        feasible_class=1,
    ).to_feasibility_representation(model)

    assert isinstance(representation, ProbabilityOfFeasibility)
    assert representation.kind is FeasibilityRepresentationKind.PROBABILITY_OF_FEASIBILITY
    assert representation.probability.model is model[1]


def test_thresholded_classification_exposes_probability_residual() -> None:
    model = _model()
    representation = ClassificationConstraint(
        output="pass",
        feasible_class=1,
        probability_threshold=0.8,
    ).to_feasibility_representation(model)

    assert isinstance(representation, ProbabilityResidualFeasibility)
    assert representation.kind is FeasibilityRepresentationKind.PROBABILITY_RESIDUAL
    assert representation.constraint.threshold == 0.8
    assert representation.constraint.probability_of_feasibility.model is model[1]


@pytest.mark.parametrize(
    ("wrapper", "payload"),
    [
        (SampleProbabilityOfFeasibility, lambda X: X),
        (SampleResidualFeasibility, lambda samples: samples),
        (ProbabilityOfFeasibility, object()),
        (ProbabilityResidualFeasibility, object()),
    ],
)
def test_feasibility_representation_kind_cannot_be_overridden(
    wrapper: type,
    payload: object,
) -> None:
    with pytest.raises(TypeError, match="kind"):
        wrapper(
            payload,
            kind=FeasibilityRepresentationKind.PROBABILITY_OF_FEASIBILITY,
        )


def test_feasibility_representation_kinds_are_semantically_distinct() -> None:
    assert len(FeasibilityRepresentationKind) == 4
    assert {kind.value for kind in FeasibilityRepresentationKind} == {
        "sample_probability_of_feasibility",
        "sample_residual",
        "probability_of_feasibility",
        "probability_residual",
    }
