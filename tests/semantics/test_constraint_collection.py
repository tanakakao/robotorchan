"""Tests for mixed semantic constraint collections."""

import pytest
import torch

from robotorchan.models.classification.binary.standard.single_task import (
    BinarySingleTaskGPClassifier,
)
from robotorchan.models.heterogeneous import HeterogeneousModel
from robotorchan.models.standard.single_task import SingleTaskGP
from robotorchan.semantics import (
    ClassificationConstraint,
    ConstraintCollection,
    ContinuousConstraint,
    FeasibilityRepresentationKind,
    ProblemSemantics,
    RegressionObjective,
)


def _model() -> HeterogeneousModel:
    train_X = torch.rand(6, 2, dtype=torch.double)
    return HeterogeneousModel(
        SingleTaskGP(train_X, torch.rand(6, 2, dtype=torch.double)),
        BinarySingleTaskGPClassifier(
            train_X,
            torch.tensor([0, 1, 0, 1, 0, 1]),
        ),
        output_names=["strength", "cost", "pass"],
    )


def test_mixed_constraint_collection_preserves_order_and_owners() -> None:
    model = _model()
    constraints = ConstraintCollection(
        ClassificationConstraint("pass"),
        ContinuousConstraint("cost", threshold=0.5),
        ContinuousConstraint("strength", threshold=0.2),
    )

    constraints.validate(model)

    assert constraints.resolve_outputs(model) == (2, 1, 0)
    assert constraints.resolve_owners(model) == ((1, 0), (0, 1), (0, 0))
    assert constraints.group_by_entry(model) == {
        1: (constraints[0],),
        0: (constraints[1], constraints[2]),
    }


def test_mixed_constraint_collection_preserves_representation_kinds() -> None:
    model = _model()
    constraints = ConstraintCollection(
        ContinuousConstraint("cost", threshold=0.5),
        ClassificationConstraint("pass"),
        ClassificationConstraint("pass", probability_threshold=0.8),
    )

    assert tuple(item.kind for item in constraints.feasibility_representations(model)) == (
        FeasibilityRepresentationKind.SAMPLE_RESIDUAL,
        FeasibilityRepresentationKind.PROBABILITY_OF_FEASIBILITY,
        FeasibilityRepresentationKind.PROBABILITY_RESIDUAL,
    )


def test_constraint_collection_can_be_empty() -> None:
    model = _model()
    constraints = ConstraintCollection()

    constraints.validate(model)

    assert len(constraints) == 0
    assert constraints.resolve_outputs(model) == ()
    assert constraints.resolve_owners(model) == ()
    assert constraints.group_by_entry(model) == {}
    assert constraints.feasibility_representations(model) == ()


def test_constraint_collection_rejects_non_constraint_items() -> None:
    with pytest.raises(TypeError, match="ConstraintCollection accepts"):
        ConstraintCollection(RegressionObjective(0))  # type: ignore[arg-type]


def test_problem_semantics_accepts_constraint_collection() -> None:
    model = _model()
    constraints = ConstraintCollection(
        ContinuousConstraint("cost", threshold=0.5),
        ClassificationConstraint("pass"),
    )
    semantics = ProblemSemantics(
        objectives=(RegressionObjective("strength"),),
        constraints=constraints,
    )

    semantics.validate(model)

    assert semantics.constraints is constraints
    assert semantics.resolve_constraint_outputs(model) == (1, 2)
