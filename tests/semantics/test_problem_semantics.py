"""Tests for heterogeneous objective and constraint composition."""

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
    ProbabilityObjective,
    ProblemSemantics,
    RegressionObjective,
)


def _model() -> HeterogeneousModel:
    train_X = torch.rand(6, 2, dtype=torch.double)
    return HeterogeneousModel(
        SingleTaskGP(train_X, torch.rand(6, 1, dtype=torch.double)),
        SingleTaskGP(train_X, torch.rand(6, 1, dtype=torch.double)),
        BinarySingleTaskGPClassifier(
            train_X,
            torch.tensor([0, 1, 0, 1, 0, 1]),
        ),
        output_names=["strength", "cost", "pass"],
    )


def test_problem_semantics_composes_mixed_objectives_and_constraints() -> None:
    model = _model()
    semantics = ProblemSemantics(
        objectives=(
            RegressionObjective("strength"),
            ProbabilityObjective("pass", class_index=1),
        ),
        constraints=(
            ContinuousConstraint("cost", threshold=0.5),
            ClassificationConstraint("pass", feasible_class=1),
        ),
    )

    semantics.validate(model)

    assert semantics.resolve_objective_outputs(model) == (0, 2)
    assert semantics.resolve_constraint_outputs(model) == (1, 2)
    assert semantics.resolve_constraint_owners(model) == ((1, 0), (2, 0))
    assert tuple(item.kind for item in semantics.feasibility_representations(model)) == (
        FeasibilityRepresentationKind.SAMPLE_RESIDUAL,
        FeasibilityRepresentationKind.PROBABILITY_OF_FEASIBILITY,
    )


def test_problem_semantics_allows_same_output_as_objective_and_constraint() -> None:
    model = _model()
    semantics = ProblemSemantics(
        objectives=(RegressionObjective("strength"),),
        constraints=(ContinuousConstraint("strength", threshold=0.5),),
    )

    semantics.validate(model)

    assert semantics.resolve_objective_outputs(model) == (0,)
    assert semantics.resolve_constraint_outputs(model) == (0,)


def test_problem_semantics_supports_no_constraints() -> None:
    model = _model()
    semantics = ProblemSemantics(objectives=(RegressionObjective("strength"),))

    semantics.validate(model)

    assert semantics.resolve_constraint_outputs(model) == ()
    assert semantics.resolve_constraint_owners(model) == ()
    assert semantics.feasibility_representations(model) == ()


def test_problem_semantics_rejects_invalid_collection_types() -> None:
    objective = RegressionObjective(0)
    constraint = ContinuousConstraint(0, threshold=0.5)

    with pytest.raises(TypeError, match="objectives"):
        ProblemSemantics(objectives=[objective])  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="constraints"):
        ProblemSemantics(
            objectives=(objective,),
            constraints=[constraint],  # type: ignore[arg-type]
        )

    with pytest.raises(TypeError, match="constraints accepts"):
        ProblemSemantics(
            objectives=(objective,),
            constraints=(objective,),  # type: ignore[arg-type]
        )


def test_problem_semantics_preserves_validation_boundaries() -> None:
    model = _model()

    with pytest.raises(TypeError, match="RegressionObjective"):
        ProblemSemantics(
            objectives=(RegressionObjective("pass"),),
        ).validate(model)

    with pytest.raises(TypeError, match="ClassificationConstraint"):
        ProblemSemantics(
            objectives=(RegressionObjective("strength"),),
            constraints=(ClassificationConstraint("cost"),),
        ).validate(model)
