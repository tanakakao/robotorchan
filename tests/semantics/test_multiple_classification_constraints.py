"""Tests for multiple classification constraints."""

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
    train_X = torch.rand(8, 2, dtype=torch.double)
    labels_a = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    labels_b = torch.tensor([1, 1, 0, 0, 1, 0, 1, 0])
    return HeterogeneousModel(
        SingleTaskGP(train_X, torch.rand(8, 1, dtype=torch.double)),
        BinarySingleTaskGPClassifier(train_X, labels_a),
        BinarySingleTaskGPClassifier(train_X, labels_b),
        output_names=["objective", "safe", "stable"],
    )


def test_multiple_classification_constraints_preserve_order_and_ownership() -> None:
    model = _model()
    constraints = ConstraintCollection(
        ClassificationConstraint("safe"),
        ContinuousConstraint("objective", threshold=0.8),
        ClassificationConstraint("stable", probability_threshold=0.7),
        ClassificationConstraint("safe", probability_threshold=0.9),
    )

    assert constraints.classification_constraints() == (
        constraints[0],
        constraints[2],
        constraints[3],
    )
    assert constraints.group_classification_by_output(model) == {
        1: (constraints[0], constraints[3]),
        2: (constraints[2],),
    }
    assert constraints.group_classification_by_entry(model) == {
        1: (constraints[0], constraints[3]),
        2: (constraints[2],),
    }


def test_multiple_classification_constraints_keep_independent_representations() -> None:
    model = _model()
    constraints = ConstraintCollection(
        ClassificationConstraint("safe"),
        ClassificationConstraint("stable"),
        ClassificationConstraint("safe", probability_threshold=0.75),
    )

    representations = constraints.feasibility_representations(model)

    assert tuple(item.kind for item in representations) == (
        FeasibilityRepresentationKind.PROBABILITY_OF_FEASIBILITY,
        FeasibilityRepresentationKind.PROBABILITY_OF_FEASIBILITY,
        FeasibilityRepresentationKind.PROBABILITY_RESIDUAL,
    )
    assert representations[0] is not representations[1]
    assert representations[0] is not representations[2]


def test_multiple_classification_constraints_evaluate_their_own_models() -> None:
    model = _model()
    safe = ClassificationConstraint("safe")
    stable = ClassificationConstraint("stable")
    X = torch.rand(3, 1, 2, dtype=torch.double)

    safe_probability = safe.to_probability_of_feasibility(model)(X)
    stable_probability = stable.to_probability_of_feasibility(model)(X)

    assert safe_probability.shape == torch.Size([3, 1])
    assert stable_probability.shape == torch.Size([3, 1])
    assert torch.all((0.0 <= safe_probability) & (safe_probability <= 1.0))
    assert torch.all((0.0 <= stable_probability) & (stable_probability <= 1.0))


def test_problem_semantics_supports_multiple_classification_constraints() -> None:
    model = _model()
    semantics = ProblemSemantics(
        objectives=(RegressionObjective("objective"),),
        constraints=(
            ClassificationConstraint("safe"),
            ClassificationConstraint("stable", probability_threshold=0.8),
        ),
    )

    semantics.validate(model)

    assert semantics.resolve_constraint_outputs(model) == (1, 2)
    assert semantics.resolve_constraint_owners(model) == ((1, 0), (2, 0))
