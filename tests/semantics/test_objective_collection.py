"""Tests for ordered semantic objective collections."""

import pytest
import torch

from robotorchan.models.classification.binary.standard.single_task import (
    BinarySingleTaskGPClassifier,
)
from robotorchan.models.heterogeneous import HeterogeneousModel
from robotorchan.models.standard.model_list import ModelListGP
from robotorchan.models.standard.single_task import SingleTaskGP
from robotorchan.semantics import (
    ObjectiveCollection,
    ProbabilityObjective,
    RegressionObjective,
)


def _model() -> HeterogeneousModel:
    train_X = torch.rand(8, 2, dtype=torch.double)
    return HeterogeneousModel(
        SingleTaskGP(train_X, torch.rand(8, 1, dtype=torch.double)),
        BinarySingleTaskGPClassifier(
            train_X,
            torch.tensor([0, 1, 0, 1, 0, 1, 0, 1]),
        ),
        output_names=["strength", "pass"],
    )


def test_objective_collection_preserves_declared_order() -> None:
    regression = RegressionObjective(output="strength")
    probability = ProbabilityObjective(output="pass", class_index=1)
    objectives = ObjectiveCollection(regression, probability)

    assert len(objectives) == 2
    assert tuple(objectives) == (regression, probability)
    assert objectives[0] is regression
    assert objectives[1] is probability


def test_objective_collection_resolves_mixed_outputs() -> None:
    model = _model()
    objectives = ObjectiveCollection(
        RegressionObjective(output="strength"),
        ProbabilityObjective(output="pass", class_index=1),
    )

    assert objectives.resolve_outputs(model) == (0, 1)
    assert objectives.validate(model) is None


def test_objective_collection_preserves_duplicate_output_semantics() -> None:
    model = _model()
    objectives = ObjectiveCollection(
        ProbabilityObjective(output="pass", class_index=0),
        ProbabilityObjective(output="pass", class_index=1),
    )

    assert objectives.resolve_outputs(model) == (1, 1)


def test_objective_collection_requires_at_least_one_objective() -> None:
    with pytest.raises(ValueError, match="at least one objective"):
        ObjectiveCollection()


def test_objective_collection_rejects_non_semantic_objective() -> None:
    with pytest.raises(TypeError, match="RegressionObjective or ProbabilityObjective"):
        ObjectiveCollection(object())  # type: ignore[arg-type]


def test_objective_collection_validation_preserves_objective_errors() -> None:
    model = _model()
    objectives = ObjectiveCollection(
        RegressionObjective(output="pass"),
        ProbabilityObjective(output="strength", class_index=1),
    )

    with pytest.raises(TypeError, match="RegressionObjective requires a regression output"):
        objectives.validate(model)


def test_objective_collection_resolves_model_list_local_outputs() -> None:
    train_X = torch.rand(7, 2, dtype=torch.double)
    first = SingleTaskGP(train_X, torch.rand(7, 1, dtype=torch.double))
    second = SingleTaskGP(train_X, torch.rand(7, 1, dtype=torch.double))
    model = HeterogeneousModel(
        ModelListGP(first, second),
        output_names=["strength", "cost"],
    )
    objectives = ObjectiveCollection(
        RegressionObjective(output="cost"),
        RegressionObjective(output="strength"),
    )

    assert objectives.resolve_outputs(model) == (1, 0)
    assert objectives.resolve_owners(model) == ((0, 1), (0, 0))
    assert objectives.group_by_entry(model) == {0: tuple(objectives)}


def test_objective_collection_groups_heterogeneous_entries() -> None:
    train_X = torch.rand(8, 2, dtype=torch.double)
    first = SingleTaskGP(train_X, torch.rand(8, 1, dtype=torch.double))
    second = SingleTaskGP(train_X, torch.rand(8, 1, dtype=torch.double))
    classifier = BinarySingleTaskGPClassifier(
        train_X,
        torch.tensor([0, 1, 0, 1, 0, 1, 0, 1]),
    )
    model = HeterogeneousModel(
        ModelListGP(first, second),
        classifier,
        SingleTaskGP(train_X, torch.rand(8, 1, dtype=torch.double)),
        output_names=["strength", "cost", "pass", "lifetime"],
    )
    cost = RegressionObjective(output="cost")
    passing = ProbabilityObjective(output="pass", class_index=1)
    strength = RegressionObjective(output="strength")
    lifetime = RegressionObjective(output="lifetime")
    objectives = ObjectiveCollection(cost, passing, strength, lifetime)

    assert objectives.resolve_owners(model) == ((0, 1), (1, 0), (0, 0), (2, 0))
    assert objectives.group_by_entry(model) == {
        0: (cost, strength),
        1: (passing,),
        2: (lifetime,),
    }


def test_regression_objective_exposes_model_list_entry_owner() -> None:
    train_X = torch.rand(7, 2, dtype=torch.double)
    model = HeterogeneousModel(
        ModelListGP(
            SingleTaskGP(train_X, torch.rand(7, 1, dtype=torch.double)),
            SingleTaskGP(train_X, torch.rand(7, 1, dtype=torch.double)),
        ),
        output_names=["first", "second"],
    )

    assert RegressionObjective(output="second").resolve_owner(model) == (0, 1)
