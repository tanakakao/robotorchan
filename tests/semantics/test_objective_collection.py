"""Tests for ordered semantic objective collections."""

import pytest
import torch

from robotorchan.models.classification.binary.standard.single_task import (
    BinarySingleTaskGPClassifier,
)
from robotorchan.models.heterogeneous import HeterogeneousModel
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
