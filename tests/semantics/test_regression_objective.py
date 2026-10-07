"""Tests for regression objective semantics."""

import pytest
import torch

from robotorchan.models.classification.binary.standard.single_task import (
    BinarySingleTaskGPClassifier,
)
from robotorchan.models.heterogeneous import HeterogeneousModel
from robotorchan.models.standard.single_task import SingleTaskGP
from robotorchan.semantics import ObjectiveDirection, RegressionObjective


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


def test_regression_objective_resolves_name_and_index() -> None:
    model = _model()

    assert RegressionObjective(output="strength").resolve_output(model) == 0
    assert RegressionObjective(output=0).resolve_output(model) == 0


def test_regression_objective_rejects_classification_output() -> None:
    model = _model()

    with pytest.raises(TypeError, match="requires a regression output"):
        RegressionObjective(output="pass").resolve_output(model)


@pytest.mark.parametrize(
    ("direction", "expected"),
    [
        (ObjectiveDirection.MAXIMIZE, [1.0, 2.0]),
        (ObjectiveDirection.MINIMIZE, [-1.0, -2.0]),
    ],
)
def test_regression_objective_converts_to_botorch_mc_objective(
    direction: ObjectiveDirection,
    expected: list[float],
) -> None:
    model = _model()
    objective = RegressionObjective(output="strength", direction=direction)
    botorch_objective = objective.to_botorch(model)
    samples = torch.tensor(
        [[[1.0, 0.2], [2.0, 0.8]]],
        dtype=torch.double,
    )

    values = botorch_objective(samples)

    assert torch.equal(values, torch.tensor([expected], dtype=torch.double))
