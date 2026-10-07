"""Tests for continuous outcome-constraint semantics."""

import pytest
import torch

from robotorchan.models.classification.binary.standard.single_task import (
    BinarySingleTaskGPClassifier,
)
from robotorchan.models.heterogeneous import HeterogeneousModel
from robotorchan.models.standard.multitask import KroneckerMultiTaskGP
from robotorchan.models.standard.single_task import SingleTaskGP
from robotorchan.semantics import (
    ConstraintDirection,
    ContinuousConstraint,
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


def test_continuous_constraint_resolves_regression_output() -> None:
    model = _model()
    constraint = ContinuousConstraint(output="cost", threshold=100.0)

    assert constraint.resolve_output(model) == 0
    assert constraint.resolve_owner(model) == (0, 0)


def test_continuous_constraint_rejects_classification_output() -> None:
    model = _model()

    with pytest.raises(TypeError, match="requires a regression output"):
        ContinuousConstraint(output="pass", threshold=0.5).resolve_output(model)


@pytest.mark.parametrize(
    ("direction", "expected"),
    [
        (
            ConstraintDirection.LESS_THAN_OR_EQUAL,
            [-1.0, 0.0, 1.0],
        ),
        (
            ConstraintDirection.GREATER_THAN_OR_EQUAL,
            [1.0, 0.0, -1.0],
        ),
    ],
)
def test_continuous_constraint_uses_botorch_feasibility_sign(
    direction: ConstraintDirection,
    expected: list[float],
) -> None:
    model = _model()
    constraint = ContinuousConstraint(
        output="cost",
        threshold=2.0,
        direction=direction,
    ).to_botorch(model)
    samples = torch.tensor([[[1.0], [2.0], [3.0]]], dtype=torch.double)

    values = constraint(samples)

    assert torch.equal(values, torch.tensor([expected], dtype=torch.double))


def test_continuous_constraint_selects_entry_local_output() -> None:
    train_X = torch.rand(6, 2, dtype=torch.double)
    classifier = BinarySingleTaskGPClassifier(
        train_X,
        torch.tensor([0, 1, 0, 1, 0, 1]),
    )
    regression = KroneckerMultiTaskGP(
        train_X,
        torch.rand(6, 2, dtype=torch.double),
    )
    model = HeterogeneousModel(
        classifier,
        regression,
        output_names=["pass", "strength", "cost"],
    )
    constraint = ContinuousConstraint(
        output="cost",
        threshold=15.0,
    ).to_botorch(model)
    samples = torch.tensor(
        [[[1.0, 10.0], [2.0, 20.0]]],
        dtype=torch.double,
    )

    assert model.output_owner(2) == (1, 1)
    assert torch.equal(
        constraint(samples),
        torch.tensor([[-5.0, 5.0]], dtype=torch.double),
    )


def test_continuous_constraint_preserves_sample_dimensions_dtype_and_device() -> None:
    model = _model()
    constraint = ContinuousConstraint(output="cost", threshold=0.5).to_botorch(model)
    samples = torch.rand(4, 3, 2, 1, dtype=torch.double)

    values = constraint(samples)

    assert values.shape == torch.Size([4, 3, 2])
    assert values.dtype == samples.dtype
    assert values.device == samples.device



@pytest.mark.parametrize("threshold", [True, "100", torch.tensor(100.0)])
def test_continuous_constraint_rejects_non_scalar_threshold_types(
    threshold: object,
) -> None:
    with pytest.raises(TypeError, match="real scalar"):
        ContinuousConstraint(output="cost", threshold=threshold)  # type: ignore[arg-type]


@pytest.mark.parametrize("threshold", [float("nan"), float("inf"), float("-inf")])
def test_continuous_constraint_requires_finite_threshold(threshold: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        ContinuousConstraint(output="cost", threshold=threshold)


def test_continuous_constraint_accepts_integer_threshold() -> None:
    constraint = ContinuousConstraint(output="cost", threshold=100)

    assert constraint.threshold == 100


def test_continuous_constraint_requires_constraint_direction() -> None:
    with pytest.raises(TypeError, match="ConstraintDirection"):
        ContinuousConstraint(
            output="cost",
            threshold=100.0,
            direction="less_than_or_equal",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "direction",
    [
        ConstraintDirection.LESS_THAN_OR_EQUAL,
        ConstraintDirection.GREATER_THAN_OR_EQUAL,
    ],
)
def test_continuous_constraint_boundary_is_feasible(
    direction: ConstraintDirection,
) -> None:
    model = _model()
    constraint = ContinuousConstraint(
        output="cost",
        threshold=2.0,
        direction=direction,
    ).to_botorch(model)
    samples = torch.tensor([[[2.0]]], dtype=torch.double)

    residual = constraint(samples)

    assert residual.item() == 0.0
