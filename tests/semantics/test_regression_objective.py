"""Tests for regression objective semantics."""

import pytest
import torch
from botorch.sampling.normal import SobolQMCNormalSampler

from robotorchan.models.classification.binary.standard.single_task import (
    BinarySingleTaskGPClassifier,
)
from robotorchan.models.heterogeneous import HeterogeneousModel
from robotorchan.models.standard.multitask import KroneckerMultiTaskGP, MultiTaskGP
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
        [[[1.0], [2.0]]],
        dtype=torch.double,
    )

    values = botorch_objective(samples)

    assert torch.equal(values, torch.tensor([expected], dtype=torch.double))


def test_regression_objective_uses_entry_local_sample_index() -> None:
    train_X = torch.rand(6, 2, dtype=torch.double)
    classifier = BinarySingleTaskGPClassifier(
        train_X,
        torch.tensor([0, 1, 0, 1, 0, 1]),
    )
    regression = SingleTaskGP(train_X, torch.rand(6, 1, dtype=torch.double))
    model = HeterogeneousModel(
        classifier,
        regression,
        output_names=["pass", "strength"],
    )
    objective = RegressionObjective(output="strength")
    samples = torch.tensor([[[1.0], [2.0]]], dtype=torch.double)

    assert objective.resolve_output(model) == 1
    assert model.output_owner(1) == (1, 0)
    assert torch.equal(
        objective.to_botorch(model)(samples),
        torch.tensor([[1.0, 2.0]], dtype=torch.double),
    )


def test_regression_objective_resolves_negative_global_index() -> None:
    model = _model()

    assert RegressionObjective(output=-2).resolve_output(model) == 0


def test_regression_objective_preserves_batch_and_sample_dimensions() -> None:
    model = _model()
    objective = RegressionObjective(output="strength").to_botorch(model)
    samples = torch.rand(4, 3, 2, 1, dtype=torch.double)

    values = objective(samples)

    assert values.shape == torch.Size([4, 3, 2])
    assert values.dtype == samples.dtype
    assert values.device == samples.device


def test_regression_objective_selects_local_output_from_multi_output_entry() -> None:
    train_X = torch.rand(6, 2, dtype=torch.double)
    regression = KroneckerMultiTaskGP(
        train_X,
        torch.rand(6, 2, dtype=torch.double),
    )
    model = HeterogeneousModel(
        regression,
        output_names=["strength", "conductivity"],
    )
    objective = RegressionObjective(output="conductivity").to_botorch(model)
    samples = torch.tensor(
        [[[1.0, 10.0], [2.0, 20.0]]],
        dtype=torch.double,
    )

    assert model.output_owner(1) == (0, 1)
    assert torch.equal(
        objective(samples),
        torch.tensor([[10.0, 20.0]], dtype=torch.double),
    )


def test_regression_objective_works_with_native_entry_posterior_samples() -> None:
    torch.manual_seed(7)
    train_X = torch.rand(8, 2, dtype=torch.double)
    regression = KroneckerMultiTaskGP(
        train_X,
        torch.rand(8, 2, dtype=torch.double),
    )
    model = HeterogeneousModel(
        regression,
        output_names=["strength", "conductivity"],
    )
    candidate = torch.rand(3, 2, dtype=torch.double)
    posterior = model.entry_posterior(0, candidate)
    samples = SobolQMCNormalSampler(torch.Size([4]), seed=17)(posterior)
    objective = RegressionObjective(output="conductivity").to_botorch(model)

    values = objective(samples)

    assert samples.shape == torch.Size([4, 3, 2])
    assert values.shape == torch.Size([4, 3])
    torch.testing.assert_close(values, samples[..., 1])


@pytest.mark.parametrize("output", ["missing", 2, -3])
def test_regression_objective_preserves_output_reference_errors(
    output: int | str,
) -> None:
    model = _model()
    objective = RegressionObjective(output=output)

    with pytest.raises((KeyError, IndexError)):
        objective.resolve_output(model)


def test_regression_objective_works_with_long_format_multitask_gp() -> None:
    torch.manual_seed(13)
    data_X = torch.rand(6, 2, dtype=torch.double)
    train_X = torch.cat(
        [
            torch.cat([data_X, torch.zeros(6, 1, dtype=torch.double)], dim=-1),
            torch.cat([data_X, torch.ones(6, 1, dtype=torch.double)], dim=-1),
        ],
        dim=0,
    )
    train_Y = torch.rand(12, 1, dtype=torch.double)
    regression = MultiTaskGP(
        train_X,
        train_Y,
        task_feature=2,
        output_tasks=[0, 1],
    )
    model = HeterogeneousModel(
        regression,
        output_names=["task_zero", "task_one"],
    )
    candidate = torch.rand(3, 2, dtype=torch.double)
    posterior = model.entry_posterior(0, candidate)
    samples = SobolQMCNormalSampler(torch.Size([4]), seed=19)(posterior)
    objective = RegressionObjective(output="task_one").to_botorch(model)

    values = objective(samples)

    assert regression.num_outputs == 2
    assert model.output_owner(1) == (0, 1)
    assert samples.shape == torch.Size([4, 3, 2])
    torch.testing.assert_close(values, samples[..., 1])


def test_regression_objective_respects_long_format_output_task_order() -> None:
    torch.manual_seed(17)
    data_X = torch.rand(5, 2, dtype=torch.double)
    rows = [
        torch.cat(
            [
                data_X,
                torch.full((5, 1), float(task), dtype=torch.double),
            ],
            dim=-1,
        )
        for task in (0, 1, 2)
    ]
    regression = MultiTaskGP(
        torch.cat(rows, dim=0),
        torch.rand(15, 1, dtype=torch.double),
        task_feature=2,
        output_tasks=[2, 0],
    )
    model = HeterogeneousModel(
        regression,
        output_names=["task_two", "task_zero"],
    )
    samples = torch.tensor(
        [[[20.0, 0.0], [21.0, 1.0]]],
        dtype=torch.double,
    )

    task_two = RegressionObjective(output="task_two").to_botorch(model)
    task_zero = RegressionObjective(output="task_zero").to_botorch(model)

    assert model.output_owners == ((0, 0), (0, 1))
    assert torch.equal(task_two(samples), torch.tensor([[20.0, 21.0]], dtype=torch.double))
    assert torch.equal(task_zero(samples), torch.tensor([[0.0, 1.0]], dtype=torch.double))
