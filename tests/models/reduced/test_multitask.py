"""Tests for multi-task input-reduction foundations."""

import torch

from robotorchan.models.reduced.multitask import (
    ReducedKroneckerMultiTaskGP,
    ReducedMultiTaskGP,
)
from robotorchan.reduction.input import PCAInputReducer


def test_reduced_multitask_keeps_task_feature_out_of_reducer() -> None:
    dtype = torch.double
    data = torch.rand(12, 4, dtype=dtype)
    task = torch.tensor([0.0, 1.0] * 6, dtype=dtype).unsqueeze(-1)
    train_X = torch.cat([data[:, :2], task, data[:, 2:]], dim=-1)
    train_Y = (data[:, :1] + task).to(dtype)

    model = ReducedMultiTaskGP(
        train_X,
        train_Y,
        task_feature=2,
        input_reducer=PCAInputReducer(n_components=2),
    )

    assert model.input_reducer.input_dim == 4
    assert model.original_task_feature == 2
    assert model.reduced_task_feature == 2
    assert model.original_input_dim == 5
    assert model.reduced_input_dim == 3
    prepared = model._prepare_inputs(train_X)
    assert prepared.shape == torch.Size([12, 3])
    assert torch.equal(prepared[:, -1], task.squeeze(-1))
    assert torch.equal(model.raw_train_X, train_X)


def test_reduced_multitask_accepts_negative_task_feature() -> None:
    dtype = torch.double
    data = torch.rand(10, 3, dtype=dtype)
    task = torch.tensor([0.0, 1.0] * 5, dtype=dtype).unsqueeze(-1)
    train_X = torch.cat([data, task], dim=-1)
    train_Y = data[:, :1]

    model = ReducedMultiTaskGP(
        train_X,
        train_Y,
        task_feature=-1,
        input_reducer=PCAInputReducer(n_components=2),
    )

    assert model.original_task_feature == 3
    assert model._prepare_inputs(train_X).shape[-1] == 3


def test_reduced_kronecker_reduces_all_input_features() -> None:
    dtype = torch.double
    train_X = torch.rand(8, 5, dtype=dtype)
    train_Y = torch.stack(
        [train_X[:, 0], train_X[:, 1] + train_X[:, 2]],
        dim=-1,
    )

    model = ReducedKroneckerMultiTaskGP(
        train_X,
        train_Y,
        input_reducer=PCAInputReducer(n_components=2),
    )

    assert model.input_reducer.input_dim == 5
    assert model.original_input_dim == 5
    assert model.reduced_input_dim == 2
    assert model._prepare_inputs(train_X).shape == torch.Size([8, 2])
    assert torch.equal(model.raw_train_X, train_X)


def test_reduced_multitask_rejects_invalid_public_dimension() -> None:
    dtype = torch.double
    data = torch.rand(10, 3, dtype=dtype)
    task = torch.tensor([0.0, 1.0] * 5, dtype=dtype).unsqueeze(-1)
    train_X = torch.cat([data, task], dim=-1)
    train_Y = data[:, :1]
    model = ReducedMultiTaskGP(
        train_X,
        train_Y,
        task_feature=-1,
        input_reducer=PCAInputReducer(n_components=2),
    )

    try:
        model._prepare_inputs(torch.rand(2, 7, dtype=dtype))
    except ValueError as error:
        assert "Expected final input dimension" in str(error)
    else:
        raise AssertionError("Expected invalid public input dimension to fail.")
