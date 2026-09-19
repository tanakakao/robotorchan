"""Tests for mixed-input reduced multi-task GP foundations."""

import torch

from robotorchan.models.reduced.mixed_multitask import (
    MixedReducedKroneckerMultiTaskGP,
    MixedReducedMultiTaskGP,
)
from robotorchan.reduction.input import PCAInputReducer


def test_mixed_reduced_multitask_preserves_categories_and_task() -> None:
    data = torch.linspace(0.0, 1.0, 12, dtype=torch.double).unsqueeze(-1)
    cont = torch.cat((data, data.square(), torch.sin(data)), dim=-1)
    cat = (torch.arange(12) % 2).double().unsqueeze(-1)
    task = (torch.arange(12) % 2).double().unsqueeze(-1)
    train_X = torch.cat((cont[:, :2], cat, cont[:, 2:], task), dim=-1)
    train_Y = (cont[:, :1] + 0.2 * task).double()

    model = MixedReducedMultiTaskGP(
        train_X,
        train_Y,
        task_feature=4,
        cat_dims=[2],
        input_reducer=PCAInputReducer(n_components=2),
    )
    reduced = model._prepare_inputs(train_X)

    assert reduced.shape[-1] == 4
    assert torch.equal(reduced[:, 2], cat.squeeze(-1))
    assert torch.equal(reduced[:, 3], task.squeeze(-1))
    assert model.reduced_task_feature == 3
    assert model.cat_dims == (2,)


def test_mixed_reduced_kronecker_preserves_categories_and_posterior() -> None:
    x = torch.linspace(0.0, 1.0, 10, dtype=torch.double).unsqueeze(-1)
    cat = (torch.arange(10) % 2).double().unsqueeze(-1)
    train_X = torch.cat((x, cat, x.square(), torch.sin(x)), dim=-1)
    train_Y = torch.cat((torch.sin(x), torch.cos(x)), dim=-1)

    model = MixedReducedKroneckerMultiTaskGP(
        train_X,
        train_Y,
        cat_dims=[1],
        input_reducer=PCAInputReducer(n_components=2),
    )
    reduced = model._prepare_inputs(train_X)
    model.eval()
    model.likelihood.eval()
    posterior = model.posterior(train_X[:2])

    assert reduced.shape[-1] == 3
    assert torch.equal(reduced[:, 2], cat.squeeze(-1))
    assert posterior.mean.shape[-1] == 2


def test_mixed_reduced_multitask_load_state_resynchronizes_projection() -> None:
    x = torch.linspace(0.0, 1.0, 10, dtype=torch.double).unsqueeze(-1)
    cat = (torch.arange(10) % 2).double().unsqueeze(-1)
    task = (torch.arange(10) % 2).double().unsqueeze(-1)
    train_X = torch.cat((x, cat, x.square(), task), dim=-1)
    train_Y = x + 0.2 * task
    source = MixedReducedMultiTaskGP(
        train_X,
        train_Y,
        task_feature=-1,
        cat_dims=[1],
        input_reducer=PCAInputReducer(n_components=1),
    )
    restored = MixedReducedMultiTaskGP(
        train_X,
        train_Y,
        task_feature=-1,
        cat_dims=[1],
        input_reducer=PCAInputReducer(n_components=1),
    )

    restored.load_state_dict(source.state_dict())

    torch.testing.assert_close(restored.train_inputs[0], restored._prepare_inputs(train_X))


def test_mixed_reduced_kronecker_load_state_resynchronizes_projection() -> None:
    x = torch.linspace(0.0, 1.0, 10, dtype=torch.double).unsqueeze(-1)
    cat = (torch.arange(10) % 2).double().unsqueeze(-1)
    train_X = torch.cat((x, cat, x.square()), dim=-1)
    train_Y = torch.cat((torch.sin(x), torch.cos(x)), dim=-1)
    source = MixedReducedKroneckerMultiTaskGP(
        train_X,
        train_Y,
        cat_dims=[1],
        input_reducer=PCAInputReducer(n_components=1),
    )
    restored = MixedReducedKroneckerMultiTaskGP(
        train_X,
        train_Y,
        cat_dims=[1],
        input_reducer=PCAInputReducer(n_components=1),
    )

    restored.load_state_dict(source.state_dict())

    torch.testing.assert_close(restored.train_inputs[0], restored._prepare_inputs(train_X))
