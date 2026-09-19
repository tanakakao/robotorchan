"""Tests for multi-task input-reduction foundations."""

import torch

from robotorchan.models.reduced.multitask import (
    AutoEncoderKroneckerMultiTaskGP,
    AutoEncoderMultiTaskGP,
    PCAKroneckerMultiTaskGP,
    PCAMultiTaskGP,
    PLSKroneckerMultiTaskGP,
    PLSMultiTaskGP,
    RandomProjectionKroneckerMultiTaskGP,
    RandomProjectionMultiTaskGP,
    ReducedKroneckerMultiTaskGP,
    ReducedMultiTaskGP,
    VAEKroneckerMultiTaskGP,
    VAEMultiTaskGP,
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


def test_pca_multitask_convenience_model_preserves_task_feature() -> None:
    dtype = torch.double
    data = torch.rand(12, 5, dtype=dtype)
    task = torch.tensor([0.0, 1.0] * 6, dtype=dtype).unsqueeze(-1)
    train_X = torch.cat([data[:, :1], task, data[:, 1:]], dim=-1)
    train_Y = data[:, :1] + task

    model = PCAMultiTaskGP(
        train_X,
        train_Y,
        task_feature=1,
        n_components=2,
    )

    prepared = model._prepare_inputs(train_X)
    assert model.input_reducer.input_dim == 5
    assert model.original_task_feature == 1
    assert model.reduced_task_feature == 2
    assert prepared.shape == torch.Size([12, 3])
    assert torch.equal(prepared[:, -1], task.squeeze(-1))
    assert model.make_mll() is not None


def test_random_projection_multitask_is_seed_reproducible() -> None:
    dtype = torch.double
    data = torch.rand(10, 4, dtype=dtype)
    task = torch.tensor([0.0, 1.0] * 5, dtype=dtype).unsqueeze(-1)
    train_X = torch.cat([data, task], dim=-1)
    train_Y = data[:, :1]

    first = RandomProjectionMultiTaskGP(
        train_X,
        train_Y,
        task_feature=-1,
        n_components=2,
        random_state=7,
    )
    second = RandomProjectionMultiTaskGP(
        train_X,
        train_Y,
        task_feature=-1,
        n_components=2,
        random_state=7,
    )

    assert torch.equal(first.input_reducer.projection, second.input_reducer.projection)
    assert torch.equal(first._prepare_inputs(train_X), second._prepare_inputs(train_X))


def test_pca_kronecker_convenience_model_supports_posterior() -> None:
    dtype = torch.double
    train_X = torch.rand(8, 5, dtype=dtype)
    train_Y = torch.stack([train_X[:, 0], train_X[:, 1]], dim=-1)
    model = PCAKroneckerMultiTaskGP(train_X, train_Y, n_components=2)

    posterior = model.posterior(train_X[:3])

    assert model.reduced_input_dim == 2
    assert posterior.mean.shape[-2:] == torch.Size([3, 2])
    assert model.make_mll() is not None


def test_random_projection_kronecker_supports_q_batch_posterior() -> None:
    dtype = torch.double
    train_X = torch.rand(8, 4, dtype=dtype)
    train_Y = torch.stack([train_X[:, 0], train_X[:, 1]], dim=-1)
    model = RandomProjectionKroneckerMultiTaskGP(
        train_X,
        train_Y,
        n_components=2,
        random_state=11,
    )
    candidates = torch.rand(2, 3, 4, dtype=dtype)

    posterior = model.posterior(candidates)

    assert posterior.mean.shape[-2:] == torch.Size([3, 2])


def test_pls_multitask_uses_long_format_outcomes_without_task_leakage() -> None:
    dtype = torch.double
    base = torch.linspace(0.0, 1.0, 8, dtype=dtype)
    data = torch.stack(
        [base, base.square(), torch.sin(base), torch.cos(base)],
        dim=-1,
    )
    data = data.repeat_interleave(2, dim=0)
    task = torch.tensor([0.0, 1.0] * 8, dtype=dtype).unsqueeze(-1)
    train_X = torch.cat([data[:, :2], task, data[:, 2:]], dim=-1)
    train_Y = 2.0 * data[:, :1] - data[:, 1:2] + 0.5 * task

    model = PLSMultiTaskGP(
        train_X,
        train_Y,
        task_feature=2,
        n_components=2,
    )

    assert model.input_reducer.input_dim == 4
    assert model.input_reducer.y_mean is not None
    assert model.input_reducer.y_mean.numel() == 1
    prepared = model._prepare_inputs(train_X)
    assert prepared.shape == torch.Size([16, 3])
    assert torch.equal(prepared[:, -1], task.squeeze(-1))
    assert model.make_mll() is not None


def test_pls_kronecker_uses_all_task_outputs_for_supervised_projection() -> None:
    dtype = torch.double
    base = torch.linspace(0.0, 1.0, 10, dtype=dtype)
    train_X = torch.stack(
        [base, base.square(), torch.sin(base), torch.cos(base), base**3],
        dim=-1,
    )
    train_Y = torch.stack(
        [2.0 * train_X[:, 0] - train_X[:, 1], train_X[:, 2] + 0.5 * train_X[:, 3]],
        dim=-1,
    )

    model = PLSKroneckerMultiTaskGP(
        train_X,
        train_Y,
        n_components=2,
    )

    assert model.input_reducer.input_dim == 5
    assert model.input_reducer.y_mean is not None
    assert model.input_reducer.y_mean.shape == torch.Size([2])
    assert model._prepare_inputs(train_X).shape == torch.Size([10, 2])
    posterior = model.posterior(train_X[:3])
    assert posterior.mean.shape[-2:] == torch.Size([3, 2])
    assert model.make_mll() is not None


def test_autoencoder_multitask_freezes_reducer_and_preserves_task_feature() -> None:
    dtype = torch.double
    data = torch.rand(12, 4, dtype=dtype)
    task = torch.tensor([0.0, 1.0] * 6, dtype=dtype).unsqueeze(-1)
    train_X = torch.cat([data[:, :2], task, data[:, 2:]], dim=-1)
    train_Y = data[:, :1] + 0.25 * task

    model = AutoEncoderMultiTaskGP(
        train_X,
        train_Y,
        task_feature=2,
        latent_dim=2,
        hidden_dims=(6,),
        epochs=2,
        random_state=7,
    )

    prepared = model._prepare_inputs(train_X)
    assert prepared.shape == torch.Size([12, 3])
    assert torch.equal(prepared[:, -1], task.squeeze(-1))
    assert model.input_reducer.encoder is not None
    assert all(
        not parameter.requires_grad for parameter in model.input_reducer.encoder.parameters()
    )
    assert model.make_mll() is not None


def test_autoencoder_kronecker_supports_q_batch_posterior() -> None:
    dtype = torch.double
    train_X = torch.rand(8, 4, dtype=dtype)
    train_Y = torch.stack([train_X[:, 0], train_X[:, 1]], dim=-1)
    model = AutoEncoderKroneckerMultiTaskGP(
        train_X,
        train_Y,
        latent_dim=2,
        hidden_dims=(6,),
        epochs=2,
        random_state=11,
    )

    posterior = model.posterior(torch.rand(2, 3, 4, dtype=dtype))

    assert posterior.mean.shape[-2:] == torch.Size([3, 2])
    assert model.input_reducer.encoder is not None
    assert all(not parameter.requires_grad for parameter in model.input_reducer.encoder.parameters())


def test_vae_multitask_uses_deterministic_posterior_mean_latent() -> None:
    dtype = torch.double
    data = torch.rand(10, 4, dtype=dtype)
    task = torch.tensor([0.0, 1.0] * 5, dtype=dtype).unsqueeze(-1)
    train_X = torch.cat([data, task], dim=-1)
    train_Y = data[:, :1]

    model = VAEMultiTaskGP(
        train_X,
        train_Y,
        task_feature=-1,
        latent_dim=2,
        hidden_dims=(6,),
        epochs=2,
        random_state=13,
    )

    first = model._prepare_inputs(train_X)
    second = model._prepare_inputs(train_X)

    assert torch.equal(first, second)
    assert torch.equal(first[:, -1], task.squeeze(-1))
    assert model.input_reducer.mu_head is not None
    assert all(
        not parameter.requires_grad for parameter in model.input_reducer.mu_head.parameters()
    )


def test_vae_kronecker_supports_posterior_and_mll() -> None:
    dtype = torch.double
    train_X = torch.rand(8, 4, dtype=dtype)
    train_Y = torch.stack([train_X[:, 0], train_X[:, 1]], dim=-1)
    model = VAEKroneckerMultiTaskGP(
        train_X,
        train_Y,
        latent_dim=2,
        hidden_dims=(6,),
        epochs=2,
        random_state=17,
    )

    posterior = model.posterior(train_X[:3])

    assert posterior.mean.shape[-2:] == torch.Size([3, 2])
    assert model.make_mll() is not None
