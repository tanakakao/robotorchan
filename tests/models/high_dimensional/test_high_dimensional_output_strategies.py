from __future__ import annotations

import torch

from robotorchan.models import (
    HigherOrderGP,
    KroneckerMultiTaskGP,
    LatentKroneckerGP,
    OutputPCAGP,
)


def _vector_training_data() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(31)
    train_X = torch.rand(12, 2, dtype=torch.double)
    x0 = train_X[:, 0]
    x1 = train_X[:, 1]
    train_Y = torch.stack(
        [
            x0 + x1,
            x0 - x1,
            x0.square(),
            x1.square(),
            torch.sin(x0),
            torch.cos(x1),
        ],
        dim=-1,
    )
    return train_X, train_Y


def test_output_pca_gp_compresses_vector_outputs_but_restores_public_dimension() -> None:
    train_X, train_Y = _vector_training_data()
    model = OutputPCAGP(train_X=train_X, train_Y=train_Y, n_components=3)
    model.eval()
    model.likelihood.eval()

    posterior = model.posterior(torch.rand(4, 2, dtype=torch.double))

    assert model.original_output_dim == 6
    assert model.reduced_output_dim == 3
    assert posterior.mean.shape == torch.Size([4, 6])
    assert posterior.variance.shape == torch.Size([4, 6])


def test_higher_order_gp_preserves_native_tensor_output_structure() -> None:
    train_X, flat_Y = _vector_training_data()
    train_Y = flat_Y.reshape(12, 2, 3)
    model = HigherOrderGP(train_X=train_X, train_Y=train_Y)
    model.eval()

    posterior = model.posterior(torch.rand(4, 2, dtype=torch.double))

    assert model.raw_train_Y.shape == torch.Size([12, 2, 3])
    assert posterior.mean.shape == torch.Size([4, 2, 3])
    assert posterior.variance.shape == torch.Size([4, 2, 3])


def test_kronecker_multitask_gp_keeps_block_design_outputs_as_tasks() -> None:
    train_X, train_Y = _vector_training_data()
    model = KroneckerMultiTaskGP(train_X=train_X, train_Y=train_Y)
    model.eval()
    model.likelihood.eval()

    posterior = model.posterior(torch.rand(4, 2, dtype=torch.double))

    assert model.raw_train_Y.shape == torch.Size([12, 6])
    assert model.num_outputs == 6
    assert posterior.mean.shape == torch.Size([4, 6])


def test_latent_kronecker_gp_keeps_explicit_output_coordinates() -> None:
    torch.manual_seed(37)
    train_X = torch.rand(8, 2, dtype=torch.double)
    train_T = torch.linspace(0.0, 1.0, 6, dtype=torch.double).unsqueeze(-1)
    train_Y = (
        train_X[:, :1]
        + 0.5 * train_X[:, 1:2]
        + torch.sin(2.0 * torch.pi * train_T.squeeze(-1)).unsqueeze(0)
    )
    train_Y[2, 4] = torch.nan

    model = LatentKroneckerGP(
        train_X=train_X,
        train_T=train_T,
        train_Y=train_Y,
    )

    assert model.raw_train_T.shape == torch.Size([6, 1])
    torch.testing.assert_close(model.raw_train_T, train_T)
    assert torch.isnan(model.raw_train_Y[2, 4])
