from __future__ import annotations

import torch

from robotorchan.models import (
    MixedAutoEncoderGP,
    MixedSupervisedAutoEncoderGP,
    MixedSupervisedVAEGP,
    MixedVAEGP,
)


def _data() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(71)
    cont = torch.randn(12, 4, dtype=torch.double)
    cat = torch.randint(0, 3, (12, 1), dtype=torch.long).to(torch.double)
    X = torch.cat((cont[:, :2], cat, cont[:, 2:]), dim=-1)
    Y = (cont[:, :1] - 0.2 * cont[:, 1:2] + 0.1 * cat).to(torch.double)
    return X, Y


def test_neural_mixed_wrappers_are_colocated_with_standard_families() -> None:
    from robotorchan.models.high_dimensional.reduced import base, supervised_neural, vae

    assert MixedAutoEncoderGP.__module__ == base.__name__
    assert MixedSupervisedAutoEncoderGP.__module__ == supervised_neural.__name__
    assert MixedVAEGP.__module__ == vae.__name__
    assert MixedSupervisedVAEGP.__module__ == vae.__name__


def test_mixed_autoencoder_reduces_continuous_columns_only() -> None:
    X, Y = _data()
    model = MixedAutoEncoderGP(
        X,
        Y,
        latent_dim=2,
        cat_dims=[2],
        hidden_dims=(6,),
        epochs=1,
        batch_size=12,
        random_state=7,
    )

    assert model.input_reducer.input_dim == 4
    assert model.original_cat_dims == [2]
    assert model.reduced_cat_dims == [2]
    torch.testing.assert_close(model.train_inputs[0][..., 2], X[..., 2])
    torch.testing.assert_close(model.raw_train_X, X)


def test_mixed_vae_negative_cat_dim_and_raw_posterior() -> None:
    X, Y = _data()
    X = torch.cat((X[:, :2], X[:, 3:], X[:, 2:3]), dim=-1)
    model = MixedVAEGP(
        X,
        Y,
        latent_dim=2,
        cat_dims=[-1],
        hidden_dims=(6,),
        epochs=1,
        batch_size=12,
        random_state=7,
    )
    model.eval()
    model.likelihood.eval()

    posterior = model.posterior(X[:3])

    assert model.original_cat_dims == [4]
    assert model.input_reducer.input_dim == 4
    assert posterior.mean.shape == torch.Size([3, 1])


def test_supervised_neural_mixed_reducers_receive_continuous_inputs() -> None:
    X, Y = _data()
    ae = MixedSupervisedAutoEncoderGP(
        X,
        Y,
        latent_dim=2,
        cat_dims=[2],
        hidden_dims=(6,),
        epochs=1,
        batch_size=12,
        random_state=7,
    )
    vae = MixedSupervisedVAEGP(
        X,
        Y,
        latent_dim=2,
        cat_dims=[2],
        hidden_dims=(6,),
        epochs=1,
        batch_size=12,
        random_state=7,
    )

    assert ae.input_reducer.input_dim == 4
    assert vae.input_reducer.input_dim == 4
    torch.testing.assert_close(ae.train_inputs[0][..., 2], X[..., 2])
    torch.testing.assert_close(vae.train_inputs[0][..., 2], X[..., 2])
