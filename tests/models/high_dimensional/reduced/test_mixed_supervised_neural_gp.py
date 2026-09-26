from __future__ import annotations

import torch

from robotorchan.models import MixedSupervisedAutoEncoderGP, MixedSupervisedVAEGP
from robotorchan.reduction import SupervisedAutoEncoderInputReducer, SupervisedVAEInputReducer


def _data() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(607)
    continuous = torch.rand(20, 6, dtype=torch.double)
    category = torch.randint(0, 3, (20, 1)).to(dtype=torch.double)
    X = torch.cat((continuous[:, :2], category, continuous[:, 2:]), dim=-1)
    Y = torch.sin(2 * torch.pi * continuous[:, :1]) + 0.3 * category
    return X, Y


def test_mixed_supervised_autoencoder_uses_continuous_x_and_outcomes() -> None:
    X, Y = _data()
    model = MixedSupervisedAutoEncoderGP(
        X, Y, latent_dim=2, cat_dims=[2], hidden_dims=(8,), epochs=5, random_state=3
    )

    assert isinstance(model.input_reducer, SupervisedAutoEncoderInputReducer)
    assert model.input_reducer.input_dim == 6
    assert model.reduced_cat_dims == [2]
    torch.testing.assert_close(model.train_inputs[0][..., 2], X[..., 2])
    torch.testing.assert_close(model.raw_train_X, X)
    torch.testing.assert_close(model.raw_train_Y, Y)


def test_mixed_supervised_vae_preserves_categories_and_original_posterior_api() -> None:
    X, Y = _data()
    model = MixedSupervisedVAEGP(
        X,
        Y,
        latent_dim=2,
        cat_dims=[2],
        hidden_dims=(8,),
        epochs=5,
        beta=0.2,
        random_state=5,
    )
    assert isinstance(model.input_reducer, SupervisedVAEInputReducer)
    assert model.input_reducer.input_dim == 6
    torch.testing.assert_close(model.train_inputs[0][..., 2], X[..., 2])

    candidates = X[:4].clone()
    model.eval()
    model.likelihood.eval()
    posterior = model.posterior(candidates)
    assert posterior.mean.shape == torch.Size([4, 1])
    assert torch.isfinite(posterior.mean).all()


def test_mixed_supervised_state_dict_round_trip() -> None:
    X, Y = _data()
    kwargs = dict(latent_dim=2, cat_dims=[2], hidden_dims=(8,), epochs=5)
    source = MixedSupervisedAutoEncoderGP(X, Y, random_state=7, **kwargs)
    target = MixedSupervisedAutoEncoderGP(X, Y, random_state=11, **kwargs)
    target.load_state_dict(source.state_dict())
    source.eval()
    target.eval()
    source.likelihood.eval()
    target.likelihood.eval()

    expected = source.posterior(X[:3])
    actual = target.posterior(X[:3])
    torch.testing.assert_close(actual.mean, expected.mean)
    torch.testing.assert_close(actual.variance, expected.variance)
