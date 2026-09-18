from __future__ import annotations

import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.sampling.normal import SobolQMCNormalSampler

from robotorchan.models import MixedAutoEncoderGP, MixedVAEGP
from robotorchan.reduction import AutoEncoderInputReducer, VAEInputReducer


def _training_data() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(503)
    continuous = torch.rand(18, 6, dtype=torch.double)
    category = torch.randint(0, 3, (18, 1)).to(dtype=torch.double)
    train_X = torch.cat((continuous[:, :3], category, continuous[:, 3:]), dim=-1)
    train_Y = torch.sin(2.0 * torch.pi * continuous[:, :1]) + 0.2 * category
    return train_X, train_Y


def _candidates(n: int, *, requires_grad: bool = False) -> torch.Tensor:
    continuous = torch.rand(n, 6, dtype=torch.double)
    category = torch.randint(0, 3, (n, 1)).to(dtype=torch.double)
    X = torch.cat((continuous[:, :3], category, continuous[:, 3:]), dim=-1)
    return X.requires_grad_(requires_grad)


def test_mixed_autoencoder_gp_reduces_only_continuous_inputs() -> None:
    train_X, train_Y = _training_data()
    model = MixedAutoEncoderGP(
        train_X,
        train_Y,
        latent_dim=2,
        cat_dims=[3],
        hidden_dims=(8,),
        epochs=5,
        random_state=17,
    )

    assert isinstance(model.input_reducer, AutoEncoderInputReducer)
    assert model.input_reducer.input_dim == 6
    assert model.original_input_dim == 7
    assert model.reduced_input_dim == 3
    assert model.reduced_cat_dims == [2]
    torch.testing.assert_close(model.raw_train_X, train_X)
    torch.testing.assert_close(model.train_inputs[0][..., 2], train_X[..., 3])


def test_mixed_vae_gp_uses_continuous_posterior_mean_latent_inputs() -> None:
    train_X, train_Y = _training_data()
    model = MixedVAEGP(
        train_X,
        train_Y,
        latent_dim=2,
        cat_dims=[3],
        hidden_dims=(8,),
        epochs=5,
        beta=0.2,
        random_state=19,
    )

    assert isinstance(model.input_reducer, VAEInputReducer)
    continuous = torch.cat((train_X[:, :3], train_X[:, 4:]), dim=-1)
    latent = model.input_reducer.transform(continuous)
    torch.testing.assert_close(model.train_inputs[0][..., :2], latent)
    torch.testing.assert_close(model.train_inputs[0][..., 2], train_X[..., 3])


def test_mixed_neural_gp_posterior_accepts_original_q_batch() -> None:
    train_X, train_Y = _training_data()
    model = MixedAutoEncoderGP(
        train_X,
        train_Y,
        latent_dim=2,
        cat_dims=[3],
        hidden_dims=(8,),
        epochs=5,
    )
    model.eval()
    model.likelihood.eval()

    posterior = model.posterior(_candidates(8).reshape(2, 4, 7))

    assert posterior.mean.shape == torch.Size([2, 4, 1])
    assert posterior.variance.shape == torch.Size([2, 4, 1])


def test_mixed_neural_acquisition_gradient_flows_through_continuous_encoder() -> None:
    train_X, train_Y = _training_data()
    model = MixedAutoEncoderGP(
        train_X,
        train_Y,
        latent_dim=2,
        cat_dims=[3],
        hidden_dims=(8,),
        epochs=5,
    )
    model.eval()
    model.likelihood.eval()
    acquisition = qLogExpectedImprovement(
        model=model,
        best_f=train_Y.max(),
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([8]), seed=23),
    )
    X = _candidates(2, requires_grad=True)

    acquisition(X).sum().backward()

    assert X.grad is not None
    assert torch.isfinite(X.grad).all()
    assert torch.count_nonzero(X.grad[:, [0, 1, 2, 4, 5, 6]]) > 0


def test_mixed_neural_state_dict_round_trip_preserves_posterior() -> None:
    train_X, train_Y = _training_data()
    source = MixedVAEGP(
        train_X,
        train_Y,
        latent_dim=2,
        cat_dims=[3],
        hidden_dims=(8,),
        epochs=5,
        beta=0.2,
        random_state=29,
    )
    target = MixedVAEGP(
        train_X,
        train_Y,
        latent_dim=2,
        cat_dims=[3],
        hidden_dims=(8,),
        epochs=5,
        beta=0.2,
        random_state=31,
    )
    target.load_state_dict(source.state_dict())
    source.eval()
    source.likelihood.eval()
    target.eval()
    target.likelihood.eval()
    X = _candidates(4)

    source_posterior = source.posterior(X)
    target_posterior = target.posterior(X)

    torch.testing.assert_close(target_posterior.mean, source_posterior.mean)
    torch.testing.assert_close(target_posterior.variance, source_posterior.variance)
