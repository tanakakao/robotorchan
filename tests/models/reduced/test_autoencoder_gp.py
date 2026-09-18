from __future__ import annotations

import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.mlls import ExactMarginalLogLikelihood

from robotorchan.models import AutoEncoderGP
from robotorchan.reduction import AutoEncoderInputReducer


def _training_data() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(311)
    train_X = torch.rand(18, 8, dtype=torch.double)
    train_Y = (
        torch.sin(2.0 * torch.pi * train_X[:, :1])
        + 0.4 * train_X[:, 1:2]
        - 0.2 * train_X[:, 2:3].square()
    )
    return train_X, train_Y


def _make_model() -> tuple[AutoEncoderGP, torch.Tensor, torch.Tensor]:
    train_X, train_Y = _training_data()
    model = AutoEncoderGP(
        train_X=train_X,
        train_Y=train_Y,
        latent_dim=3,
        hidden_dims=(10,),
        epochs=10,
        learning_rate=5e-3,
        random_state=17,
    )
    return model, train_X, train_Y


def test_autoencoder_gp_preserves_raw_data_and_uses_latent_inputs() -> None:
    model, train_X, train_Y = _make_model()

    assert isinstance(model.input_reducer, AutoEncoderInputReducer)
    assert model.input_reducer.is_fitted is True
    assert model.original_input_dim == 8
    assert model.reduced_input_dim == 3
    assert torch.equal(model.raw_train_X, train_X)
    assert torch.equal(model.raw_train_Y, train_Y)
    assert model.train_inputs[0].shape == torch.Size([18, 3])
    assert model.input_reducer.encoder is not None
    assert all(
        not parameter.requires_grad for parameter in model.input_reducer.encoder.parameters()
    )


def test_autoencoder_gp_make_mll_and_q_batch_posterior() -> None:
    model, _, _ = _make_model()
    mll = model.make_mll()

    assert isinstance(mll, ExactMarginalLogLikelihood)

    model.eval()
    posterior = model.posterior(torch.rand(2, 4, 8, dtype=torch.double))

    assert posterior.mean.shape == torch.Size([2, 4, 1])
    assert posterior.variance.shape == torch.Size([2, 4, 1])
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all()


def test_autoencoder_gp_acquisition_gradient_flows_to_original_inputs() -> None:
    model, _, train_Y = _make_model()
    model.eval()
    model.likelihood.eval()
    acquisition = qLogExpectedImprovement(
        model=model,
        best_f=train_Y.max(),
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([16]), seed=13),
    )
    X = torch.rand(2, 8, dtype=torch.double, requires_grad=True)

    value = acquisition(X)
    value.sum().backward()

    assert X.grad is not None
    assert X.grad.shape == X.shape
    assert torch.isfinite(X.grad).all()


def test_autoencoder_gp_conditioning_preserves_frozen_reducer() -> None:
    model, _, _ = _make_model()
    assert model.input_reducer is not None
    assert model.input_reducer.encoder is not None
    before = [parameter.detach().clone() for parameter in model.input_reducer.encoder.parameters()]
    new_X = torch.rand(2, 8, dtype=torch.double)
    new_Y = torch.sin(2.0 * torch.pi * new_X[:, :1])

    model.eval()
    _ = model.posterior(new_X)
    conditioned = model.condition_on_observations(X=new_X, Y=new_Y)

    assert isinstance(conditioned, AutoEncoderGP)
    assert conditioned.input_reducer is not None
    assert conditioned.input_reducer.encoder is not None
    after = list(conditioned.input_reducer.encoder.parameters())
    for expected, actual in zip(before, after, strict=True):
        torch.testing.assert_close(actual, expected)


def test_autoencoder_gp_state_dict_round_trip_preserves_posterior() -> None:
    source, train_X, train_Y = _make_model()
    target = AutoEncoderGP(
        train_X=train_X,
        train_Y=train_Y,
        latent_dim=3,
        hidden_dims=(10,),
        epochs=10,
        learning_rate=5e-3,
        random_state=91,
    )
    target.load_state_dict(source.state_dict())

    test_X = torch.rand(5, 8, dtype=torch.double)
    source.eval()
    target.eval()
    source.likelihood.eval()
    target.likelihood.eval()

    source_posterior = source.posterior(test_X)
    target_posterior = target.posterior(test_X)

    torch.testing.assert_close(target_posterior.mean, source_posterior.mean)
    torch.testing.assert_close(target_posterior.variance, source_posterior.variance)
