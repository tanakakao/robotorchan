from __future__ import annotations

import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.sampling.normal import SobolQMCNormalSampler

from robotorchan.models import VAEGP
from robotorchan.models.neural_reduction import VAEInputReducer


def _training_data() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(409)
    train_X = torch.rand(18, 8, dtype=torch.double)
    train_Y = torch.sin(2.0 * torch.pi * train_X[:, :1]) + 0.3 * train_X[:, 1:2]
    return train_X, train_Y


def _make_model(random_state: int = 31) -> tuple[VAEGP, torch.Tensor, torch.Tensor]:
    train_X, train_Y = _training_data()
    model = VAEGP(
        train_X=train_X,
        train_Y=train_Y,
        latent_dim=3,
        hidden_dims=(10,),
        epochs=10,
        learning_rate=5e-3,
        beta=0.2,
        random_state=random_state,
    )
    return model, train_X, train_Y


def test_vae_gp_uses_frozen_posterior_mean_latent_inputs() -> None:
    model, train_X, train_Y = _make_model()

    assert isinstance(model.input_reducer, VAEInputReducer)
    assert model.reduced_input_dim == 3
    assert torch.equal(model.raw_train_X, train_X)
    assert torch.equal(model.raw_train_Y, train_Y)
    torch.testing.assert_close(model.train_inputs[0], model.input_reducer.transform(train_X))


def test_vae_gp_posterior_and_acquisition_gradient_use_original_space() -> None:
    model, _, train_Y = _make_model()
    model.eval()
    model.likelihood.eval()
    X = torch.rand(2, 8, dtype=torch.double, requires_grad=True)

    posterior = model.posterior(X)
    acquisition = qLogExpectedImprovement(
        model=model,
        best_f=train_Y.max(),
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([16]), seed=17),
    )
    acquisition(X).sum().backward()

    assert posterior.mean.shape == torch.Size([2, 1])
    assert X.grad is not None
    assert torch.isfinite(X.grad).all()


def test_vae_gp_state_dict_round_trip_preserves_posterior() -> None:
    source, train_X, train_Y = _make_model(random_state=31)
    target = VAEGP(
        train_X=train_X,
        train_Y=train_Y,
        latent_dim=3,
        hidden_dims=(10,),
        epochs=10,
        learning_rate=5e-3,
        beta=0.2,
        random_state=91,
    )
    target.load_state_dict(source.state_dict())
    test_X = torch.rand(4, 8, dtype=torch.double)
    source.eval()
    target.eval()
    source.likelihood.eval()
    target.likelihood.eval()

    source_posterior = source.posterior(test_X)
    target_posterior = target.posterior(test_X)
    torch.testing.assert_close(target_posterior.mean, source_posterior.mean)
    torch.testing.assert_close(target_posterior.variance, source_posterior.variance)
