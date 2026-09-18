import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.sampling.normal import SobolQMCNormalSampler

from robotorchan.models import SupervisedAutoEncoderGP
from robotorchan.reduction import SupervisedAutoEncoderInputReducer


def _data():
    torch.manual_seed(701)
    X = torch.rand(24, 8, dtype=torch.double)
    Y = 1.5 * X[:, :1] - 0.8 * X[:, 1:2] + torch.sin(torch.pi * X[:, 2:3])
    return X, Y


def _make_model(X, Y, *, random_state=17):
    return SupervisedAutoEncoderGP(
        X,
        Y,
        latent_dim=3,
        hidden_dims=(10,),
        epochs=12,
        learning_rate=5e-3,
        random_state=random_state,
        supervised_weight=1.0,
    )


def test_supervised_autoencoder_gp_uses_supervised_reducer_and_raw_data():
    X, Y = _data()
    model = _make_model(X, Y)

    assert isinstance(model.input_reducer, SupervisedAutoEncoderInputReducer)
    torch.testing.assert_close(model.raw_train_X, X)
    torch.testing.assert_close(model.raw_train_Y, Y)
    torch.testing.assert_close(model.train_inputs[0], model.input_reducer.transform(X))
    assert model.make_mll().model is model


def test_supervised_autoencoder_gp_posterior_and_acquisition_keep_original_x_gradients():
    X, Y = _data()
    model = _make_model(X, Y)
    model.eval()

    candidate = X[:2].clone().requires_grad_(True)
    posterior = model.posterior(candidate)
    assert posterior.mean.shape == (2, 1)

    acquisition = qLogExpectedImprovement(
        model=model,
        best_f=Y.max(),
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([16])),
    )
    value = acquisition(candidate.unsqueeze(0))
    value.sum().backward()

    assert candidate.grad is not None
    assert candidate.grad.shape == candidate.shape
    assert torch.isfinite(candidate.grad).all()


def test_supervised_autoencoder_gp_state_dict_round_trip_resynchronizes_latent_training_data():
    X, Y = _data()
    source = _make_model(X, Y, random_state=17)
    target = _make_model(X, Y, random_state=99)

    target.load_state_dict(source.state_dict())

    torch.testing.assert_close(
        target.input_reducer.transform(X),
        source.input_reducer.transform(X),
    )
    torch.testing.assert_close(target.train_inputs[0], source.train_inputs[0])

    source.eval()
    target.eval()
    torch.testing.assert_close(
        target.posterior(X[:4]).mean,
        source.posterior(X[:4]).mean,
    )
