import torch

from robotorchan.models import SupervisedVAEGP
from robotorchan.models.supervised_neural_reduction import SupervisedVAEInputReducer


def _data():
    torch.manual_seed(1001)
    X = torch.rand(20, 7, dtype=torch.double)
    Y = 1.4 * X[:, :1] - 0.6 * X[:, 1:2] + torch.sin(torch.pi * X[:, 2:3])
    return X, Y


def test_supervised_vae_reducer_requires_y_and_tracks_losses():
    X, Y = _data()
    reducer = SupervisedVAEInputReducer(
        latent_dim=3,
        hidden_dims=(8,),
        epochs=3,
        random_state=17,
    )

    reducer.fit(X, Y)

    assert reducer.transform(X).shape == (20, 3)
    assert torch.isfinite(reducer.reconstruction_loss)
    assert torch.isfinite(reducer.kl_loss)
    assert torch.isfinite(reducer.supervised_loss)
    assert torch.isfinite(reducer.combined_loss)
    assert reducer.predict_auxiliary(X).shape == Y.shape


def test_supervised_vae_gp_uses_frozen_supervised_representation():
    X, Y = _data()
    model = SupervisedVAEGP(
        X,
        Y,
        latent_dim=3,
        hidden_dims=(8,),
        epochs=3,
        random_state=17,
    )

    reducer = model.input_reducer
    assert isinstance(reducer, SupervisedVAEInputReducer)
    assert model.train_inputs[0].shape == (20, 3)
    assert model.posterior(X[:4]).mean.shape == (4, 1)
    assert model.make_mll().model is model
    assert all(not parameter.requires_grad for parameter in reducer.parameters())


def test_supervised_vae_gp_state_dict_round_trip():
    X, Y = _data()
    source = SupervisedVAEGP(
        X,
        Y,
        latent_dim=3,
        hidden_dims=(8,),
        epochs=2,
        random_state=17,
    )
    target = SupervisedVAEGP(
        X,
        Y,
        latent_dim=3,
        hidden_dims=(8,),
        epochs=2,
        random_state=71,
    )

    target.load_state_dict(source.state_dict())

    torch.testing.assert_close(target.input_reducer.transform(X), source.input_reducer.transform(X))
    source.eval()
    target.eval()
    torch.testing.assert_close(target.posterior(X[:4]).mean, source.posterior(X[:4]).mean)
