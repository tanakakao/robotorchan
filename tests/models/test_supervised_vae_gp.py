import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.sampling.normal import SobolQMCNormalSampler

from robotorchan.models import SupervisedVAEGP, SupervisedVAEInputReducer


def _data():
    torch.manual_seed(1001)
    X = torch.rand(18, 6, dtype=torch.double)
    Y = 1.2 * X[:, :1] - 0.5 * X[:, 1:2] + torch.sin(torch.pi * X[:, 2:3])
    return X, Y


def _reducer():
    return SupervisedVAEInputReducer(
        latent_dim=2,
        hidden_dims=(7,),
        epochs=3,
        beta=0.2,
        supervised_weight=0.7,
        random_state=23,
    )


def test_supervised_vae_reducer_uses_y_and_exposes_losses():
    X, Y = _data()
    reducer = _reducer()
    reducer.fit(X, Y)

    assert reducer.transform(X).shape == (18, 2)
    assert reducer.predict_auxiliary(X).shape == Y.shape
    assert torch.isfinite(reducer.reconstruction_loss)
    assert torch.isfinite(reducer.kl_loss)
    assert torch.isfinite(reducer.supervised_loss)
    assert torch.isfinite(reducer.combined_loss)


def test_supervised_vae_reducer_requires_y():
    X, _ = _data()
    reducer = _reducer()

    try:
        reducer.fit(X)
    except ValueError as error:
        assert "requires paired Y" in str(error)
    else:
        raise AssertionError("Expected supervised VAE fitting without Y to fail.")


def test_supervised_vae_gp_preserves_original_space_acquisition_gradients():
    X, Y = _data()
    model = SupervisedVAEGP(
        X,
        Y,
        latent_dim=2,
        hidden_dims=(7,),
        epochs=3,
        beta=0.2,
        supervised_weight=0.7,
        random_state=23,
    )
    model.eval()

    candidate = X[:2].clone().requires_grad_(True)
    acquisition = qLogExpectedImprovement(
        model=model,
        best_f=Y.max(),
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([16])),
    )
    acquisition(candidate.unsqueeze(0)).sum().backward()

    assert candidate.grad is not None
    assert torch.isfinite(candidate.grad).all()
    assert model.make_mll().model is model


def test_supervised_vae_gp_state_dict_round_trip():
    X, Y = _data()
    kwargs = {
        "latent_dim": 2,
        "hidden_dims": (7,),
        "epochs": 3,
        "beta": 0.2,
        "supervised_weight": 0.7,
    }
    source = SupervisedVAEGP(X, Y, random_state=23, **kwargs)
    target = SupervisedVAEGP(X, Y, random_state=71, **kwargs)

    target.load_state_dict(source.state_dict())

    torch.testing.assert_close(
        target.input_reducer.transform(X),
        source.input_reducer.transform(X),
    )
    source.eval()
    target.eval()
    torch.testing.assert_close(target.posterior(X[:3]).mean, source.posterior(X[:3]).mean)
