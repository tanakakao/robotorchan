import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.sampling.normal import SobolQMCNormalSampler

from robotorchan.models import JointEncoderGP


def _data():
    torch.manual_seed(801)
    X = torch.rand(20, 7, dtype=torch.double)
    Y = 1.3 * X[:, :1] - X[:, 1:2] + torch.sin(torch.pi * X[:, 2:3])
    return X, Y


def _make_model(X, Y, *, random_state=13):
    return JointEncoderGP(
        X,
        Y,
        latent_dim=3,
        hidden_dims=(8,),
        random_state=random_state,
    )


def test_joint_encoder_gp_keeps_original_training_inputs_and_raw_data():
    X, Y = _data()
    model = _make_model(X, Y)

    torch.testing.assert_close(model.raw_train_X, X)
    torch.testing.assert_close(model.raw_train_Y, Y)
    torch.testing.assert_close(model.train_inputs[0], X)
    assert model.encode(X).shape == (20, 3)
    assert model.make_mll().model is model


def test_joint_encoder_gp_mll_backpropagates_to_encoder():
    X, Y = _data()
    model = _make_model(X, Y)
    model.train()
    model.likelihood.train()

    mll = model.make_mll()
    output = model(X)
    loss = -mll(output, model.train_targets)
    loss.backward()

    encoder_gradients = [parameter.grad for parameter in model.encoder.parameters()]
    assert all(gradient is not None for gradient in encoder_gradients)
    assert all(
        torch.isfinite(gradient).all()
        for gradient in encoder_gradients
        if gradient is not None
    )
    assert any(gradient.abs().sum() > 0 for gradient in encoder_gradients if gradient is not None)


def test_joint_encoder_gp_posterior_and_acquisition_keep_original_x_gradients():
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
    assert torch.isfinite(candidate.grad).all()


def test_joint_encoder_gp_state_dict_round_trip():
    X, Y = _data()
    source = _make_model(X, Y, random_state=13)
    target = _make_model(X, Y, random_state=99)

    target.load_state_dict(source.state_dict())

    torch.testing.assert_close(target.encode(X), source.encode(X))
    source.eval()
    target.eval()
    torch.testing.assert_close(target.posterior(X[:4]).mean, source.posterior(X[:4]).mean)
