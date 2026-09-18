import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.sampling.normal import SobolQMCNormalSampler

from robotorchan.models import HybridAutoEncoderGP


def _data():
    torch.manual_seed(901)
    X = torch.rand(20, 7, dtype=torch.double)
    Y = X[:, :1] - 0.7 * X[:, 1:2] + torch.sin(torch.pi * X[:, 2:3])
    return X, Y


def _make_model(X, Y, *, reconstruction_weight=0.5, random_state=19):
    return HybridAutoEncoderGP(
        X,
        Y,
        latent_dim=3,
        hidden_dims=(8,),
        reconstruction_weight=reconstruction_weight,
        random_state=random_state,
    )


def test_hybrid_autoencoder_gp_reconstructs_original_input_shape():
    X, Y = _data()
    model = _make_model(X, Y)

    assert model.reconstruct(X).shape == X.shape
    assert model.reconstruction_loss().ndim == 0
    assert torch.isfinite(model.reconstruction_loss())
    assert model.make_mll().model is model


def test_training_loss_backpropagates_to_encoder_decoder_and_gp():
    X, Y = _data()
    model = _make_model(X, Y)

    loss = model.training_loss()
    loss.backward()

    encoder_grads = [parameter.grad for parameter in model.encoder.parameters()]
    decoder_grads = [parameter.grad for parameter in model.decoder.parameters()]
    gp_grads = [parameter.grad for parameter in model.covar_module.parameters()]

    assert any(gradient is not None and gradient.abs().sum() > 0 for gradient in encoder_grads)
    assert any(gradient is not None and gradient.abs().sum() > 0 for gradient in decoder_grads)
    assert any(gradient is not None and gradient.abs().sum() > 0 for gradient in gp_grads)




def test_zero_reconstruction_weight_removes_decoder_gradient():
    X, Y = _data()
    model = _make_model(X, Y, reconstruction_weight=0.0)

    model.training_loss().backward()

    assert all(parameter.grad is None for parameter in model.decoder.parameters())
    assert any(parameter.grad is not None for parameter in model.encoder.parameters())


def test_hybrid_autoencoder_gp_acquisition_keeps_original_x_gradients():
    X, Y = _data()
    model = _make_model(X, Y)
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


def test_hybrid_autoencoder_gp_state_dict_round_trip():
    X, Y = _data()
    source = _make_model(X, Y, random_state=19)
    target = _make_model(X, Y, random_state=91)

    target.load_state_dict(source.state_dict())

    torch.testing.assert_close(target.encode(X), source.encode(X))
    torch.testing.assert_close(target.reconstruct(X), source.reconstruct(X))
