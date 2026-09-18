import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.sampling.normal import SobolQMCNormalSampler

from robotorchan.models import JointVAEGP


def _data():
    torch.manual_seed(1010)
    X = torch.rand(18, 7, dtype=torch.double)
    Y = 1.2 * X[:, :1] - 0.7 * X[:, 1:2] + torch.sin(torch.pi * X[:, 2:3])
    return X, Y


def _model(X, Y, *, random_state=19):
    return JointVAEGP(
        X,
        Y,
        latent_dim=3,
        hidden_dims=(8,),
        random_state=random_state,
        beta=0.2,
        reconstruction_weight=0.5,
    )


def test_joint_vae_gp_distribution_and_sampling_shapes():
    X, Y = _data()
    model = _model(X, Y)

    mu, logvar = model.encode_distribution(X[:4])
    assert mu.shape == (4, 3)
    assert logvar.shape == (4, 3)
    assert model.sample_latent(X[:4], 5).shape == (5, 4, 3)
    assert model.reconstruct(X[:4]).shape == (4, 7)


def test_training_loss_backpropagates_to_variational_and_decoder_parameters():
    X, Y = _data()
    model = _model(X, Y)

    loss = model.training_loss()
    loss.backward()

    assert model.mu_head.weight.grad is not None
    assert model.logvar_head.weight.grad is not None
    assert next(model.decoder.parameters()).grad is not None
    assert torch.isfinite(model.mu_head.weight.grad).all()
    assert torch.isfinite(model.logvar_head.weight.grad).all()


def test_joint_vae_gp_posterior_and_acquisition_keep_original_x_gradients():
    X, Y = _data()
    model = _model(X, Y)
    model.eval()

    candidate = X[:2].clone().requires_grad_(True)
    assert model.posterior(candidate).mean.shape == (2, 1)

    acquisition = qLogExpectedImprovement(
        model=model,
        best_f=Y.max(),
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([16])),
    )
    acquisition(candidate.unsqueeze(0)).sum().backward()

    assert candidate.grad is not None
    assert torch.isfinite(candidate.grad).all()


def test_joint_vae_gp_state_dict_round_trip():
    X, Y = _data()
    source = _model(X, Y, random_state=19)
    target = _model(X, Y, random_state=91)

    target.load_state_dict(source.state_dict())

    source_mu, source_logvar = source.encode_distribution(X)
    target_mu, target_logvar = target.encode_distribution(X)
    torch.testing.assert_close(target_mu, source_mu)
    torch.testing.assert_close(target_logvar, source_logvar)
    source.eval()
    target.eval()
    torch.testing.assert_close(target.posterior(X[:4]).mean, source.posterior(X[:4]).mean)
