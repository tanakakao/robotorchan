import torch

from robotorchan.models import JointVAEGP


def _data():
    torch.manual_seed(1011)
    X = torch.rand(18, 7, dtype=torch.double)
    Y = 1.2 * X[:, :1] - 0.7 * X[:, 1:2] + torch.sin(torch.pi * X[:, 2:3])
    return X, Y


def _model(X, Y):
    model = JointVAEGP(
        X,
        Y,
        latent_dim=3,
        hidden_dims=(8,),
        random_state=19,
        beta=0.2,
        reconstruction_weight=0.5,
    )
    model.eval()
    return model


def test_uncertainty_aware_posterior_has_original_output_shape():
    X, Y = _data()
    model = _model(X, Y)

    torch.manual_seed(5)
    posterior = model.uncertainty_aware_posterior(X[:4], n_latent_samples=8)

    assert posterior.mean.shape == (4, 1)
    assert posterior.variance.shape == (4, 1)
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all()
    assert (posterior.variance >= 0).all()


def test_uncertainty_aware_posterior_supports_batched_q_inputs():
    X, Y = _data()
    model = _model(X, Y)
    candidates = X[:6].reshape(2, 3, 7)

    torch.manual_seed(7)
    posterior = model.uncertainty_aware_posterior(candidates, n_latent_samples=6)

    assert posterior.mean.shape == (2, 3, 1)
    assert posterior.variance.shape == (2, 3, 1)


def test_uncertainty_aware_posterior_propagates_original_x_gradients():
    X, Y = _data()
    model = _model(X, Y)
    candidate = X[:2].clone().requires_grad_(True)

    torch.manual_seed(11)
    posterior = model.uncertainty_aware_posterior(candidate, n_latent_samples=6)
    (posterior.mean.sum() + posterior.variance.sum()).backward()

    assert candidate.grad is not None
    assert torch.isfinite(candidate.grad).all()
    assert candidate.grad.abs().sum() > 0


def test_latent_uncertainty_increases_moment_matched_variance_when_spread_grows():
    X, Y = _data()
    model = _model(X, Y)
    candidate = X[:3]

    with torch.no_grad():
        model.logvar_head.weight.zero_()
        model.logvar_head.bias.fill_(-12.0)
    torch.manual_seed(13)
    low = model.uncertainty_aware_posterior(candidate, n_latent_samples=64)

    with torch.no_grad():
        model.logvar_head.bias.fill_(0.5)
    torch.manual_seed(13)
    high = model.uncertainty_aware_posterior(candidate, n_latent_samples=64)

    assert high.variance.mean() > low.variance.mean()


def test_uncertainty_aware_posterior_validates_sample_count():
    X, Y = _data()
    model = _model(X, Y)

    try:
        model.uncertainty_aware_posterior(X[:2], n_latent_samples=0)
    except ValueError as error:
        assert "positive integer" in str(error)
    else:
        raise AssertionError("Expected ValueError for non-positive latent sample count.")
