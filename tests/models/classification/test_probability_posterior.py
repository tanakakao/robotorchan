"""BoTorch Posterior semantics for binary class-probability predictions."""

import torch
from botorch.posteriors import Posterior

from robotorchan.models.classification import (
    BinarySingleTaskGPClassifier,
    ClassificationProbabilityPosterior,
    LabelNoiseBinarySingleTaskGPClassifier,
)


def _model(noisy: bool = False) -> BinarySingleTaskGPClassifier:
    train_x = torch.tensor([[0.0], [0.3], [0.7], [1.0]], dtype=torch.double)
    train_y = torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.double)
    if noisy:
        model = LabelNoiseBinarySingleTaskGPClassifier(
            train_x,
            train_y,
            false_positive_rate=0.1,
            false_negative_rate=0.2,
            inducing_points=4,
        )
    else:
        model = BinarySingleTaskGPClassifier(train_x, train_y, inducing_points=4)
    model.eval()
    model.likelihood.eval()
    return model


@torch.no_grad()
def test_binary_probability_posterior_matches_legacy_moments() -> None:
    model = _model()
    for X in (
        torch.tensor([[[0.3]], [[0.7]]], dtype=torch.double),
        torch.tensor([[[0.3], [0.7]]], dtype=torch.double),
    ):
        posterior = model.probability_posterior(X)
        assert isinstance(posterior, Posterior)
        assert isinstance(posterior, ClassificationProbabilityPosterior)
        assert posterior.mean.shape == (*X.shape[:-1], 2)
        assert posterior.variance.shape == posterior.mean.shape
        predicted = model.predict_proba(X)
        variance = model.probability_variance(X)
        if X.shape[-2] == 1:
            predicted = predicted.unsqueeze(-2)
            variance = variance.unsqueeze(-2)
        torch.testing.assert_close(posterior.mean, predicted)
        torch.testing.assert_close(posterior.variance, variance)
        samples = posterior.rsample(torch.Size([5]))
        assert samples.shape == (5, *X.shape[:-1], 2)
        torch.testing.assert_close(samples.sum(dim=-1), torch.ones_like(samples[..., 0]))

        base = torch.randn(
            (3, *posterior.base_sample_shape),
            dtype=posterior.dtype,
            device=posterior.device,
        )
        first = posterior.rsample_from_base_samples(torch.Size([3]), base)
        second = posterior.rsample_from_base_samples(torch.Size([3]), base)
        torch.testing.assert_close(first, second, atol=0.0, rtol=0.0)
        assert first.shape == (3, *X.shape[:-1], 2)


@torch.no_grad()
def test_binary_probability_interval_matches_monotone_probit_quantiles() -> None:
    model = _model()
    X = torch.tensor([[[0.2], [0.5], [0.8]]], dtype=torch.double)
    posterior = model.probability_posterior(X)
    lower, upper = posterior.credible_interval(0.95)
    latent = model.latent_posterior(X)
    z = torch.special.ndtri(torch.tensor(0.975, dtype=torch.double))
    sigma = latent.variance.sqrt().squeeze(-1)
    mu = latent.mean.squeeze(-1)
    reference_lower = torch.special.ndtr(mu - z * sigma)
    reference_upper = torch.special.ndtr(mu + z * sigma)

    assert lower.shape == posterior.mean.shape
    assert upper.shape == posterior.mean.shape
    torch.testing.assert_close(lower[..., 1], reference_lower)
    torch.testing.assert_close(upper[..., 1], reference_upper)
    torch.testing.assert_close(lower[..., 0], 1.0 - reference_upper)
    torch.testing.assert_close(upper[..., 0], 1.0 - reference_lower)
    assert ((lower <= posterior.mean) & (posterior.mean <= upper)).all()


@torch.no_grad()
def test_observed_label_noise_is_included_in_probability_interval() -> None:
    model = _model(noisy=True)
    X = torch.tensor([[[0.2], [0.5]]], dtype=torch.double)
    posterior = model.probability_posterior(X)
    lower, upper = posterior.credible_interval()
    latent = model.latent_posterior(X)
    z = torch.special.ndtri(torch.tensor(0.975, dtype=torch.double))
    clean_lower = torch.special.ndtr((latent.mean - z * latent.variance.sqrt()).squeeze(-1))
    clean_upper = torch.special.ndtr((latent.mean + z * latent.variance.sqrt()).squeeze(-1))
    torch.testing.assert_close(lower[..., 1], 0.1 + 0.7 * clean_lower)
    torch.testing.assert_close(upper[..., 1], 0.1 + 0.7 * clean_upper)


def test_probability_posterior_preserves_gradient_path() -> None:
    model = _model()
    X = torch.tensor([[[0.25]], [[0.75]]], dtype=torch.double, requires_grad=True)
    posterior = model.probability_posterior(X)
    (posterior.mean[..., 1] + posterior.variance[..., 1]).sum().backward()
    assert X.grad is not None
    assert torch.isfinite(X.grad).all()
