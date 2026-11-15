"""Probability-posterior contracts for ordinal and multiclass GP classifiers."""

import math

import pytest
import torch
from botorch.posteriors import Posterior

from robotorchan.models.classification import (
    MixedMulticlassSingleTaskGPClassifier,
    MixedOrdinalSingleTaskGPClassifier,
    MulticlassSingleTaskGPClassifier,
    OrdinalSingleTaskGPClassifier,
)


def _model(kind: str, mixed: bool) -> object:
    train_x = torch.tensor([[0.0], [0.3], [0.6], [1.0]], dtype=torch.double)
    if mixed:
        train_x = torch.cat(
            (train_x, torch.tensor([[0.0], [1.0], [0.0], [1.0]], dtype=torch.double)),
            dim=-1,
        )
    train_y = torch.tensor([0, 1, 2, 1])
    if kind == "ordinal":
        cls = MixedOrdinalSingleTaskGPClassifier if mixed else OrdinalSingleTaskGPClassifier
    else:
        cls = MixedMulticlassSingleTaskGPClassifier if mixed else MulticlassSingleTaskGPClassifier
    kwargs = {"cat_dims": [1]} if mixed else {}
    model = cls(train_x, train_y, num_classes=3, inducing_points=4, **kwargs)
    model.eval()
    model.likelihood.eval()
    return model


@pytest.mark.parametrize("kind", ["ordinal", "multiclass"])
@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("q", [1, 2])
def test_probability_posterior_preserves_botorch_contract(
    kind: str, mixed: bool, q: int
) -> None:
    model = _model(kind, mixed)
    X = torch.tensor([[[0.2], [0.8]]], dtype=torch.double)[:, :q]
    if mixed:
        X = torch.cat(
            (X, torch.tensor([[[0.0], [1.0]]], dtype=torch.double)[:, :q]),
            dim=-1,
        )
    with torch.no_grad():
        kwargs = {"num_samples": 256} if kind == "multiclass" else {}
        posterior = model.probability_posterior(X, **kwargs)
        assert isinstance(posterior, Posterior)
        assert posterior.mean.shape == (1, q, 3)
        assert posterior.variance.shape == (1, q, 3)
        assert posterior.dtype == X.dtype
        assert posterior.device == X.device
        assert torch.isfinite(posterior.mean).all()
        assert torch.isfinite(posterior.variance).all()
        assert (posterior.variance >= 0).all()
        torch.testing.assert_close(
            posterior.mean.sum(dim=-1),
            torch.ones(1, q, dtype=X.dtype),
            atol=1e-8,
            rtol=1e-8,
        )
        samples = posterior.rsample(torch.Size([4, 2]))
        assert samples.shape == (4, 2, 1, q, 3)
        torch.testing.assert_close(samples.sum(dim=-1), torch.ones_like(samples[..., 0]))
        latent = model.latent_posterior(X)
        torch.testing.assert_close(posterior.base_sample_shape, latent.base_sample_shape)
        base = torch.randn((3, *posterior.base_sample_shape), dtype=X.dtype)
        first = posterior.rsample_from_base_samples(torch.Size([3]), base)
        second = posterior.rsample_from_base_samples(torch.Size([3]), base)
        assert first.shape == (3, 1, q, 3)
        torch.testing.assert_close(first, second, atol=0, rtol=0)
        lower, upper = posterior.credible_interval(0.9, num_samples=256)
        assert lower.shape == posterior.mean.shape
        assert upper.shape == posterior.mean.shape
        assert torch.isfinite(lower).all()
        assert torch.isfinite(upper).all()
        assert (lower <= upper).all()


@torch.no_grad()
def test_ordinal_probability_posterior_matches_dense_gaussian_integration() -> None:
    model = _model("ordinal", mixed=False)
    X = torch.tensor([[[0.2], [0.5], [0.8]]], dtype=torch.double)
    posterior = model.probability_posterior(X)
    latent = model.latent_posterior(X)
    mu = latent.mean.squeeze(-1)
    sigma = latent.variance.squeeze(-1).clamp_min(0).sqrt()
    z = torch.linspace(-10.0, 10.0, 16001, dtype=torch.double)
    values = mu.unsqueeze(0) + z[:, None, None] * sigma.unsqueeze(0)
    probabilities = model.likelihood.class_probabilities(values)
    density = (-(z.square()) / 2.0).exp() / math.sqrt(2.0 * math.pi)
    weights = density[:, None, None, None]
    expected_mean = torch.trapezoid(probabilities * weights, z, dim=0)
    expected_variance = torch.trapezoid(
        (probabilities - expected_mean).square() * weights, z, dim=0
    )
    torch.testing.assert_close(posterior.mean, model.predict_proba(X), atol=0, rtol=0)
    torch.testing.assert_close(posterior.mean, expected_mean, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        posterior.variance, expected_variance, atol=1e-5, rtol=1e-4
    )
    assert posterior.variance.max() < 0.25


@torch.no_grad()
def test_multiclass_probability_moments_are_seeded_and_mc_consistent() -> None:
    model = _model("multiclass", mixed=False)
    X = torch.tensor([[[0.2], [0.7]]], dtype=torch.double)
    first = model.probability_posterior(X, num_samples=1024, seed=7)
    second = model.probability_posterior(X, num_samples=1024, seed=7)
    torch.testing.assert_close(first.mean, second.mean, atol=0, rtol=0)
    torch.testing.assert_close(first.variance, second.variance, atol=0, rtol=0)
    torch.manual_seed(123)
    reference = model.sample_class_probabilities(X, torch.Size([8192]))
    torch.testing.assert_close(
        first.mean, reference.mean(dim=0), atol=0.025, rtol=0.04
    )
    torch.testing.assert_close(
        first.variance, reference.var(dim=0, unbiased=False), atol=0.025, rtol=0.08
    )


@pytest.mark.parametrize("kind", ["ordinal", "multiclass"])
def test_probability_posterior_moments_preserve_candidate_gradients(kind: str) -> None:
    model = _model(kind, mixed=False)
    X = torch.tensor([[[0.25], [0.75]]], dtype=torch.double, requires_grad=True)
    kwargs = {"num_samples": 128} if kind == "multiclass" else {}
    posterior = model.probability_posterior(X, **kwargs)
    (posterior.mean[..., 1] + posterior.variance[..., 1]).sum().backward()
    assert X.grad is not None
    assert torch.isfinite(X.grad).all()
