"""Contracts for the optional NGBoost probabilistic surrogate."""

import importlib.util

import pytest
import torch

from robotorchan.models.non_gp.distribution_posterior import GaussianDistributionPosterior

NGBOOST_AVAILABLE = importlib.util.find_spec("ngboost") is not None


def test_gaussian_distribution_posterior_shapes() -> None:
    mean = torch.zeros(3, 1, dtype=torch.double)
    variance = torch.full_like(mean, 0.25)
    posterior = GaussianDistributionPosterior(mean, variance)
    assert posterior.mean.shape == torch.Size([3, 1])
    assert posterior.variance.shape == torch.Size([3, 1])
    samples = posterior.rsample(torch.Size([5]))
    assert samples.shape == torch.Size([5, 3, 1])
    assert torch.isfinite(samples).all()


@pytest.mark.skipif(not NGBOOST_AVAILABLE, reason="optional ngboost dependency is not installed")
def test_ngboost_surrogate_fit_posterior_and_mc_acquisition() -> None:
    from botorch.acquisition.monte_carlo import qUpperConfidenceBound
    from botorch.sampling.normal import IIDNormalSampler

    from robotorchan.models import NGBoostSurrogate

    train_x = torch.linspace(0, 1, 12, dtype=torch.double).unsqueeze(-1)
    train_y = torch.sin(train_x * 4.0)
    model = NGBoostSurrogate(
        train_x,
        train_y,
        random_state=0,
        n_estimators=20,
        verbose=False,
    )
    assert torch.equal(model.raw_train_X, train_x)
    assert torch.equal(model.raw_train_Y, train_y)
    assert model.supports_mll is False
    model.fit()
    posterior = model.posterior(train_x[:3])
    assert posterior.mean.shape == torch.Size([3, 1])
    assert posterior.variance.shape == torch.Size([3, 1])
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all()
    assert torch.all(posterior.variance >= 0)
    samples = posterior.rsample(torch.Size([4]))
    assert samples.shape == torch.Size([4, 3, 1])
    acquisition = qUpperConfidenceBound(
        model=model,
        beta=0.2,
        sampler=IIDNormalSampler(sample_shape=torch.Size([8])),
    )
    value = acquisition(train_x[:1].unsqueeze(0))
    assert value.shape == torch.Size([1])
    assert torch.isfinite(value).all()


@pytest.mark.skipif(not NGBOOST_AVAILABLE, reason="optional ngboost dependency is not installed")
def test_ngboost_supports_posterior_variance_active_learning() -> None:
    from robotorchan.acquisition import PosteriorStd, PosteriorVariance
    from robotorchan.models import NGBoostSurrogate

    train_x = torch.linspace(0, 1, 12, dtype=torch.double).unsqueeze(-1)
    train_y = torch.sin(train_x * 4.0)
    model = NGBoostSurrogate(
        train_x,
        train_y,
        random_state=0,
        n_estimators=20,
        verbose=False,
    )
    model.fit()
    candidates = torch.tensor([[[0.15]], [[0.55]], [[0.9]]], dtype=torch.double)
    variance = PosteriorVariance(model)(candidates)
    std = PosteriorStd(model)(candidates)
    assert variance.shape == torch.Size([3])
    assert torch.isfinite(variance).all()
    assert torch.all(variance >= 0)
    assert torch.allclose(std.square(), variance, rtol=1e-6, atol=1e-8)


@pytest.mark.skipif(not NGBOOST_AVAILABLE, reason="optional ngboost dependency is not installed")
def test_ngboost_supports_gradient_free_bo_candidate_search() -> None:
    from botorch.acquisition.logei import qLogExpectedImprovement
    from botorch.sampling.normal import IIDNormalSampler

    from robotorchan.models import NGBoostSurrogate

    train_x = torch.linspace(0, 1, 12, dtype=torch.double).unsqueeze(-1)
    train_y = torch.sin(train_x * 4.0)
    model = NGBoostSurrogate(
        train_x,
        train_y,
        random_state=0,
        n_estimators=20,
        verbose=False,
    )
    model.fit()

    acquisition = qLogExpectedImprovement(
        model=model,
        best_f=train_y.max(),
        sampler=IIDNormalSampler(sample_shape=torch.Size([16])),
    )
    candidates = torch.linspace(0, 1, 41, dtype=torch.double).view(-1, 1, 1)
    values = acquisition(candidates)

    best_index = torch.argmax(values)
    best_candidate = candidates[best_index]
    assert best_candidate.shape == torch.Size([1, 1])
    assert torch.isfinite(best_candidate).all()
    assert torch.isfinite(values).all()
