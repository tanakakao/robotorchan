"""LatentKronecker posterior moments and pathwise samples share outcome units."""

import pytest
import torch
from botorch.models.latent_kronecker_gp import (
    LatentKroneckerGP as BoTorchLatentKroneckerGP,
)
from botorch.posteriors.transformed import TransformedPosterior

from robotorchan.models import LatentKroneckerGP, MixedLatentKroneckerGP


@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("standardize", [False, True])
def test_latent_kronecker_posterior_moments_and_samples(mixed, standardize):
    dtype = torch.double
    X = torch.linspace(0.1, 0.9, 5, dtype=dtype).unsqueeze(-1)
    kwargs = {}
    if mixed:
        X = torch.cat([X, (torch.arange(5) % 2).to(dtype).unsqueeze(-1)], dim=-1)
        kwargs["cat_dims"] = [1]
    T = torch.linspace(0, 1, 3, dtype=dtype).unsqueeze(-1)
    Y = 6 + 2 * torch.sin(X[:, :1] * 3 + T.squeeze(-1))
    if not standardize:
        kwargs["outcome_transform"] = None
    model_cls = MixedLatentKroneckerGP if mixed else LatentKroneckerGP
    model = model_cls(train_X=X, train_T=T, train_Y=Y, **kwargs)
    model.eval()

    test_X = X[1:3]
    with torch.no_grad():
        with model.use_iterative_methods():
            raw = BoTorchLatentKroneckerGP.posterior(model, test_X, T)
        posterior = model.posterior(test_X, T)
        if standardize:
            expected_mean, expected_var = model.outcome_transform.untransform(
                raw.mean.unsqueeze(-1), raw.variance.unsqueeze(-1)
            )
            expected_mean = expected_mean.squeeze(-1)
            expected_var = expected_var.squeeze(-1)
            assert isinstance(posterior, TransformedPosterior)
        else:
            expected_mean, expected_var = raw.mean, raw.variance
            assert not isinstance(posterior, TransformedPosterior)

        torch.testing.assert_close(posterior.mean, expected_mean)
        torch.testing.assert_close(posterior.variance, expected_var)
        assert posterior.mean.numel() == test_X.shape[-2] * T.shape[-2]
        assert torch.isfinite(posterior.mean).all()
        assert torch.isfinite(posterior.variance).all()

        # Identical base samples must produce identical physical-scale pathwise
        # samples, including in the Standardize case (no double transform).
        sample_shape = torch.Size([2])
        base_samples = torch.randn(sample_shape + raw.base_sample_shape, dtype=dtype)
        with model.use_iterative_methods():
            raw_samples = raw.rsample_from_base_samples(sample_shape, base_samples)
        wrapped_samples = posterior.rsample_from_base_samples(sample_shape, base_samples)
        torch.testing.assert_close(wrapped_samples, raw_samples)
        assert wrapped_samples.shape[-2:] == (test_X.shape[-2], T.shape[-2])
