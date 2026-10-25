"""Independent uncertainty-component reference for heteroskedastic GP models."""

import torch

from robotorchan.models import HeteroskedasticSingleTaskGP


class _FixedLogNoisePosterior:
    def __init__(self, mean: torch.Tensor) -> None:
        self.mean = mean


class _FixedLogNoiseModel:
    """Deterministic test double for a fitted log-observation-variance GP."""

    def posterior(self, X: torch.Tensor) -> _FixedLogNoisePosterior:
        return _FixedLogNoisePosterior(
            torch.log(
                torch.where(
                    X[..., :1] < 0.5,
                    torch.full_like(X[..., :1], 0.01),
                    torch.full_like(X[..., :1], 0.09),
                )
            )
        )


def test_heteroskedastic_noise_is_separate_from_latent_posterior_variance() -> None:
    """Compare explicit log-noise mapping with the response GP's latent variance."""
    train_x = torch.linspace(0.0, 1.0, 8, dtype=torch.float64).unsqueeze(-1)
    train_y = torch.sin(train_x * 3.0)
    model = HeteroskedasticSingleTaskGP(train_x, train_y)
    model.eval()
    model.likelihood.eval()
    model.noise_model = _FixedLogNoiseModel()
    model._noise_model_fitted = True
    candidates = torch.tensor([[0.2], [0.8]], dtype=torch.float64)

    with torch.no_grad():
        latent = model.posterior(candidates, observation_noise=False)
        noise = model.predicted_noise(candidates)

    expected_noise = torch.tensor([[0.01], [0.09]], dtype=torch.float64)
    torch.testing.assert_close(noise, expected_noise, atol=1e-12, rtol=1e-12)
    assert latent.variance.shape == noise.shape
    assert torch.isfinite(latent.variance).all()
    assert (latent.variance >= 0).all()
    assert noise[1, 0] > noise[0, 0]
    # The model exposes observation noise separately: latent posterior
    # variance is not silently replaced by the input-dependent noise values.
    assert not torch.allclose(latent.variance, noise)


def test_heteroskedastic_manual_observation_variance_identity() -> None:
    """Check the conditional variance identity without claiming a public noise posterior."""
    train_x = torch.linspace(0.0, 1.0, 8, dtype=torch.float64).unsqueeze(-1)
    train_y = torch.sin(train_x * 3.0)
    model = HeteroskedasticSingleTaskGP(train_x, train_y)
    model.eval()
    model.likelihood.eval()
    model.noise_model = _FixedLogNoiseModel()
    model._noise_model_fitted = True
    candidates = torch.tensor([[0.2], [0.8]], dtype=torch.float64)

    with torch.no_grad():
        latent = model.posterior(candidates, observation_noise=False)
        predicted_noise = model.predicted_noise(candidates)

    # For independent zero-mean observation error with fixed conditional
    # variance v(x), Var[y(x) | data] = Var[f(x) | data] + v(x).
    manual_observation_variance = latent.variance + predicted_noise
    expected = latent.variance + torch.tensor([[0.01], [0.09]], dtype=torch.float64)
    torch.testing.assert_close(manual_observation_variance, expected, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(
        manual_observation_variance - latent.variance,
        predicted_noise,
        atol=1e-12,
        rtol=1e-12,
    )
    assert torch.all(manual_observation_variance >= latent.variance)
