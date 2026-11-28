"""Independent Gaussian likelihood reference for joint heteroskedastic GP."""

import math

import pytest
import torch

from robotorchan.models import (
    JointHeteroskedasticSingleTaskGP,
    MixedJointHeteroskedasticSingleTaskGP,
)


@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("noise_floor", [1e-6, 10.0])
def test_joint_training_loss_matches_clamped_gaussian_reference(
    mixed: bool, noise_floor: float
) -> None:
    torch.manual_seed(11)
    x = torch.linspace(0, 1, 8, dtype=torch.double).unsqueeze(-1)
    y = torch.sin(x)
    kwargs = {"num_inducing": 4, "noise_floor": noise_floor, "num_mc_samples": 3}
    if mixed:
        x = torch.cat((x, (torch.arange(8) % 2).to(x).unsqueeze(-1)), dim=-1)
        model = MixedJointHeteroskedasticSingleTaskGP(x, y, cat_dims=[1], **kwargs)
    else:
        model = JointHeteroskedasticSingleTaskGP(x, y, **kwargs)

    # Initialize lazy variational parameters before comparing seeded draws.
    # The first forward pass can consume RNG and invalidate seed replay.
    model.response_model.model(x)
    model.noise_model.model(x)

    # Independent scalar Gaussian expectation, conditional on each sampled log variance.
    response = model.response_model.model(x)
    noise = model.noise_model.model(x)
    torch.manual_seed(123)
    standard_draws = torch.randn(3, x.shape[0], dtype=x.dtype, device=x.device)
    log_noise = noise.mean + standard_draws * noise.variance.sqrt()
    if noise_floor == 10.0:
        assert torch.any(log_noise.exp() < noise_floor)
    terms = []
    for i in range(x.shape[0]):
        variance = log_noise[:, i].exp().clamp_min(noise_floor)
        second_moment = response.variance[i] + response.mean[i].square()
        expected_error = y[i, 0].square() - 2 * y[i, 0] * response.mean[i] + second_moment
        terms.append(
            0.5 * (math.log(2 * math.pi) + variance.log() + expected_error / variance)
        )
    gaussian_nll = torch.stack(terms, dim=-1)
    kl_response = model.response_model.model.variational_strategy.kl_divergence().sum()
    kl_noise = model.noise_model.model.variational_strategy.kl_divergence().sum()
    expected = (gaussian_nll.mean(0).sum() + kl_response + kl_noise) / x.shape[-2]

    torch.manual_seed(123)
    actual = model.training_loss()
    torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-9)
