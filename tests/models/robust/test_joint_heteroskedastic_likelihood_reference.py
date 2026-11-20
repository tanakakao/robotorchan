"""Independent Gaussian likelihood reference for joint heteroskedastic GP."""

import math

import pytest
import torch

from robotorchan.models import (
    JointHeteroskedasticSingleTaskGP,
    MixedJointHeteroskedasticSingleTaskGP,
)


@pytest.mark.parametrize("mixed", [False, True])
def test_joint_training_loss_matches_clamped_gaussian_reference(mixed: bool) -> None:
    torch.manual_seed(11)
    x = torch.linspace(0, 1, 8, dtype=torch.double).unsqueeze(-1)
    y = torch.sin(x)
    kwargs = {"num_inducing": 4, "noise_floor": 10.0, "num_mc_samples": 3}
    if mixed:
        x = torch.cat((x, (torch.arange(8) % 2).to(x).unsqueeze(-1)), dim=-1)
        model = MixedJointHeteroskedasticSingleTaskGP(x, y, cat_dims=[1], **kwargs)
    else:
        model = JointHeteroskedasticSingleTaskGP(x, y, **kwargs)

    # Initialize lazy variational parameters before comparing seeded draws.
    # The first forward pass can consume RNG and invalidate seed replay.
    model.response_model.model(x)
    model.noise_model.model(x)

    # Independently reconstruct the Gaussian negative ELBO from latent draws.
    torch.manual_seed(123)
    response = model.response_model.model(x).rsample(torch.Size([3]))
    log_noise = model.noise_model.model(x).rsample(torch.Size([3]))
    variance = log_noise.exp().clamp_min(model.noise_floor)
    assert torch.any(log_noise.exp() < model.noise_floor)
    gaussian_nll = 0.5 * (
        math.log(2 * math.pi) + variance.log() + (y.squeeze(-1) - response).square() / variance
    )
    kl_response = model.response_model.model.variational_strategy.kl_divergence().sum()
    kl_noise = model.noise_model.model.variational_strategy.kl_divergence().sum()
    expected = (gaussian_nll.mean(0).sum() + kl_response + kl_noise) / x.shape[-2]

    torch.manual_seed(123)
    actual = model.training_loss()
    torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-9)
