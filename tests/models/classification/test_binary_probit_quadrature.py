"""Independent one-dimensional integration references for binary GP uncertainty."""

import math

import pytest
import torch
from torch.distributions import Normal

from robotorchan.acquisition import BALD, ProbabilityVariance
from robotorchan.models.classification import BinarySingleTaskGPClassifier
from robotorchan.models.classification.binary.robust.input_dependent_noise import (
    InputDependentLabelNoiseBinarySingleTaskGPClassifier,
)
from robotorchan.models.classification.binary.robust.label_noise import (
    LabelNoiseBinarySingleTaskGPClassifier,
)


@pytest.mark.parametrize("noise_kind", ["none", "fixed", "input_dependent"])
def test_binary_probability_variance_and_bald_match_dense_integration(noise_kind: str) -> None:
    """Check likelihood moments against a separately discretized Gaussian integral."""
    train_x = torch.tensor([[0.0], [0.3], [0.7], [1.0]], dtype=torch.float64)
    train_y = torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float64)
    if noise_kind == "fixed":
        model = LabelNoiseBinarySingleTaskGPClassifier(
            train_x,
            train_y,
            false_positive_rate=0.1,
            false_negative_rate=0.2,
            inducing_points=4,
        )
    elif noise_kind == "input_dependent":

        def noise_rates(X: torch.Tensor) -> torch.Tensor:
            return torch.stack(
                (torch.full_like(X[..., 0], 0.1), torch.full_like(X[..., 0], 0.2)),
                dim=-1,
            )

        model = InputDependentLabelNoiseBinarySingleTaskGPClassifier(
            train_x,
            train_y,
            noise_rate_model=noise_rates,
            inducing_points=4,
        )
    else:
        model = BinarySingleTaskGPClassifier(train_x, train_y, inducing_points=4)
    model.eval()
    model.likelihood.eval()
    candidates = torch.tensor([[[0.2]], [[0.5]], [[0.8]]], dtype=torch.float64)

    with torch.no_grad():
        latent = model.latent_posterior(candidates)
        mu = latent.mean.squeeze(-1)
        sigma = latent.variance.squeeze(-1).clamp_min(0.0).sqrt()

        grid = torch.linspace(-11.0, 11.0, 40001, dtype=torch.float64)
        density = torch.exp(-grid.square() / 2.0) / math.sqrt(2.0 * math.pi)
        values = mu.unsqueeze(0) + grid[:, None, None] * sigma.unsqueeze(0)
        positive = Normal(0.0, 1.0).cdf(values)
        if noise_kind != "none":
            positive = 0.1 + 0.7 * positive

        weights = density[:, None, None]
        mean_reference = torch.trapezoid(positive * weights, grid, dim=0).squeeze(-1)
        second_reference = torch.trapezoid(positive.square() * weights, grid, dim=0).squeeze(-1)
        variance_reference = (second_reference - mean_reference.square()).clamp_min(0.0)
        conditional_entropy = -torch.special.xlogy(positive, positive) - torch.special.xlogy(
            1.0 - positive, 1.0 - positive
        )
        expected_entropy = torch.trapezoid(conditional_entropy * weights, grid, dim=0).squeeze(-1)
        predictive_entropy = -torch.special.xlogy(
            mean_reference, mean_reference
        ) - torch.special.xlogy(1.0 - mean_reference, 1.0 - mean_reference)
        bald_reference = (predictive_entropy - expected_entropy).clamp_min(0.0)

        probabilities = model.predict_proba(candidates)
        variance = model.probability_variance(candidates)
        bald = model.mutual_information(candidates)
        acquired_variance = ProbabilityVariance(model)(candidates)
        acquired_bald = BALD(model)(candidates)

    assert probabilities.shape == (3, 2)
    assert variance.shape == (3, 2)
    assert bald.shape == (3,)
    torch.testing.assert_close(probabilities[..., 1], mean_reference, atol=1e-8, rtol=1e-8)
    torch.testing.assert_close(variance[..., 1], variance_reference, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(variance[..., 0], variance[..., 1])
    torch.testing.assert_close(bald, bald_reference, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(acquired_variance, variance_reference, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(acquired_bald, bald_reference, atol=1e-5, rtol=1e-5)

    with torch.no_grad():
        joint_candidates = candidates[:2].transpose(0, 1)
        joint_variance = model.probability_variance(joint_candidates)
        joint_bald_marginal = model.mutual_information(joint_candidates)
    assert joint_variance.shape == (1, 2, 2)
    assert joint_bald_marginal.shape == (1, 2)


def test_binary_probability_uncertainty_is_deterministic_and_differentiable() -> None:
    """Quadrature is reproducible and permits optimizing the acquisition in X."""
    train_x = torch.tensor([[0.0], [0.4], [0.7], [1.0]], dtype=torch.float64)
    train_y = torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float64)
    model = BinarySingleTaskGPClassifier(train_x, train_y, inducing_points=4)
    model.eval()
    model.likelihood.eval()
    candidates = torch.tensor([[[0.2]], [[0.8]]], dtype=torch.float64, requires_grad=True)
    variance = ProbabilityVariance(model)(candidates)
    bald = BALD(model)(candidates)
    torch.manual_seed(987)
    torch.testing.assert_close(variance, ProbabilityVariance(model)(candidates), atol=0, rtol=0)
    torch.testing.assert_close(bald, BALD(model)(candidates), atol=0, rtol=0)
    (variance + bald).sum().backward()
    assert candidates.grad is not None
    assert torch.isfinite(candidates.grad).all()
    assert torch.isfinite(variance).all()
    assert torch.isfinite(bald).all()


def test_binary_mc_fallback_remains_available() -> None:
    """Retain an explicit Monte Carlo path for debugging and comparisons."""
    train_x = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.float64)
    train_y = torch.tensor([0.0, 1.0, 1.0], dtype=torch.float64)
    model = BinarySingleTaskGPClassifier(train_x, train_y, inducing_points=3)
    model.eval()
    candidates = torch.tensor([[[0.3]], [[0.7]]], dtype=torch.float64)
    with torch.no_grad():
        variance = model.probability_variance(candidates, num_samples=64, method="mc")
        bald = model.mutual_information(candidates, num_samples=64, method="mc")
    assert variance.shape == (2, 1, 2)
    assert bald.shape == (2, 1)
    assert torch.isfinite(variance).all()
    assert torch.isfinite(bald).all()
