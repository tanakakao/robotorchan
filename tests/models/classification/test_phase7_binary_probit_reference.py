"""Independent probit-integral reference for binary GP classification."""

import math

import torch

from robotorchan.models.classification import BinarySingleTaskGPClassifier


def test_binary_predictive_probability_matches_gaussian_probit_integral() -> None:
    """Integrate latent Gaussian uncertainty analytically using a probit link."""
    train_x = torch.linspace(0.0, 1.0, 8, dtype=torch.float64).unsqueeze(-1)
    train_y = (train_x[:, 0] >= 0.5).to(dtype=torch.float64)
    model = BinarySingleTaskGPClassifier(train_x, train_y)
    candidates = torch.tensor([[0.15], [0.5], [0.85]], dtype=torch.float64)

    model.eval()
    model.likelihood.eval()
    with torch.no_grad():
        latent = model.latent_posterior(candidates)
        probabilities = model.predict_proba(candidates)

    mean = latent.mean.squeeze(-1)
    variance = latent.variance.squeeze(-1)
    # If f ~ N(mu, sigma²), E[Phi(f)] = Phi(mu / sqrt(1 + sigma²)).
    standardized = mean / torch.sqrt(1.0 + variance)
    expected_positive = 0.5 * torch.erfc(-standardized / math.sqrt(2.0))

    assert probabilities.shape == (3, 2)
    torch.testing.assert_close(probabilities[:, 1], expected_positive, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(probabilities[:, 0], 1.0 - expected_positive)
    torch.testing.assert_close(probabilities.sum(dim=-1), torch.ones(3, dtype=torch.float64))
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
