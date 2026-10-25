"""Phase 8 independent Gaussian references for BoTorch-native analytic BO."""

import math

import torch
from botorch.acquisition.analytic import (
    LogExpectedImprovement,
    LogProbabilityOfImprovement,
    UpperConfidenceBound,
)

from robotorchan.models import SingleTaskGP


def test_native_analytic_bo_matches_gaussian_closed_forms() -> None:
    """Verify log-EI, log-PI and UCB values against normal CDF/PDF identities."""
    train_x = torch.linspace(0.0, 1.0, 8, dtype=torch.float64).unsqueeze(-1)
    train_y = torch.sin(5.0 * train_x)
    model = SingleTaskGP(train_x, train_y)
    model.eval()
    candidates = torch.tensor([[[0.12]], [[0.25]], [[0.42]]], dtype=torch.float64)
    best_f = 0.0
    beta = 0.4

    with torch.no_grad():
        posterior = model.posterior(candidates)
        mean = posterior.mean.squeeze(-1).squeeze(-1)
        std = posterior.variance.squeeze(-1).squeeze(-1).sqrt()
        z = (mean - best_f) / std
        normal = torch.distributions.Normal(
            torch.tensor(0.0, dtype=torch.float64),
            torch.tensor(1.0, dtype=torch.float64),
        )
        expected_ei = (mean - best_f) * normal.cdf(z) + std * torch.exp(normal.log_prob(z))
        expected_log_ei = expected_ei.log()
        expected_log_pi = normal.cdf(z).log()
        expected_ucb = mean + math.sqrt(beta) * std

        actual_log_ei = LogExpectedImprovement(model, best_f=best_f)(candidates)
        actual_log_pi = LogProbabilityOfImprovement(model, best_f=best_f)(candidates)
        actual_ucb = UpperConfidenceBound(model, beta=beta)(candidates)

    for actual, expected in (
        (actual_log_ei, expected_log_ei),
        (actual_log_pi, expected_log_pi),
        (actual_ucb, expected_ucb),
    ):
        assert actual.shape == (3,)
        torch.testing.assert_close(actual, expected, atol=1e-8, rtol=1e-8)
