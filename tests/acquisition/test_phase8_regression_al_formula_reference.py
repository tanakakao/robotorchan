"""Phase 8 fixed-parameter regression active-learning formula references."""

import torch

from robotorchan.acquisition import BoundaryVariance, PosteriorStd, PosteriorVariance, Straddle
from robotorchan.models import SingleTaskGP


def test_regression_al_scores_match_independent_moment_formulas() -> None:
    """Compare four acquisition values with explicit posterior-moment formulas."""
    train_x = torch.linspace(0.0, 1.0, 8, dtype=torch.float64).unsqueeze(-1)
    train_y = torch.sin(4.0 * train_x)
    model = SingleTaskGP(train_x, train_y)
    model.eval()
    model.likelihood.eval()
    candidates = torch.tensor([[[0.15]], [[0.45]], [[0.85]]], dtype=torch.float64)
    target = 0.35
    beta = 1.7

    with torch.no_grad():
        posterior = model.posterior(candidates)
        mean = posterior.mean[..., 0, 0]
        variance = posterior.variance[..., 0, 0]
        std = variance.clamp_min(0.0).sqrt()
        expected_straddle = beta * std - (mean - target).abs()
        scale = std.clamp_min(torch.finfo(variance.dtype).eps)
        expected_boundary = variance.clamp_min(0.0) * torch.exp(
            -0.5 * ((mean - target) / scale).square()
        )
        actual_variance = PosteriorVariance(model)(candidates)
        actual_std = PosteriorStd(model)(candidates)
        actual_straddle = Straddle(model, target=target, beta=beta)(candidates)
        actual_boundary = BoundaryVariance(model, target=target)(candidates)

    for actual, expected in (
        (actual_variance, variance),
        (actual_std, std),
        (actual_straddle, expected_straddle),
        (actual_boundary, expected_boundary),
    ):
        assert actual.shape == (3,)
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
