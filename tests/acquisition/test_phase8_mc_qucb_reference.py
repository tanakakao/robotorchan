"""Phase 8 independent qUCB Monte Carlo aggregation reference."""

import math

import torch
from botorch.acquisition.monte_carlo import qUpperConfidenceBound
from botorch.sampling.normal import SobolQMCNormalSampler

from robotorchan.models import SingleTaskGP


def test_q_ucb_matches_samplewise_upper_confidence_reference() -> None:
    """Compare native qUCB with samplewise absolute-deviation aggregation."""
    train_x = torch.linspace(0.0, 1.0, 8, dtype=torch.float64).unsqueeze(-1)
    train_y = torch.sin(5.0 * train_x)
    model = SingleTaskGP(train_x, train_y)
    model.eval()
    candidates = torch.tensor(
        [[[0.15], [0.45]], [[0.55], [0.85]]],
        dtype=torch.float64,
    )
    beta = 0.3
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([64]), seed=23)
    acquisition = qUpperConfidenceBound(model=model, beta=beta, sampler=sampler)

    with torch.no_grad():
        actual = acquisition(candidates)
        posterior = model.posterior(candidates)
        samples = sampler(posterior).squeeze(-1)
        mean = posterior.mean.squeeze(-1)
        adjusted = mean + math.sqrt(beta * math.pi / 2.0) * (samples - mean).abs()
        expected = adjusted.max(dim=-1).values.mean(dim=0)

    assert samples.shape == (64, 2, 2)
    assert actual.shape == (2,)
    torch.testing.assert_close(actual, expected, atol=1e-8, rtol=1e-8)
