"""Phase 8 Monte Carlo simple-regret sample aggregation reference."""

import torch
from botorch.acquisition.monte_carlo import qSimpleRegret
from botorch.sampling.normal import SobolQMCNormalSampler

from robotorchan.models import SingleTaskGP


def test_q_simple_regret_matches_explicit_sample_maximum_average() -> None:
    """Check q-batch maximization and sample averaging against posterior draws."""
    train_x = torch.linspace(0.0, 1.0, 8, dtype=torch.float64).unsqueeze(-1)
    train_y = torch.sin(5.0 * train_x)
    model = SingleTaskGP(train_x, train_y)
    model.eval()
    candidates = torch.tensor(
        [[[0.15], [0.45]], [[0.55], [0.85]]],
        dtype=torch.float64,
    )
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([64]), seed=17)
    acquisition = qSimpleRegret(model=model, sampler=sampler)

    with torch.no_grad():
        actual = acquisition(candidates)
        # Reuse the acquisition sampler's established base samples to avoid
        # comparing different Monte Carlo draws.
        posterior = model.posterior(candidates)
        samples = sampler(posterior).squeeze(-1)
        expected = samples.max(dim=-1).values.mean(dim=0)

    assert samples.shape == (64, 2, 2)
    assert actual.shape == (2,)
    torch.testing.assert_close(actual, expected, atol=1e-8, rtol=1e-8)
