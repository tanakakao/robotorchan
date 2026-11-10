"""Real two-step sequential RGPE Bayesian optimization smoke test."""

import torch
from botorch.models import SingleTaskGP

from robotorchan.optim.transfer.rgpe_sequential import propose_rgpe_candidate


def test_rgpe_sequential_two_iterations():
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    x = torch.tensor([[0.0], [0.4], [1.0]], dtype=torch.double)
    objective = lambda values: -((values - 0.7) ** 2)
    y = objective(x)
    source_model = SingleTaskGP(x, y)

    for iteration in range(2):
        candidate, weights = propose_rgpe_candidate(
            [source_model],
            x,
            y,
            bounds,
            n_bootstrap=8,
            raw_samples=16,
            num_restarts=2,
            seed=iteration,
        )
        assert candidate.shape == (1, 1)
        assert torch.isfinite(candidate).all()
        assert bool(((candidate >= bounds[0]) & (candidate <= bounds[1])).all())
        assert weights.shape == (2,)
        assert torch.allclose(weights.sum(), weights.new_tensor(1.0))
        x = torch.cat([x, candidate], dim=0)
        y = torch.cat([y, objective(candidate)], dim=0)

    assert x.shape == (5, 1)
    assert y.shape == (5, 1)
