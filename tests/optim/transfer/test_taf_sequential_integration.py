"""Real two-iteration sequential TAF optimization smoke test."""

import torch
from botorch.models import SingleTaskGP

from robotorchan.optim.transfer.taf_sequential import propose_taf_candidate


def test_taf_sequential_two_iterations():
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    x = torch.tensor([[0.0], [0.4], [1.0]], dtype=torch.double)

    def objective(values):
        return -((values - 0.7) ** 2)

    y = objective(x)
    source = SingleTaskGP(x, objective(x - 0.1))
    weights = torch.tensor([0.5, 0.5], dtype=torch.double)

    for iteration in range(2):
        candidate = propose_taf_candidate(
            [source],
            x,
            y,
            bounds,
            weights,
            raw_samples=16,
            num_restarts=2,
            seed=iteration,
        )
        assert candidate.shape == (1, 1)
        assert torch.isfinite(candidate).all()
        assert bool(((candidate >= bounds[0]) & (candidate <= bounds[1])).all())
        x = torch.cat([x, candidate], dim=0)
        y = torch.cat([y, objective(candidate)], dim=0)

    assert x.shape == (5, 1)
    assert y.shape == (5, 1)
