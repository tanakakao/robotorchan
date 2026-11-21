"""Phase 3 independent seeded qRandomizedStraddle numerical reference."""

import torch

from robotorchan.acquisition import qRandomizedStraddle
from robotorchan.models import SingleTaskGP


def test_randomized_straddle_matches_seeded_inverse_cdf_batch_reference() -> None:
    train_x = torch.linspace(0.0, 1.0, 8, dtype=torch.double).unsqueeze(-1)
    model = SingleTaskGP(train_x, torch.sin(5.0 * train_x))
    model.eval()
    model.likelihood.eval()

    X = torch.tensor(
        [[[0.15], [0.45], [0.8]], [[0.2], [0.5], [0.9]]],
        dtype=torch.double,
    )
    target = 0.3
    seed = 42

    reference_rng = torch.Generator().manual_seed(seed)
    uniform = torch.rand((), dtype=torch.double, generator=reference_rng)
    beta = -2.0 * torch.log1p(-uniform)
    posterior = model.posterior(X)
    mean = posterior.mean.squeeze(-1)
    std = posterior.variance.clamp_min(0.0).sqrt().squeeze(-1)
    expected = (beta.sqrt() * std - (mean - target).abs()).mean(dim=-1)

    acquisition = qRandomizedStraddle(
        model, target=target, generator=torch.Generator().manual_seed(seed)
    )
    actual = acquisition(X)

    assert actual.shape == (2,)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(acquisition(X), expected, atol=1e-12, rtol=1e-12)

    acquisition.resample()
    next_uniform = torch.rand((), dtype=torch.double, generator=reference_rng)
    next_beta = -2.0 * torch.log1p(-next_uniform)
    next_expected = (next_beta.sqrt() * std - (mean - target).abs()).mean(dim=-1)
    torch.testing.assert_close(acquisition(X), next_expected, atol=1e-12, rtol=1e-12)
