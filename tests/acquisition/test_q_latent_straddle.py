"""qLatentStraddle marginal batch semantics."""

from unittest.mock import patch

import pytest
import torch

from robotorchan.acquisition import LatentStraddle, qLatentStraddle
from robotorchan.models.classification import BinarySingleTaskGPClassifier


@pytest.mark.parametrize("q", [1, 2, 4])
def test_q_latent_straddle_matches_mean_pointwise_score(q: int) -> None:
    train_x = torch.tensor([[0.0], [0.3], [0.7], [1.0]], dtype=torch.double)
    train_y = torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.double)
    model = BinarySingleTaskGPClassifier(train_x, train_y, inducing_points=4)
    X = torch.linspace(0.1, 0.9, q, dtype=torch.double).reshape(1, q, 1)
    mean = torch.linspace(-0.5, 0.5, q, dtype=torch.double).reshape(1, q, 1)
    variance = torch.full_like(mean, 0.25)
    posterior = type("Posterior", (), {"mean": mean, "variance": variance})()
    with patch.object(model, "latent_posterior", return_value=posterior):
        actual = qLatentStraddle(model, beta=2.0)(X)
        if q == 1:
            expected = LatentStraddle(model, beta=2.0)(X)
        else:
            expected = (2.0 * variance.sqrt() - mean.abs()).mean(dim=-2).squeeze(-1)
    assert actual.shape == torch.Size([1])
    torch.testing.assert_close(actual, expected)
