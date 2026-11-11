"""Focused checks for the binary quadratic BOCS surrogate."""

import pytest
import torch
from botorch.acquisition.analytic import PosteriorMean

from robotorchan.models.standard import BOCSSurrogate, bocs_quadratic_features


def test_quadratic_features_and_posterior():
    X = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.double
    )
    Y = (1 + 2 * X[:, :1] + 3 * X[:, 1:2] + 4 * X[:, :1] * X[:, 1:2])
    assert bocs_quadratic_features(X).tolist() == [
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 1.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
    ]
    model = BOCSSurrogate(X, Y, prior_precision=0.01, noise_variance=1e-4)
    posterior = model.posterior(X.unsqueeze(0))
    assert posterior.mean.shape == (1, 4, 1)
    assert posterior.variance.shape == (1, 4, 1)
    assert torch.allclose(posterior.mean.squeeze(0), Y, atol=0.03)
    assert torch.all(posterior.variance > 0)
    values = PosteriorMean(model)(X.unsqueeze(-2))
    assert values.shape == (4,)
    assert torch.all(torch.isfinite(values))


def test_reject_nonbinary_training_and_queries():
    X = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.double)
    Y = torch.tensor([[1.0], [2.0]], dtype=torch.double)
    with pytest.raises(ValueError, match="binary"):
        BOCSSurrogate(torch.tensor([[0.5, 1.0]], dtype=torch.double), Y[:1])
    model = BOCSSurrogate(X, Y)
    with pytest.raises(ValueError, match="binary"):
        model.posterior(torch.tensor([[[0.5, 1.0]]], dtype=torch.double))
