"""Focused binary COMBO graph-diffusion surrogate tests."""

import pytest
import torch
from botorch.acquisition.analytic import PosteriorMean

from robotorchan.models.standard import BinaryGraphDiffusionKernel, COMBOSurrogate


def test_binary_diffusion_kernel_is_symmetric_and_positive_definite():
    X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=torch.double)
    kernel = BinaryGraphDiffusionKernel(2).double()
    covariance = kernel(X).to_dense()
    assert torch.allclose(covariance, covariance.T)
    assert torch.allclose(covariance.diag(), torch.ones(4, dtype=torch.double))
    assert torch.all(torch.linalg.eigvalsh(covariance) > 0)


def test_combo_surrogate_posterior_and_botorch_acquisition():
    X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=torch.double)
    Y = X.sum(dim=-1, keepdim=True)
    model = COMBOSurrogate(X, Y)
    model.eval()
    posterior = model.posterior(X.unsqueeze(0))
    assert posterior.mean.shape == (1, 4, 1)
    assert posterior.variance.shape == (1, 4, 1)
    values = PosteriorMean(model)(X.unsqueeze(-2))
    assert values.shape == (4,)
    assert torch.isfinite(values).all()


def test_combo_rejects_nonbinary_points():
    X = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    Y = X.clone()
    with pytest.raises(ValueError, match="binary"):
        COMBOSurrogate(torch.tensor([[0.5]], dtype=torch.double), Y[:1])
    model = COMBOSurrogate(X, Y)
    with pytest.raises(ValueError, match="binary"):
        model.posterior(torch.tensor([[[0.5]]], dtype=torch.double))
