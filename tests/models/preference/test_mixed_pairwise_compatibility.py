"""Minimal compatibility check for mixed covariance with PairwiseGP."""

import torch
from gpytorch.kernels import ScaleKernel

from robotorchan.models import PairwiseGP
from robotorchan.models.base import make_mixed_covar_module


def test_pairwise_gp_with_mixed_covariance() -> None:
    """Fit and predict using continuous and categorical design coordinates."""
    torch.manual_seed(0)
    datapoints = torch.tensor(
        [
            [0.0, 0.0],
            [0.3, 1.0],
            [0.7, 0.0],
            [1.0, 1.0],
        ],
        dtype=torch.double,
    )
    comparisons = torch.tensor([[1, 0], [2, 3], [1, 2]], dtype=torch.long)
    covariance = ScaleKernel(make_mixed_covar_module(input_dim=2, cat_dims=[1]))
    model = PairwiseGP(
        datapoints=datapoints,
        comparisons=comparisons,
        covar_module=covariance,
    )
    model.fit(optimizer_kwargs={"options": {"maxiter": 10}})
    model.eval()

    test_x = torch.tensor([[0.5, 0.0], [0.5, 1.0]], dtype=torch.double)
    posterior = model.posterior(test_x)
    assert posterior.mean.shape == (2, 1)
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all()
    assert (posterior.variance >= 0).all()
