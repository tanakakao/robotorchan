"""Fit contract for the pairwise Laplace preference model."""

from unittest.mock import patch

import torch
from botorch.models.pairwise_gp import PairwiseLaplaceMarginalLogLikelihood

from robotorchan.models import PairwiseGP


def test_pairwise_fit_uses_laplace_mll() -> None:
    datapoints = torch.linspace(0.0, 1.0, 5).unsqueeze(-1)
    comparisons = torch.tensor([[1, 0], [2, 1], [3, 2], [4, 3]])
    model = PairwiseGP(datapoints, comparisons)

    assert model.supports_fit is True
    assert isinstance(model.make_mll(), PairwiseLaplaceMarginalLogLikelihood)

    with patch("robotorchan.models.preference.pairwise.fit_gpytorch_mll") as fit:
        assert model.fit(optimizer_kwargs={"options": {"maxiter": 5}}) is None

    fit.assert_called_once()
    args, kwargs = fit.call_args
    assert isinstance(args[0], PairwiseLaplaceMarginalLogLikelihood)
    assert kwargs["optimizer_kwargs"] == {"options": {"maxiter": 5}}
