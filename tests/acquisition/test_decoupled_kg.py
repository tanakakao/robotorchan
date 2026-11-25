"""Numerical contracts for finite-decision-set masked KG."""

import pytest
import torch
from botorch.models import ModelListGP, SingleTaskGP
from botorch.sampling.list_sampler import ListSampler
from botorch.sampling.normal import SobolQMCNormalSampler

from robotorchan.acquisition import create_acquisition


@pytest.mark.parametrize("mask", [[True, False], [False, True], [True, True]])
def test_decoupled_kg(mask):
    X = torch.linspace(0, 1, 6, dtype=torch.double).unsqueeze(-1)
    model = ModelListGP(SingleTaskGP(X, X.sin()), SingleTaskGP(X, X.cos()))
    model.eval()
    acq = create_acquisition(
        "qDecoupledKnowledgeGradient",
        model=model,
        decision_X=X,
        weights=X.new_tensor([1.0, 0.0]),
        sampler=ListSampler(
            SobolQMCNormalSampler(torch.Size([8]), seed=42),
            SobolQMCNormalSampler(torch.Size([8]), seed=43),
        ),
        X_evaluation_mask=torch.tensor([mask]),
    )
    candidate = X.new_tensor([[[0.35]], [[0.65]]]).requires_grad_()
    value = acq(candidate)
    assert value.shape == (2,)
    assert torch.isfinite(value).all()
    if not mask[0]:
        torch.testing.assert_close(value, torch.zeros_like(value), atol=1e-8, rtol=0)
    else:
        (gradient,) = torch.autograd.grad(value.sum(), candidate)
        assert torch.isfinite(gradient).all()
