"""Phase 4: reuse optimizer dispatch for classification-weighted acquisitions."""

import pytest
import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from torch import nn

from robotorchan.acquisition.classification_constraints import (
    FeasibilityWeightedAcquisition,
)
from robotorchan.optim import optimize_acqf


class _Score(AcquisitionFunction):
    def __init__(self):
        super().__init__(model=None)

    def forward(self, X):
        return -(X[..., 0] - 0.65).square().sum(dim=-1) + 1.0


class _Feasibility(nn.Module):
    def forward(self, X):
        return torch.sigmoid(3 * X[..., 0])


@pytest.mark.parametrize("q", [1, 2])
def test_classifier_weighted_q_batch_uses_existing_sampling_dispatch(q):
    acquisition = FeasibilityWeightedAcquisition(_Score(), _Feasibility())
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    candidates, values = optimize_acqf(
        acquisition,
        bounds,
        q,
        optimizer="sobol",
        raw_samples=64,
        seed=13,
    )
    assert candidates.shape == (q, 1)
    assert torch.isfinite(values).all()
    assert ((candidates >= 0) & (candidates <= 1)).all()


def test_classifier_weighted_q_batch_rejects_joint_fantasization():
    acquisition = FeasibilityWeightedAcquisition(_Score(), _Feasibility())
    with pytest.raises(NotImplementedError, match="joint fantasization"):
        acquisition.fantasize(torch.tensor([[0.5]], dtype=torch.double))
