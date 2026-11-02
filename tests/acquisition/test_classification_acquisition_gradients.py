"""Gradient contracts for classification acquisitions used by gradient optimizers."""

import pytest
import torch

from robotorchan.acquisition.active_learning.classification import (
    BALD,
    LatentStraddle,
    MarginUncertainty,
    PredictiveEntropy,
    ProbabilityVariance,
)
from robotorchan.models.classification.binary.standard.single_task import (
    BinarySingleTaskGPClassifier,
)


@pytest.mark.parametrize(
    "acquisition_class",
    [BALD, LatentStraddle, MarginUncertainty, PredictiveEntropy, ProbabilityVariance],
)
def test_classification_acquisition_has_finite_candidate_gradient(
    acquisition_class: type,
) -> None:
    torch.manual_seed(0)
    train_x = torch.linspace(0.0, 1.0, 8, dtype=torch.double).unsqueeze(-1)
    train_y = (train_x.squeeze(-1) > 0.5).to(dtype=torch.double)
    model = BinarySingleTaskGPClassifier(train_x, train_y, inducing_points=4)
    model.eval()
    acquisition = acquisition_class(model)
    candidates = torch.tensor([[[0.3]], [[0.7]]], dtype=torch.double, requires_grad=True)

    values = acquisition(candidates)
    gradient = torch.autograd.grad(values.sum(), candidates)[0]

    assert values.shape == torch.Size([2])
    assert gradient.shape == candidates.shape
    assert torch.isfinite(values).all()
    assert torch.isfinite(gradient).all()
