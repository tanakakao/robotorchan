"""BoTorch optimizer smoke tests for binary classification acquisitions."""

import pytest
import torch
from botorch.optim import optimize_acqf

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
def test_classification_acquisition_optimize_acqf_q_one(acquisition_class: type) -> None:
    torch.manual_seed(0)
    train_x = torch.linspace(0.0, 1.0, 8, dtype=torch.double).unsqueeze(-1)
    train_y = (train_x.squeeze(-1) > 0.5).to(dtype=torch.double)
    model = BinarySingleTaskGPClassifier(train_x, train_y, inducing_points=4)
    model.eval()
    acquisition = acquisition_class(model)
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)

    candidates, value = optimize_acqf(
        acq_function=acquisition,
        bounds=bounds,
        q=1,
        num_restarts=2,
        raw_samples=8,
        options={"maxiter": 10},
    )

    assert candidates.shape == torch.Size([1, 1])
    assert torch.isfinite(candidates).all()
    assert torch.isfinite(value).all()
    assert ((candidates >= bounds[0]) & (candidates <= bounds[1])).all()
