"""Runtime forward contracts for classification active-learning acquisitions."""

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
    "acquisition_type",
    [BALD, LatentStraddle, MarginUncertainty, PredictiveEntropy, ProbabilityVariance],
)
def test_binary_gp_classification_acquisition_forward(acquisition_type: type) -> None:
    train_x = torch.linspace(0.0, 1.0, 8, dtype=torch.double).unsqueeze(-1)
    train_y = (train_x.squeeze(-1) > 0.5).to(dtype=torch.double)
    model = BinarySingleTaskGPClassifier(train_x, train_y, inducing_points=4)
    model.eval()
    candidates = torch.tensor([[[0.2]], [[0.8]]], dtype=torch.double)
    kwargs = {"num_samples": 8} if acquisition_type in (BALD, ProbabilityVariance) else {}
    acquisition = acquisition_type(model, **kwargs)

    with torch.no_grad():
        scores = acquisition(candidates)

    assert scores.shape == torch.Size([2])
    assert torch.isfinite(scores).all()


@pytest.mark.parametrize(
    "acquisition_type",
    [BALD, LatentStraddle, MarginUncertainty, PredictiveEntropy, ProbabilityVariance],
)
def test_binary_gp_classification_acquisition_rejects_q_two(acquisition_type: type) -> None:
    train_x = torch.linspace(0.0, 1.0, 8, dtype=torch.double).unsqueeze(-1)
    train_y = (train_x.squeeze(-1) > 0.5).to(dtype=torch.double)
    model = BinarySingleTaskGPClassifier(train_x, train_y, inducing_points=4)
    model.eval()
    kwargs = {"num_samples": 8} if acquisition_type in (BALD, ProbabilityVariance) else {}
    acquisition = acquisition_type(model, **kwargs)
    candidates = torch.tensor([[[0.2], [0.8]]], dtype=torch.double)

    with pytest.raises(ValueError, match="q=1"):
        acquisition(candidates)
