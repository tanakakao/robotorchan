"""Classification acquisition contracts for Sobol candidate search."""

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
from robotorchan.optim import SobolSearchStrategy


@pytest.mark.parametrize(
    "acquisition_class",
    [BALD, LatentStraddle, MarginUncertainty, PredictiveEntropy, ProbabilityVariance],
)
def test_binary_classification_sobol_search_q_one(acquisition_class: type) -> None:
    torch.manual_seed(0)
    train_x = torch.linspace(0.0, 1.0, 8, dtype=torch.double).unsqueeze(-1)
    train_y = (train_x.squeeze(-1) > 0.5).to(dtype=torch.double)
    model = BinarySingleTaskGPClassifier(train_x, train_y, inducing_points=4)
    model.eval()
    acquisition = acquisition_class(model)
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    result = SobolSearchStrategy(bounds, num_samples=16, seed=7).optimize(
        acquisition,
        q=1,
    )

    assert result.candidates.shape == torch.Size([1, 1])
    assert torch.isfinite(result.candidates).all()
    assert result.acquisition_value is not None
    assert torch.isfinite(result.acquisition_value).all()
    assert ((result.candidates >= bounds[0]) & (result.candidates <= bounds[1])).all()
    assert result.metadata["sampler"] == "sobol"
