"""Classification acquisition contracts for the CMA-ES backend."""

import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
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
from robotorchan.optim.backends import optimize_acqf_cmaes


class _FakeCMA:
    population_size = 2

    def __init__(self, *, mean, sigma, **kwargs):
        self.mean = mean
        self._index = 0

    def ask(self):
        point = np.full_like(self.mean, (0.25, 0.75)[self._index % 2])
        self._index += 1
        return point

    def tell(self, solutions):
        pass

    def should_stop(self):
        return True


@pytest.mark.parametrize(
    "acquisition_class",
    [BALD, LatentStraddle, MarginUncertainty, PredictiveEntropy, ProbabilityVariance],
)
def test_binary_classification_cmaes_q_one(acquisition_class: type) -> None:
    torch.manual_seed(0)
    train_x = torch.linspace(0.0, 1.0, 8, dtype=torch.double).unsqueeze(-1)
    train_y = (train_x.squeeze(-1) > 0.5).to(dtype=torch.double)
    model = BinarySingleTaskGPClassifier(train_x, train_y, inducing_points=4)
    model.eval()
    acquisition = acquisition_class(model)
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)

    with patch.dict(sys.modules, {"cmaes": SimpleNamespace(CMA=_FakeCMA)}):
        candidates, value = optimize_acqf_cmaes(
            acquisition,
            bounds,
            q=1,
            max_generations=1,
            seed=7,
        )

    assert candidates.shape == torch.Size([1, 1])
    assert torch.isfinite(candidates).all()
    assert torch.isfinite(value).all()
    assert ((candidates >= bounds[0]) & (candidates <= bounds[1])).all()
