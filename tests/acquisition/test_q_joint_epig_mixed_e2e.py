"""Joint qEPIG with independently enumerated categorical batch assignments."""

from itertools import product

import torch
from botorch.acquisition import AcquisitionFunction
from botorch.models import MixedSingleTaskGP
from botorch.optim import optimize_acqf

from robotorchan.acquisition import qJointExpectedPredictiveInformationGain


class FlattenedBatchAcquisition(AcquisitionFunction):
    """Adapt a fixed-size joint batch to flattened continuous optimization."""

    def __init__(self, acquisition: AcquisitionFunction, q: int, d: int) -> None:
        super().__init__(model=acquisition.model)
        self.acquisition = acquisition
        self.q = q
        self.d = d

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.acquisition(X.reshape(*X.shape[:-2], self.q, self.d))


def test_joint_epig_optimizes_batch_with_independent_categories() -> None:
    train_X = torch.tensor(
        [
            [0.0, 0.0],
            [0.5, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 1.0],
            [1.0, 1.0],
        ],
        dtype=torch.double,
    )
    train_Y = torch.sin(4.0 * train_X[:, :1]) + 0.3 * train_X[:, 1:]
    model = MixedSingleTaskGP(train_X, train_Y, cat_dims=[1])
    model.eval()
    target_X = torch.tensor([[0.25, 0.0], [0.75, 1.0]], dtype=torch.double)
    acquisition = qJointExpectedPredictiveInformationGain(model, target_X, observation_noise=1e-3)
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
    flat_acquisition = FlattenedBatchAcquisition(acquisition, q=2, d=2)
    flat_bounds = bounds.repeat(1, 2)
    results = [
        optimize_acqf(
            acq_function=flat_acquisition,
            bounds=flat_bounds,
            q=1,
            num_restarts=2,
            raw_samples=16,
            fixed_features={1: categories[0], 3: categories[1]},
            options={"maxiter": 30},
        )
        for categories in product((0.0, 1.0), repeat=2)
    ]
    flat_candidates, value = max(results, key=lambda result: result[1].item())
    candidates = flat_candidates.reshape(2, 2)
    assert candidates.shape == (2, 2)
    assert torch.isfinite(value).all()
    assert ((candidates[:, 0] >= 0) & (candidates[:, 0] <= 1)).all()
    assert ((candidates[:, 1] == 0) | (candidates[:, 1] == 1)).all()
    assert torch.isfinite(acquisition(candidates.unsqueeze(0))).all()
