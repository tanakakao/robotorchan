"""Registered MC acquisitions with joint q-batch optimizers."""

import pytest
import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.acquisition.monte_carlo import qSimpleRegret, qUpperConfidenceBound
from botorch.models import SingleTaskGP
from botorch.sampling.normal import SobolQMCNormalSampler

from robotorchan.optim.backends import (
    optimize_acqf_botorch,
    optimize_acqf_ga,
    optimize_acqf_sampling,
)


@pytest.mark.parametrize("q", [2, 4])
@pytest.mark.parametrize("optimizer_name", ["botorch", "ga", "sobol"])
@pytest.mark.parametrize(
    "acquisition_name",
    ["qLogExpectedImprovement", "qSimpleRegret", "qUpperConfidenceBound"],
)
def test_registered_joint_q_batch(
    q: int, optimizer_name: str, acquisition_name: str
) -> None:
    train_x = torch.linspace(0, 1, 8, dtype=torch.double).unsqueeze(-1)
    train_y = 1 - (train_x - 0.7).square()
    model = SingleTaskGP(train_x, train_y)
    model.eval()
    sampler = SobolQMCNormalSampler(torch.Size([16]), seed=7)
    acquisitions = {
        "qLogExpectedImprovement": lambda: qLogExpectedImprovement(
            model, best_f=train_y.max(), sampler=sampler
        ),
        "qSimpleRegret": lambda: qSimpleRegret(model, sampler=sampler),
        "qUpperConfidenceBound": lambda: qUpperConfidenceBound(
            model, beta=0.1, sampler=sampler
        ),
    }
    optimizers = {
        "botorch": (optimize_acqf_botorch, {"num_restarts": 2, "raw_samples": 16}),
        "ga": (optimize_acqf_ga, {"population_size": 12, "generations": 2, "seed": 7}),
        "sobol": (
            optimize_acqf_sampling,
            {"num_samples": 16, "method": "sobol", "seed": 7},
        ),
    }
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    optimizer, options = optimizers[optimizer_name]
    acquisition = acquisitions[acquisition_name]()
    candidates, value = optimizer(acquisition, bounds, q=q, **options)
    assert candidates.shape == (q, 1)
    assert value.numel() == 1
    assert torch.isfinite(candidates).all()
    assert torch.isfinite(value).all()
    assert torch.all(candidates >= bounds[0])
    assert torch.all(candidates <= bounds[1])
    with torch.no_grad():
        reevaluated = acquisition(candidates)
    torch.testing.assert_close(value.reshape(()), reevaluated.reshape(()), atol=1e-5, rtol=1e-4)
